# -*- coding: utf-8 -*-
"""VC-LAD preprocessing, training, calibrated scoring and local metrics.

The shared reconstruction architecture is defined in model.py.
The public per-vehicle training and inference entry point is run.py.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .model import VCLAD

# Compatibility name for checkpoints/scripts from the standalone experiments.
ReconstructionGRU = VCLAD


CONTINUOUS_COLUMNS = [
    "车速",
    "总电压",
    "总电流",
    "SOC",
    "绝缘电阻",
    "电池单体电压最高值",
    "电池单体电压最低值",
    "最高温度值",
    "最低温度值",
]
SEGMENT_COLUMN = "segment_id"
FULL_SEQUENCE_SEGMENT = "__full_sequence__"
TIME_LIKE_COLUMNS = {"数据时间", "timestamp", "time", "row_index", "label", "fault_type"}
METRICS = ["A-ROC", "A-PR", "V-ROC", "V-PR", "F1", "R-F1", "Aff-F1"]


@dataclass(frozen=True)
class TrainSettings:
    window_size: int = 30
    stride: int = 1
    hidden_dim: int = 48
    gru_layers: int = 1
    dropout: float = 0.1
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    w_p: float = 0.8
    w_s: float = 0.2


@dataclass(frozen=True)
class EvalSettings:
    batch_size: int = 128
    score_mode: str = "last"
    smooth_window: int = 7
    score_clip: Optional[float] = 20.0
    w_p: float = 0.8
    w_s: float = 0.2
    threshold_points: int = 200
    vus_tolerance_range: Tuple[int, int] = (0, 25)
    vus_num_tolerances: int = 26
    affiliation_theta: int = 30


class MinMaxScaler:
    """Small serializable feature scaler with the paper's [0.01, 0.99] range."""

    def __init__(self, feature_range: Tuple[float, float] = (0.01, 0.99)):
        self.feature_range = feature_range
        self.data_min_: Optional[np.ndarray] = None
        self.data_max_: Optional[np.ndarray] = None

    def fit(self, values: np.ndarray) -> "MinMaxScaler":
        values = np.asarray(values, dtype=np.float32)
        self.data_min_ = np.nanmin(values, axis=0)
        self.data_max_ = np.nanmax(values, axis=0)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.data_min_ is None or self.data_max_ is None:
            raise RuntimeError("scaler has not been fitted")
        values = np.asarray(values, dtype=np.float32)
        low, high = self.feature_range
        denominator = self.data_max_ - self.data_min_
        denominator = np.where(np.abs(denominator) < 1e-8, 1.0, denominator)
        scaled = (values - self.data_min_) / denominator
        return (scaled * (high - low) + low).astype(np.float32)

    def state_dict(self) -> Dict[str, Any]:
        if self.data_min_ is None or self.data_max_ is None:
            raise RuntimeError("scaler has not been fitted")
        return {
            "feature_range": tuple(float(x) for x in self.feature_range),
            "data_min": self.data_min_.astype(np.float32),
            "data_max": self.data_max_.astype(np.float32),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "MinMaxScaler":
        scaler = cls(tuple(state["feature_range"]))
        scaler.data_min_ = np.asarray(state["data_min"], dtype=np.float32)
        scaler.data_max_ = np.asarray(state["data_max"], dtype=np.float32)
        return scaler




def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False


def discover_vehicles(data_path: Path, sample_mode: str) -> List[str]:
    train_dir = Path(data_path) / sample_mode / "train"
    return sorted(path.stem.replace("train_", "") for path in train_dir.glob("train_*.csv"))


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")


def _clean_continuous(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing continuous columns: {missing}")
    values = frame[list(columns)].copy()
    values = values.replace([254, 255, 65534, 65535, "0xFE", "0xFF"], np.nan)
    values = values.apply(pd.to_numeric, errors="coerce")
    values = values.interpolate(limit_direction="both").ffill().bfill().fillna(0)
    return values.to_numpy(dtype=np.float32)


def _make_windows(values: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    windows = [
        values[start : start + window_size]
        for start in range(0, len(values) - window_size + 1, stride)
    ]
    return np.asarray(windows, dtype=np.float32)


def _groups(frame: pd.DataFrame):
    if SEGMENT_COLUMN in frame.columns:
        return frame.groupby(SEGMENT_COLUMN, sort=False).groups.items()
    return [(FULL_SEQUENCE_SEGMENT, frame.index)]


def _training_windows(
    frame: pd.DataFrame,
    scaler: MinMaxScaler,
    window_size: int,
    stride: int,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    batches: List[np.ndarray] = []
    segment_info: List[Dict[str, Any]] = []
    for segment_id, indices in _groups(frame):
        positions = np.asarray(list(indices), dtype=int)
        segment = frame.iloc[positions]
        if len(segment) < window_size:
            segment_info.append(
                {"segment_id": str(segment_id), "rows": len(segment), "windows": 0, "skipped": True}
            )
            continue
        values = scaler.transform(_clean_continuous(segment, CONTINUOUS_COLUMNS))
        windows = _make_windows(values, window_size, stride)
        batches.append(windows)
        segment_info.append(
            {
                "segment_id": str(segment_id),
                "rows": len(segment),
                "windows": len(windows),
                "skipped": False,
            }
        )
    if not batches:
        raise ValueError(f"no segment is long enough for window_size={window_size}")
    return np.concatenate(batches).astype(np.float32), segment_info


def _robust_stats(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"median": 0.0, "iqr": 1.0, "q25": 0.0, "q75": 1.0, "q90": 1.0, "q95": 1.0}
    q25, median, q75, q90, q95 = np.percentile(array, [25, 50, 75, 90, 95])
    scale = max(float(q75 - q25), float(q90 - median), 1e-8)
    return {
        "median": float(median),
        "iqr": scale,
        "q25": float(q25),
        "q75": float(q75),
        "q90": float(q90),
        "q95": float(q95),
    }


def _raw_score_components(model: VCLAD, values: torch.Tensor):
    reconstructed = model(values)["x_recon"]
    p_time = torch.mean((reconstructed - values) ** 2, dim=-1)
    if values.size(1) > 1:
        differences = values[:, 1:, :] - values[:, :-1, :]
        reconstructed_differences = reconstructed[:, 1:, :] - reconstructed[:, :-1, :]
        s_short = torch.mean((reconstructed_differences - differences) ** 2, dim=-1)
        s_time = F.pad(s_short, (1, 0), value=0.0)
    else:
        s_time = torch.zeros_like(p_time)
    return p_time, s_time


def _calibrate(
    values: torch.Tensor,
    stats: Dict[str, float],
    score_clip: Optional[float],
) -> torch.Tensor:
    center = float(stats.get("median", 0.0))
    scale = max(float(stats.get("iqr", stats.get("scale", 1.0))), 1e-8)
    calibrated = F.relu((values - center) / scale)
    if score_clip is not None:
        calibrated = torch.clamp(calibrated, max=float(score_clip))
    return calibrated


def _reduce_time(values: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "last":
        return values[:, -1]
    if mode == "mean":
        return values.mean(dim=1)
    if mode == "max":
        return values.max(dim=1).values
    raise ValueError(f"unknown score mode: {mode}")


@torch.no_grad()
def _score_batch(
    model: VCLAD,
    values: torch.Tensor,
    score_stats: Dict[str, Dict[str, float]],
    score_clip: Optional[float],
    score_mode: str,
    w_p: float,
    w_s: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    p_time, s_time = _raw_score_components(model, values)
    total_time = (
        w_p * _calibrate(p_time, score_stats.get("p", {}), score_clip)
        + w_s * _calibrate(s_time, score_stats.get("s", {}), score_clip)
    )
    return _reduce_time(total_time, score_mode), _reduce_time(p_time, score_mode), _reduce_time(s_time, score_mode)


@torch.no_grad()
def _compute_score_stats(
    model: VCLAD,
    windows: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    loader = DataLoader(TensorDataset(torch.from_numpy(windows)), batch_size=batch_size, shuffle=False)
    p_values: List[float] = []
    s_values: List[float] = []
    model.eval()
    for (batch,) in loader:
        p_time, s_time = _raw_score_components(model, batch.to(device))
        p_values.extend(p_time[:, -1].cpu().numpy().tolist())
        s_values.extend(s_time[:, -1].cpu().numpy().tolist())
    return {"p": _robust_stats(p_values), "s": _robust_stats(s_values)}


def checkpoint_path(model_dir: Path, vehicle_id: str) -> Path:
    return Path(model_dir) / f"model_{vehicle_id}.pt"


def history_path(model_dir: Path, vehicle_id: str) -> Path:
    return Path(model_dir) / f"history_{vehicle_id}.csv"


def train_vehicle(
    data_path: Path,
    sample_mode: str,
    vehicle_id: str,
    model_dir: Path,
    settings: TrainSettings,
    device: torch.device,
    force: bool = False,
) -> None:
    """Train one vehicle model and save a standalone checkpoint."""
    destination = checkpoint_path(model_dir, vehicle_id)
    if destination.exists() and not force:
        return

    source = Path(data_path) / sample_mode / "train" / f"train_{vehicle_id}.csv"
    if not source.exists():
        raise FileNotFoundError(source)
    frame = _read_csv(source)
    raw = _clean_continuous(frame, CONTINUOUS_COLUMNS)
    if len(raw) < settings.window_size:
        raise ValueError(f"vehicle {vehicle_id}: only {len(raw)} training rows")

    scaler = MinMaxScaler().fit(raw)
    windows, segment_info = _training_windows(frame, scaler, settings.window_size, settings.stride)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(windows)),
        batch_size=settings.batch_size,
        shuffle=True,
        drop_last=False,
    )
    model = VCLAD(
        cont_dim=len(CONTINUOUS_COLUMNS),
        hidden_dim=settings.hidden_dim,
        gru_layers=settings.gru_layers,
        dropout=settings.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )

    history: List[Dict[str, float]] = []
    print(f"Training vehicle {vehicle_id}: {len(windows)} windows")
    for epoch in range(1, settings.epochs + 1):
        model.train()
        losses: List[float] = []
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstruction = model(batch)["x_recon"]
            loss = F.mse_loss(reconstruction, batch)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if not losses:
            raise RuntimeError(f"vehicle {vehicle_id}: no finite losses in epoch {epoch}")
        mean_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "recon_loss": mean_loss})
        print(f"  epoch {epoch:03d}/{settings.epochs}: recon_loss={mean_loss:.7f}")

    score_stats = _compute_score_stats(model, windows, settings.batch_size, device)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "model_config": {
            "cont_dim": len(CONTINUOUS_COLUMNS),
            "hidden_dim": settings.hidden_dim,
            "gru_layers": settings.gru_layers,
            "dropout": settings.dropout,
            "window_size": settings.window_size,
            "stride": settings.stride,
        },
        "score_stats": score_stats,
        "score_weights": {"w_p": settings.w_p, "w_s": settings.w_s},
        "scaler": scaler.state_dict(),
        "continuous_columns": CONTINUOUS_COLUMNS,
        "segment_info": segment_info,
        "training_settings": settings.__dict__,
        "history": history,
    }
    torch.save(checkpoint, destination)
    pd.DataFrame(history).to_csv(history_path(model_dir, vehicle_id), index=False)


def _read_labels(data_path: Path, sample_mode: str, vehicle_id: str, expected_len: int):
    path = Path(data_path) / sample_mode / "label" / f"test_{vehicle_id}_labels.csv"
    if not path.exists():
        raise FileNotFoundError(f"label file not found: {path}")
    frame = pd.read_csv(path)
    if "fault_type" in frame.columns:
        values = pd.to_numeric(frame["fault_type"], errors="coerce").fillna(0).to_numpy()
    elif "label" in frame.columns:
        values = pd.to_numeric(frame["label"], errors="coerce").fillna(0).to_numpy()
    else:
        candidates = [
            column
            for column in frame.columns
            if column not in TIME_LIKE_COLUMNS and column != SEGMENT_COLUMN
        ]
        if not candidates:
            raise ValueError(f"no label column in {path}")
        values = pd.to_numeric(frame[candidates[0]], errors="coerce").fillna(0).to_numpy()
    if len(values) != expected_len:
        raise ValueError(f"label/test length mismatch: {len(values)} != {expected_len}")
    return (values > 0).astype(int)


def _test_windows(
    data_path: Path,
    sample_mode: str,
    vehicle_id: str,
    scaler: MinMaxScaler,
    continuous_columns: Sequence[str],
    window_size: int,
    stride: int,
):
    path = Path(data_path) / sample_mode / "test" / f"test_{vehicle_id}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = _read_csv(path)
    labels = _read_labels(data_path, sample_mode, vehicle_id, len(frame))
    batches: List[np.ndarray] = []
    label_batches: List[np.ndarray] = []
    end_indices: List[int] = []
    segment_ids: List[str] = []
    end_times: List[str] = []

    for segment_id, indices in _groups(frame):
        positions = np.asarray(list(indices), dtype=int)
        segment = frame.iloc[positions]
        if len(segment) < window_size:
            continue
        values = scaler.transform(_clean_continuous(segment, continuous_columns))
        batches.append(_make_windows(values, window_size, stride))
        local_ends = np.arange(window_size - 1, len(segment), stride, dtype=int)
        global_ends = positions[local_ends]
        label_batches.append(labels[global_ends])
        end_indices.extend(global_ends.tolist())
        segment_ids.extend([str(segment_id)] * len(global_ends))
        if "数据时间" in frame.columns:
            end_times.extend(frame.iloc[global_ends]["数据时间"].astype(str).tolist())
        else:
            end_times.extend([""] * len(global_ends))

    if not batches:
        raise ValueError(f"vehicle {vehicle_id}: no test segment supports window_size={window_size}")
    return (
        np.concatenate(batches).astype(np.float32),
        np.concatenate(label_batches).astype(int),
        {
            "row_index": np.asarray(end_indices, dtype=int),
            "segment_id": np.asarray(segment_ids, dtype=object),
            "data_time": np.asarray(end_times, dtype=object),
        },
    )


@torch.no_grad()
def _predict(
    model: VCLAD,
    windows: np.ndarray,
    score_stats: Dict[str, Dict[str, float]],
    settings: EvalSettings,
    device: torch.device,
):
    loader = DataLoader(TensorDataset(torch.from_numpy(windows)), batch_size=settings.batch_size, shuffle=False)
    total_values: List[float] = []
    p_values: List[float] = []
    s_values: List[float] = []
    for (batch,) in loader:
        total, p_score, s_score = _score_batch(
            model,
            batch.to(device),
            score_stats,
            settings.score_clip,
            settings.score_mode,
            settings.w_p,
            settings.w_s,
        )
        total_values.extend(total.cpu().numpy().tolist())
        p_values.extend(p_score.cpu().numpy().tolist())
        s_values.extend(s_score.cpu().numpy().tolist())
    return tuple(np.asarray(values, dtype=np.float64) for values in (total_values, p_values, s_values))


def _component_clip_rate(values: np.ndarray, stats: Dict[str, float], clip: Optional[float]) -> float:
    if clip is None or not stats or len(values) == 0:
        return float("nan")
    center = float(stats.get("median", 0.0))
    scale = max(float(stats.get("iqr", stats.get("scale", 1.0))), 1e-8)
    calibrated = np.maximum((values - center) / scale, 0.0)
    return float(np.mean(calibrated > float(clip)))


def _smooth(values: np.ndarray, window: int, segment_ids: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if window <= 1 or len(values) == 0:
        return values
    output = np.empty_like(values)
    helper = pd.DataFrame({"segment_id": segment_ids, "position": np.arange(len(values))})
    for _, positions in helper.groupby("segment_id", sort=False)["position"]:
        positions = positions.to_numpy(dtype=int)
        output[positions] = (
            pd.Series(values[positions]).rolling(window, min_periods=1, center=True).mean().to_numpy()
        )
    return output


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if len(values) == 0:
        return values
    low, high = np.percentile(values, [1, 99])
    values = np.clip(values, low, high)
    minimum, maximum = values.min(), values.max()
    if maximum > minimum:
        return (values - minimum) / (maximum - minimum)
    return np.full_like(values, 0.5)


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    positive = y_true == 1
    negative = y_true == 0
    n_positive, n_negative = int(positive.sum()), int(negative.sum())
    if n_positive == 0 or n_negative == 0:
        return 0.5
    order = np.argsort(y_score)
    sorted_scores = y_score[order]
    ranks = np.empty(len(y_score), dtype=np.float64)
    start = 0
    while start < len(y_score):
        end = start + 1
        while end < len(y_score) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = ranks[positive].sum()
    return float((rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative))


def average_precision_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Non-interpolated binary AP, evaluated once per distinct score threshold.

    Equal scores form one threshold group, so their row order cannot change AP.
    """
    labels = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("labels and scores must be equal-length one-dimensional arrays")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("AP requires binary labels and finite scores")
    positives = int(np.count_nonzero(labels == 1))
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_positive = labels[order] == 1
    group_ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    true_positive = np.cumsum(sorted_positive)[group_ends]
    precision = true_positive / (group_ends + 1)
    recall = true_positive / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_positive = float(((y_true == 1) & (y_pred == 1)).sum())
    false_positive = float(((y_true == 0) & (y_pred == 1)).sum())
    false_negative = float(((y_true == 1) & (y_pred == 0)).sum())
    denominator = 2.0 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2.0 * true_positive / denominator


def _ranges(labels: np.ndarray) -> List[Tuple[int, int]]:
    result: List[Tuple[int, int]] = []
    position = 0
    while position < len(labels):
        if labels[position] != 1:
            position += 1
            continue
        start = position
        while position < len(labels) and labels[position] == 1:
            position += 1
        result.append((start, position - 1))
    return result


def _thresholds(scores: np.ndarray, points: int) -> np.ndarray:
    if len(scores) == 0:
        return np.array([0.0])
    minimum, maximum = float(scores.min()), float(scores.max())
    return np.array([minimum]) if maximum <= minimum else np.linspace(minimum, maximum, points)


def _iou(first: Tuple[int, int], second: Tuple[int, int]) -> float:
    intersection = max(0, min(first[1], second[1]) - max(first[0], second[0]) + 1)
    union = (first[1] - first[0] + 1) + (second[1] - second[0] + 1) - intersection
    return intersection / union if union > 0 else 0.0


def _range_precision(predicted: List[Tuple[int, int]], truth: List[Tuple[int, int]]) -> float:
    if not predicted:
        return 0.0
    scores = []
    for prediction in predicted:
        overlaps = [item for item in truth if item[1] >= prediction[0] and item[0] <= prediction[1]]
        scores.append(0.0 if not overlaps else sum(_iou(prediction, item) for item in overlaps) / len(overlaps))
    return float(np.mean(scores))


def _range_recall(predicted: List[Tuple[int, int]], truth: List[Tuple[int, int]]) -> float:
    if not truth:
        return 0.0
    scores = []
    for actual in truth:
        overlaps = [item for item in predicted if item[1] >= actual[0] and item[0] <= actual[1]]
        if not overlaps:
            scores.append(0.0)
            continue
        covered_start = min(item[0] for item in overlaps)
        covered_end = max(item[1] for item in overlaps)
        overlap_reward = (covered_end - covered_start + 1) / (actual[1] - actual[0] + 1)
        scores.append((0.5 + 0.5 * overlap_reward) / len(overlaps))
    return float(np.mean(scores))


def range_f1(y_true: np.ndarray, scores: np.ndarray, points: int) -> float:
    truth = _ranges(y_true)
    best = 0.0
    for threshold in _thresholds(scores, points):
        predicted = _ranges((scores >= threshold).astype(int))
        precision, recall = _range_precision(predicted, truth), _range_recall(predicted, truth)
        if precision + recall > 0:
            best = max(best, 2.0 * precision * recall / (precision + recall))
    return best


def affiliation_f1(y_true: np.ndarray, scores: np.ndarray, theta: int, points: int) -> float:
    truth = _ranges(y_true)
    if not truth:
        return 0.0
    best = 0.0
    for threshold in _thresholds(scores, points):
        predicted_points = np.where(scores >= threshold)[0]
        if len(predicted_points) == 0:
            continue
        precision = np.mean(
            [
                np.exp(
                    -(min(0 if start <= point <= end else min(abs(point - start), abs(point - end)) for start, end in truth) ** 2)
                    / (2 * theta**2)
                )
                for point in predicted_points
            ]
        )
        recall = np.mean(
            [
                np.exp(
                    -(min(0 if start <= point <= end else min(abs(point - start), abs(point - end)) for point in predicted_points) ** 2)
                    / (2 * theta**2)
                )
                for start, end in truth
            ]
        )
        if precision + recall > 0:
            best = max(best, float(2 * precision * recall / (precision + recall)))
    return best


def vus_roc_pr(
    y_true: np.ndarray,
    scores: np.ndarray,
    tolerance_range: Tuple[int, int],
    num_tolerances: int,
) -> Tuple[float, float]:
    """Paper V-ROC/V-PR: tolerance-averaged label-dilation AUC/AP, not official VUS."""
    truth = _ranges(y_true)
    roc_values, pr_values = [], []
    for tolerance in np.linspace(tolerance_range[0], tolerance_range[1], num_tolerances):
        adjusted = np.zeros_like(y_true)
        for start, end in truth:
            adjusted[max(0, start - int(tolerance)) : min(len(y_true), end + int(tolerance) + 1)] = 1
        if len(np.unique(adjusted)) < 2:
            roc_values.append(0.5)
            pr_values.append(0.5)
        else:
            roc_values.append(roc_auc_score(adjusted, scores))
            pr_values.append(average_precision_score(adjusted, scores))
    return float(np.mean(roc_values)), float(np.mean(pr_values))


def evaluate_metrics(y_true: np.ndarray, raw_scores: np.ndarray, normalized_scores: np.ndarray, settings: EvalSettings):
    if len(np.unique(y_true)) < 2:
        metrics = {"A-ROC": 0.5, "A-PR": 0.5, "V-ROC": 0.5, "V-PR": 0.5}
    else:
        metrics = {
            "A-ROC": roc_auc_score(y_true, raw_scores),
            "A-PR": average_precision_score(y_true, raw_scores),
        }
        metrics["V-ROC"], metrics["V-PR"] = vus_roc_pr(
            y_true, raw_scores, settings.vus_tolerance_range, settings.vus_num_tolerances
        )
    thresholds = _thresholds(normalized_scores, settings.threshold_points)
    metrics["F1"] = max(_f1(y_true, (normalized_scores >= value).astype(int)) for value in thresholds)
    metrics["R-F1"] = range_f1(y_true, normalized_scores, settings.threshold_points)
    metrics["Aff-F1"] = affiliation_f1(
        y_true, normalized_scores, settings.affiliation_theta, settings.threshold_points
    )
    return metrics


def evaluate_vehicle(
    data_path: Path,
    sample_mode: str,
    vehicle_id: str,
    model_dir: Path,
    settings: EvalSettings,
    device: torch.device,
):
    """Evaluate one standalone checkpoint and return metrics, scores, clip rates."""
    path = checkpoint_path(model_dir, vehicle_id)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    config = checkpoint["model_config"]
    model = VCLAD(
        cont_dim=int(config["cont_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        gru_layers=int(config["gru_layers"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    scaler = MinMaxScaler.from_state_dict(checkpoint["scaler"])
    windows, labels, meta = _test_windows(
        data_path,
        sample_mode,
        vehicle_id,
        scaler,
        checkpoint["continuous_columns"],
        int(config["window_size"]),
        int(config["stride"]),
    )
    raw_scores, p_scores_raw, s_scores_raw = _predict(
        model, windows, checkpoint["score_stats"], settings, device
    )
    clip_rates = {
        "P_clip_rate": _component_clip_rate(
            p_scores_raw, checkpoint["score_stats"].get("p", {}), settings.score_clip
        ),
        "S_clip_rate": _component_clip_rate(
            s_scores_raw, checkpoint["score_stats"].get("s", {}), settings.score_clip
        ),
    }
    segment_ids = meta["segment_id"]
    raw_scores = _smooth(raw_scores, settings.smooth_window, segment_ids)
    p_scores = _smooth(p_scores_raw, settings.smooth_window, segment_ids)
    s_scores = _smooth(s_scores_raw, settings.smooth_window, segment_ids)
    normalized = _normalize_scores(raw_scores)
    metrics = evaluate_metrics(labels, raw_scores, normalized, settings)
    result = pd.DataFrame(
        {
            "row_index": meta["row_index"],
            "segment_id": segment_ids,
            "data_time": meta["data_time"],
            "anomaly_score_raw": raw_scores,
            "anomaly_score_norm": normalized,
            "P_score": p_scores,
            "S_score": s_scores,
            "label": labels,
        }
    )
    return metrics, result, clip_rates
