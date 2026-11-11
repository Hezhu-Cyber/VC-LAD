"""VC-LAD cloud reconstruction model and calibrated P/S scoring.

Default architecture: nine features, hidden size 48, one GRU layer.
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class VCLAD(nn.Module):
    """VC-LAD cloud detector: feature projection, single-layer GRU, and reconstruction head."""

    def __init__(
        self,
        cont_dim: int = 9,
        hidden_dim: int = 48,
        gru_layers: int = 1,
        dropout: float = 0.1,
        window_size: int = 30,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.cont_dim = cont_dim
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers
        self.dropout = dropout
        self.cont_proj = nn.Sequential(
            nn.Linear(cont_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.recon_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, cont_dim),
        )

    def forward(self, values: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.cont_proj(values)
        hidden, _ = self.gru(hidden)
        return {"x_recon": self.recon_head(hidden), "hidden": hidden}


def reconstruction_loss(
    x_cont: torch.Tensor,
    out: Dict[str, torch.Tensor],
    lambda_recon: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean squared reconstruction loss over all time steps and variables."""
    x_recon = out["x_recon"]
    recon_loss = F.mse_loss(x_recon, x_cont)
    total = lambda_recon * recon_loss
    return total, recon_loss.detach()


def _calibrated_component(
    values: torch.Tensor,
    score_stats: Optional[Dict[str, Dict[str, float]]],
    key: str,
    clip: float = 20.0,
) -> torch.Tensor:
    """利用训练集正常数据统计量进行稳健校准"""
    if not score_stats or key not in score_stats:
        return values
    stats = score_stats.get(key, {})
    center = float(stats.get("median", 0.0))
    scale = max(float(stats.get("iqr", stats.get("scale", 1.0))), 1e-8)
    z = (values - center) / scale
    return torch.clamp(F.relu(z), max=clip)


def resolve_score_weights(weights=None, default_p=0.8, default_s=0.2):
    """Read normalized P/S weights; incompatible legacy metadata uses paper defaults.

    Explicit normalized ablation weights (including 1/0 and 0/1) remain valid.
    This does not rewrite checkpoints or establish provenance of earlier results.
    """
    import math
    import warnings
    weights = weights or {}
    wp = float(weights.get("w_p", default_p))
    ws = float(weights.get("w_s", default_s))
    if (not math.isfinite(wp) or not math.isfinite(ws) or wp < 0 or ws < 0
            or not math.isclose(wp + ws, 1.0, rel_tol=1e-7, abs_tol=1e-7)):
        warnings.warn(
            "Legacy/non-normalized detection weights replaced by paper defaults "
            f"({default_p}, {default_s}); rerun evaluation before comparing results.",
            RuntimeWarning, stacklevel=2,
        )
        return float(default_p), float(default_s)
    return wp, ws


@torch.no_grad()
def anomaly_scores(
    model: VCLAD,
    x_cont: torch.Tensor,
    score_mode: str = "last",
    w_p: float = 0.8,
    w_s: float = 0.2,
    score_stats: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, torch.Tensor]:
    """Combine calibrated value and temporal-change reconstruction scores."""
    w_p, w_s = resolve_score_weights({"w_p": w_p, "w_s": w_s})
    model.eval()
    out = model(x_cont)
    x_recon = out["x_recon"]

    # 1. 重构分数 P-score
    p_t = torch.mean((x_recon - x_cont) ** 2, dim=-1)

    # 2. 差分重构分数 S-score
    # 对首时间步填充零（而非复制后续值），避免引入伪值影响 mean/max 模式
    if x_cont.size(1) > 1:
        dx = x_cont[:, 1:, :] - x_cont[:, :-1, :]
        dx_hat = x_recon[:, 1:, :] - x_recon[:, :-1, :]
        s_t_short = torch.mean((dx_hat - dx) ** 2, dim=-1)  # [B, L-1]
        s_t = F.pad(s_t_short, (1, 0), value=0.0)           # [B, L], 首位填零
    else:
        s_t = torch.zeros_like(p_t)

    # 校准并加权组合
    p_mix = _calibrated_component(p_t, score_stats, "p")
    s_mix = _calibrated_component(s_t, score_stats, "s")
    total_t = w_p * p_mix + w_s * s_mix

    if score_mode == "last":
        score = total_t[:, -1]
        p_score = p_t[:, -1]
        s_score = s_t[:, -1]
    elif score_mode == "mean":
        score = total_t.mean(dim=1)
        p_score = p_t.mean(dim=1)
        s_score = s_t.mean(dim=1)
    elif score_mode == "max":
        score = total_t.max(dim=1).values
        p_score = p_t.max(dim=1).values
        s_score = s_t.max(dim=1).values
    else:
        raise ValueError(f"Unknown score_mode: {score_mode}")

    return {
        "score": score,
        "p_score": p_score,
        "s_score": s_score,
    }


# Legacy import names; all model imports share the same implementation.
HydraADLiteV2 = VCLAD
hydraad_lite_v2_loss = reconstruction_loss
anomaly_scores_v2 = anomaly_scores
