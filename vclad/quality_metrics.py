# -*- coding: utf-8 -*-
import logging
from typing import Optional

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GBT32960FieldRanges:
    """GB/T 32960.3-2016 vehicle data field range checks."""

    INVALID_MARKERS = {
        0xFF: "invalid",
        0xFE: "abnormal",
        255: "invalid",
        254: "abnormal",
    }

    FIELD_RANGES = {
        "最高报警等级": {"type": "discrete", "valid_values": [0, 1, 2, 3], "invalid_markers": [0xFF]},
        "车速": {"type": "continuous", "range": (0, 220), "invalid_markers": []},
        "车辆状态": {"type": "discrete", "valid_values": [1, 2, 3], "invalid_markers": [0xFE, 0xFF]},
        "充电状态": {"type": "discrete", "valid_values": [1, 2, 3, 4], "invalid_markers": [0xFE, 0xFF]},
        "运行模式": {"type": "discrete", "valid_values": [1, 2, 3], "invalid_markers": [0xFE, 0xFF]},
        "累计里程": {"type": "continuous", "range": (0, 999999.9), "invalid_markers": [0xFE, 0xFF]},
        "总电压": {"type": "continuous", "range": (0, 1000), "invalid_markers": [0xFE, 0xFF]},
        "总电流": {"type": "continuous", "range": (-1000, 1000), "invalid_markers": [0xFE, 0xFF]},
        "SOC": {"type": "continuous", "range": (0, 100), "invalid_markers": [0xFE, 0xFF]},
        "DC-DC状态": {"type": "discrete", "valid_values": [1, 2], "invalid_markers": [0xFE, 0xFF]},
        "挡位": {"type": "discrete", "valid_values": [0, 13, 14, 15]},
        "绝缘电阻": {"type": "continuous", "range": (0, 60000)},
        "经度": {"type": "continuous", "range": (-180, 180)},
        "纬度": {"type": "continuous", "range": (-90, 90)},
        "电池单体电压最高值": {"type": "continuous", "range": (0, 15), "invalid_markers": [0xFE, 0xFF]},
        "电池单体电压最低值": {"type": "continuous", "range": (0, 15), "invalid_markers": [0xFE, 0xFF]},
        "最高温度值": {"type": "continuous", "range": (-40, 210), "invalid_markers": [0xFE, 0xFF]},
        "最低温度值": {"type": "continuous", "range": (-40, 210), "invalid_markers": [0xFE, 0xFF]},
        "通用报警标志": {"type": "discrete", "valid_values": [0, 1]},
        "可充电储能装置故障总数": {"type": "continuous", "range": (0, 252), "invalid_markers": [0xFE, 0xFF]},
        "驱动电机故障总数": {"type": "continuous", "range": (0, 252), "invalid_markers": [0xFE, 0xFF]},
        "发动机故障总数": {"type": "continuous", "range": (0, 252), "invalid_markers": [0xFE, 0xFF]},
        "其他故障总数": {"type": "continuous", "range": (0, 252), "invalid_markers": [0xFE, 0xFF]},
    }

    @classmethod
    def check_value_validity(cls, field_name: str, series: pd.Series) -> pd.Series:
        if field_name not in cls.FIELD_RANGES:
            return pd.Series(True, index=series.index)

        config = cls.FIELD_RANGES[field_name]
        is_valid = series.notna()

        for marker in config.get("invalid_markers", []):
            is_valid &= series != marker

        if config["type"] == "continuous":
            min_v, max_v = config["range"]
            numeric_s = pd.to_numeric(series, errors="coerce")
            is_valid &= (numeric_s >= min_v) & (numeric_s <= max_v)
        elif config["type"] == "discrete":
            is_valid &= series.isin(set(config["valid_values"]))

        return is_valid.fillna(False)


def _numeric(df, col):
    return pd.to_numeric(df[col], errors="coerce")


def _check_speed_acceleration_row(df):
    if "车速" not in df.columns:
        return pd.Series(True, index=df.index)
    speeds = _numeric(df, "车速")
    speed_diff = speeds.diff()
    consistent = pd.Series(True, index=df.index)
    if "加速度" in df.columns:
        accels = _numeric(df, "加速度")
        rule_check = (np.sign(speed_diff) == np.sign(accels)) | (accels.abs() < 0.5)
        consistent &= rule_check.fillna(True)
    return consistent


def _check_driving_vehicle_status_row(df):
    if "车速" not in df.columns:
        return pd.Series(True, index=df.index)
    driving_mask = _numeric(df, "车速") > 0
    consistent = pd.Series(True, index=df.index)
    if "车辆状态" in df.columns:
        status = _numeric(df, "车辆状态")
        consistent &= ~(driving_mask & (status != 1))
    return consistent


def _check_driving_charge_status_row(df):
    if "车速" not in df.columns:
        return pd.Series(True, index=df.index)
    driving_mask = _numeric(df, "车速") > 0
    consistent = pd.Series(True, index=df.index)
    if "充电状态" in df.columns:
        charge = _numeric(df, "充电状态")
        consistent &= ~(driving_mask & ~charge.isin([2, 3, 4]))
    return consistent


def _check_driving_gear_row(df):
    if "车速" not in df.columns:
        return pd.Series(True, index=df.index)
    driving_mask = _numeric(df, "车速") > 0
    consistent = pd.Series(True, index=df.index)
    if "挡位" in df.columns:
        gear = _numeric(df, "挡位")
        consistent &= ~(driving_mask & gear.isin([0, 15]))
    return consistent


def _check_parking_logic_row(df):
    if "车速" not in df.columns:
        return pd.Series(True, index=df.index)
    speeds = _numeric(df, "车速")
    consistent = pd.Series(True, index=df.index)

    if "充电状态" in df.columns:
        charging_parked = _numeric(df, "充电状态") == 1
        consistent &= ~(charging_parked & (speeds > 0))

    if "挡位" in df.columns:
        gear = _numeric(df, "挡位")
        consistent &= ~((gear == 15) & (speeds > 0))

    return consistent


def _check_voltage_logic_row(df):
    consistent = pd.Series(True, index=df.index)
    if "电池单体电压最高值" in df.columns and "电池单体电压最低值" in df.columns:
        v_max = _numeric(df, "电池单体电压最高值")
        v_min = _numeric(df, "电池单体电压最低值")
        consistent &= v_max >= v_min
    return consistent


def _check_temperature_logic_row(df):
    consistent = pd.Series(True, index=df.index)
    if "最高温度值" in df.columns and "最低温度值" in df.columns:
        t_max = _numeric(df, "最高温度值")
        t_min = _numeric(df, "最低温度值")
        consistent &= t_max >= t_min
    return consistent


class QualityMetrics:
    """Window-level data quality metrics used by the GPC experiments."""

    def __init__(self, require_historical: bool = False, sigma_A: float = 1.0):
        self.field_ranges = GBT32960FieldRanges()

    def compute_accuracy(self, data: pd.DataFrame, alpha: float = 0.6) -> float:
        if data.empty:
            return 0.0
        row_is_accurate = pd.Series(True, index=data.index)
        checked = False
        for col in data.columns:
            if col in self.field_ranges.FIELD_RANGES:
                checked = True
                row_is_accurate &= self.field_ranges.check_value_validity(col, data[col])
        if not checked:
            return 0.0
        return float(np.clip(row_is_accurate.mean(), 0, 1))

    def compute_completeness(self, data: pd.DataFrame, ignore_cols: list = None) -> float:
        if data.empty:
            return 0.0

        target_df = data.copy()
        if ignore_cols:
            cols_to_drop = [c for c in ignore_cols if c in target_df.columns]
            if cols_to_drop:
                target_df = target_df.drop(columns=cols_to_drop)

        core_cols = [c for c in target_df.columns if c in self.field_ranges.FIELD_RANGES]
        time_cols = [c for c in target_df.columns if "时间" in c or "time" in c.lower()]
        check_cols = list(dict.fromkeys(core_cols + time_cols))
        if not check_cols:
            return 0.0

        complete_rows = target_df[check_cols].notna().all(axis=1)
        return float(np.clip(complete_rows.mean(), 0, 1))

    def compute_consistency(self, data: pd.DataFrame) -> float:
        if data.empty:
            return 0.0
        row_is_consistent = pd.Series(True, index=data.index)
        row_is_consistent &= _check_speed_acceleration_row(data)
        row_is_consistent &= _check_driving_vehicle_status_row(data)
        row_is_consistent &= _check_driving_charge_status_row(data)
        row_is_consistent &= _check_driving_gear_row(data)
        row_is_consistent &= _check_parking_logic_row(data)
        row_is_consistent &= _check_voltage_logic_row(data)
        row_is_consistent &= _check_temperature_logic_row(data)
        return float(np.clip(row_is_consistent.mean(), 0, 1))

    def compute_timeliness(
        self,
        data: pd.DataFrame,
        timestamps: Optional[np.ndarray] = None,
        reporting_interval: float = 10.0,
        network_tolerance: float = 5.0,
    ) -> float:
        if data.empty:
            return 0.0

        n = len(data)
        threshold = reporting_interval + network_tolerance
        if timestamps is None:
            time_cols = [c for c in data.columns
                         if ("时间" in c or "time" in c.lower()) and "score" not in c.lower()]
            if not time_cols:
                return 0.0
            # Preserve row alignment: an invalid timestamp remains an invalid row.
            times = pd.to_datetime(data[time_cols[0]], errors="coerce")
            valid = times.notna().to_numpy()
            intervals = times.diff().dt.total_seconds().to_numpy()[1:]
        else:
            times = np.asarray(timestamps, dtype=float)
            if times.ndim != 1 or len(times) != n:
                raise ValueError("timestamps must contain one elapsed-second value per row")
            valid = np.isfinite(times)
            intervals = np.diff(times)
        if n > 1:
            valid[1:] &= (np.isfinite(intervals) & (intervals >= 0.0)
                          & (intervals <= threshold))
        return float(valid.mean())


    def compute_reasonability(self, data, field_names=None, alpha=0.6):
        if isinstance(data, np.ndarray):
            if field_names is None:
                field_names = [f"feat_{i}" for i in range(data.shape[1])]
            data = pd.DataFrame(data, columns=field_names)
        return self.compute_accuracy(data)
