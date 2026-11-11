"""Prepare chronological healthy training data without crossing gaps or splits."""
import numpy as np
import pandas as pd
from .detector import CONTINUOUS_COLUMNS

def split_healthy(frame, minimum_records=500, fraction=0.7, maximum_gap=15):
    required=['数据时间','最高报警等级']+CONTINUOUS_COLUMNS
    missing=set(required)-set(frame.columns)
    if missing: raise ValueError(f'Missing fields: {sorted(missing)}')
    frame=frame.copy()
    frame['数据时间']=pd.to_datetime(frame['数据时间'],errors='coerce')
    frame=frame.dropna(subset=['数据时间']).sort_values('数据时间',kind='stable').reset_index(drop=True)
    healthy=pd.to_numeric(frame['最高报警等级'],errors='coerce').eq(0)
    gap=frame['数据时间'].diff().dt.total_seconds()
    boundary=gap.gt(maximum_gap)|gap.lt(0)|~healthy|~healthy.shift(fill_value=False)
    frame['segment_id']=boundary.cumsum()
    healthy_frame=frame.loc[healthy].copy()
    counts=healthy_frame.groupby('segment_id')['segment_id'].transform('size')
    healthy_frame=healthy_frame.loc[counts.ge(minimum_records)].reset_index(drop=True)
    if healthy_frame.empty:
        raise ValueError('No alarm-free segment has at least the required records; raw export is preserved.')
    cut=int(len(healthy_frame)*fraction)
    if cut<=0 or cut>=len(healthy_frame): raise ValueError('Empty chronological split')
    return healthy_frame.iloc[:cut].copy().reset_index(drop=True),healthy_frame.iloc[cut:].copy().reset_index(drop=True)
