# Public real-data schema

Each file represents one vehicle under a neutral filename (`vehicle_001.csv` through `vehicle_010.csv`). CSV encoding is UTF-8 with BOM for Excel compatibility. Only the source vehicle-data sheet is exported; motor-detail and cell/probe-detail sheets are excluded.

| Field | Purpose |
| --- | --- |
| 数据时间 | Timestamp, ordering, continuity and timeliness checks |
| 最高报警等级 | Select no-reported-alarm backgrounds; not fault ground truth |
| 车速 | Model input 1; consistency checks |
| 总电压 | Model input 2 |
| 总电流 | Model input 3 |
| SOC | Model input 4 |
| 绝缘电阻 | Model input 5 |
| 电池单体电压最高值 | Model input 6; voltage consistency |
| 电池单体电压最低值 | Model input 7; voltage consistency |
| 最高温度值 | Model input 8; temperature consistency |
| 最低温度值 | Model input 9; temperature consistency |
| 车辆状态 | Vehicle-side consistency checks; not a model input |
| 充电状态 | Vehicle-side consistency checks; not a model input |
| 挡位 | Vehicle-side consistency checks; not a model input |

No acceleration column exists in these source files. The speed/acceleration consistency rule can only be applied when an acceleration signal is supplied; this release does not fabricate one or silently derive one from the signal being checked.

Export performs column projection only. Source rows, order, timestamps, numerical values, missing entries and invalid-value markers are retained. There is no interpolation, normalization, deduplication, healthy filtering or injected fault in `data/real`. Excel datetime objects are serialized as ISO timestamps. `manifest.json` records omitted fields and missing counts. Original workbooks are not distributed.

Coordinates, cumulative mileage, subsystem/cell/probe identifiers, detailed fault lists, engine/hydrogen-system fields and other unused columns are removed. Absolute timestamps remain because they are used by time-based checks; they have not been shifted. Neutral filenames and field removal do not establish formal anonymity of time-series data.

Training preparation is separate from export. `run.py` excludes invalid timestamps, stably sorts records, selects alarm-level-zero runs separated by gaps greater than 15 seconds or alarmed records, retains runs of at least 500 records, and splits chronologically 70/30. It fits scaling/calibration on training data only. The extracted detector interpolates invalid/missing model values within each segment and uses its original zero fallback for entirely absent values; raw exports remain unchanged. An alarm level of zero means no alarm was reported; it does not independently establish that a record is fault-free.
