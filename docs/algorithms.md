# Core algorithms

## Executable detector and variable ranking

For a length-30 window of the nine ordered model features, reconstruct with the 48-dimensional one-layer GRU. Let P be the mean squared reconstruction error and S the mean squared temporal-difference reconstruction error. Calibrate P and S separately with healthy-training median and scale `max(Q75-Q25, Q90-median, 1e-8)`, clipped to [0,20]. Use last-step score `0.8*P + 0.2*S`.

At the peak-scoring window overlapping an event, take the last-step per-variable P/S errors, max-normalize each vector separately, combine with weights 0.8/0.2, and sort descending. Zero contribution vectors remain zero. Executable versions are in `detector.py`, `model.py` and `ranking.py`.

## Vehicle-side screening — pseudocode, not a released executable pipeline

```text
for each vehicle and condition:
    derive four window quality indicators from healthy calibration records
    use the 14th percentile of each indicator as its lower threshold
    for insufficient condition support, fall back to vehicle calibration
    for unseen vehicles, use the mean of available vehicle thresholds
for each incoming window:
    compute accuracy, completeness, consistency and timeliness
    flag if any indicator is below its applicable threshold
    also flag any hard validity, consistency or timeliness violation
    upload flagged windows for cloud confirmation
```

Quality-indicator functions are executable in `quality_metrics.py`. Condition assignment, support-based fallback, streaming orchestration and cloud confirmation are not implemented by this pseudocode.

## Semi-synthetic fault construction — pseudocode except for envelope

```text
select eligible alarm-free test backgrounds; do not inject into training
sample the event location and duration under the experiment constraints
Type 1: perturb minimum cell voltage and pack voltage; adjust voltage spread
Type 2: perturb pack current and insulation resistance
Type 3: perturb maximum temperature and pack voltage
apply cosine envelope (rise 0.30, fall 0.20) and the experiment noise/amplitudes
record event intervals, type and affected-variable annotations separately
```

Only the exact envelope function is released here. This pseudocode omits amplitude/noise sampling and selection details and cannot regenerate the paper's benchmark. Exported real CSV files contain no injected events.
