# VC-LAD

Core implementation of VC-LAD and a ten-vehicle real-data subset.

## Quick start

Python 3.9 or later. CPU execution is sufficient for the example.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python demo.py
python run.py --csv data/real/vehicle_007.csv --output outputs/vehicle_007 --epochs 30
```

`demo.py` uses generated synthetic background data for a one-epoch smoke test. It does not demonstrate fault-detection accuracy. `run.py` selects alarm-free contiguous segments of at least 500 records, performs a chronological 70/30 split, trains one vehicle model, and writes last-step anomaly scores and the top-ranked variable for each test window. A vehicle with insufficient eligible data raises an explicit error. Outputs are local and ignored by Git.

## Contents

- `vclad/detector.py`: scaling, training, robust calibration, scoring and experimental metric implementations.
- `vclad/model.py`: the shared `VCLAD` model, reconstruction loss and P/S scoring API; training and inference use this model through `detector.py`.
- `vclad/ranking.py`: last-step variable contributions and event-overlap peak-window selection.
- `vclad/quality_metrics.py`: field validity, completeness, consistency and timestamp checks. Checks apply to fields present; missing schema should not be interpreted as a complete quality assessment.
- `vclad/envelope.py`: the paper's cosine envelope only; not the complete injection generator.
- `vclad/preprocessing.py`, `run.py`, `demo.py`: portable public-subset preparation and runnable examples.
- `data/real/`: ten CSV files, 14 retained columns each.
- `data/manifest.json`: per-file record counts, missing-value counts, removed fields and checksums.
- `docs/`: data schema, scope, algorithm pseudocode and provenance.
- `configs/paper.json`: reference paper settings. The runnable example uses the corresponding defaults directly; it does not read this JSON automatically.

The ten vehicles are a public subset, not the complete paper benchmark. This package does not contain the original full 100-vehicle/531-event evaluation data, trained weights, full fault-injection orchestration, baseline implementations, figures, production dashboard, or private deployment files. It cannot by itself reproduce all paper tables. No fault ground-truth annotations are supplied, and alarm status must not be treated as such annotations.

The local V-ROC/V-PR implementations average results over dilated label ranges. They are not the official VUS implementation. R-F1 and Aff-F1 are also the project's local definitions; see their code before comparing with other implementations. The reference evaluator retains its experimental smoothing option; the public `run.py` example explicitly applies no score smoothing.

See [data schema](docs/data_schema.md) and [release scope](docs/publication_scope.md).

A-PR uses non-interpolated average precision at distinct score thresholds, including tied-score groups. Earlier results computed with the extracted row-wise AP implementation require recalculation when ties occur.
