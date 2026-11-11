# Implementation scope and provenance

This repository contains selected VC-LAD implementation modules and a ten-vehicle real-data subset.

The following source modules were extracted from the research workspace (paths relative to that workspace):

- `experiment/实验2/standalone_reconad.py` -> `vclad/detector.py`; the model definition is consolidated in `vclad/model.py`, and AP uses distinct score thresholds.
- `experiment/实验1/model/ReconAD/ReconAD.py` -> `vclad/model.py`, with the canonical model name `VCLAD` and compatibility aliases.
- `core/quality_metrics.py` -> `vclad/quality_metrics.py`.
- `_smooth_fault_envelope` from `experiment/Data/create_data/build_semi_synthetic_dataset_softened_v4.py` -> `vclad/envelope.py`.
- Last-step contribution formula from `CloudAnomalyLocalizer._normalized_variable_scores` in `experiment/实验4/exp4_localization_quality_rules_only.py` -> portable `vclad/ranking.py`.

The examples, preprocessing wrapper, data projection tool, tests and documentation are release-specific additions. The detector/scoring modules use the aligned 0.8/0.2 default; variable ranking normalizes last-step P/S contributions separately; the envelope uses 0.30/0.20; timestamp validity includes legal timestamps and [0,15]-second gaps.

Executable edge threshold calibration/orchestration and the full semi-synthetic fault generator are omitted. Their high-level algorithms are documented as pseudocode in `algorithms.md`.

The published data are not accompanied by independent fault-event labels. Do not report the example's scores as measured detection performance. The local metric functions are made available for inspection and externally supplied correctly aligned labels, not automatically applied to alarm levels.

A reuse license has not yet been specified.

A-PR follows the standard average-precision definition stated in the paper: recall increments are weighted by precision at distinct score thresholds. This corrects order dependence for tied scores in the extracted implementation and also affects local V-PR calculations. Scores and model weights are unchanged by this correction; metrics previously calculated with the older function must be recomputed before comparison. Paper tables have not been recalculated using this public subset.
