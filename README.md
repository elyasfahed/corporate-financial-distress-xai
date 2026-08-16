# Explainable AI for Corporate Financial Distress Prediction

Code and selected research outputs for an M.Sc. Finance thesis on
explainable machine-learning models for one-year-ahead corporate financial
distress prediction using CRSP/Compustat data.

## Repository contents

- `src/`: data, feature, modelling, evaluation, robustness and explainability code
- `scripts/`: reproducibility, validation and output-production entry points
- `tests/`: unit and regression tests
- `outputs/figures/`: selected final figures
- `outputs/tables/`: selected final result tables
- `outputs/models/configs/`: lightweight final model configurations
- `outputs/verification/`: lightweight verification records

## Study design

References to the pre-specified study design identify choices fixed before the
final models were estimated: sample construction, predictors, model selection,
evaluation, robustness checks and explainability analysis. Numbered references
link implementation details to the relevant part of that design.

## Held-out results

The final test sample contains 25,512 firm-year observations from fiscal years
2015–2023, including 404 distress events (1.58%). Performance on that sample is:

| Model | PR-AUC | ROC-AUC |
|---|---:|---:|
| Logistic regression (ridge) | 0.1751 | 0.9203 |
| XGBoost | 0.1729 | 0.9303 |
| Random forest | 0.1608 | 0.9272 |
| Neural network | 0.1338 | 0.9140 |

Logistic regression has the highest point estimate on the primary metric,
PR-AUC, but its differences from the two tree models are not statistically
significant after Holm correction. The results therefore do not support a
general machine-learning advantage over the regularised linear benchmark.
XGBoost has the highest ROC-AUC and is used for the main SHAP and LIME analysis.
The complete metrics are available in the
[four-model results table](outputs/tables/model_results/final_primary/model_performance_test_4models.csv).

## Data availability

The raw and processed CRSP, Compustat and WRDS datasets are not included because
they are licensed data. Row-level data, trained model binaries, optimisation
databases and logs are also excluded.

Users with appropriate data access can configure the licensed data location
from `.env.example`. Never commit credentials or downloaded data.

## Environment

The project targets Python 3.11. Create an isolated environment and install the
locked dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
```

The complete research environment requires licensed inputs and additional large
artifacts. This repository contains the publishable code and lightweight output
tables and figures.

## Repository provenance

This is a publication snapshot with a clean history. The author remains
responsible for the methodology, implementation and reported results.

The included verification record was produced in the complete research
environment. This reduced publication snapshot is not byte-identical to that
environment and therefore is not expected to reproduce its code-fingerprint
check.

## Research scope

The model portfolio comprises regularised logistic regression, random forest,
XGBoost and a neural network. The explanation layer uses SHAP and LIME. The
repository separates prediction, model explanation and economic interpretation;
model attributions are not causal effects.
