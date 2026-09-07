# 👗 StyleFit AI

**Personalized Clothing Fit & Size Intelligence**

StyleFit AI is a full-stack machine learning product that predicts how a selected clothing size is likely to fit:

- Small
- Fit
- Large

The project combines data analysis, machine learning, explainability, customer insights and a modern full-stack product experience.

## Project status

🚧 Model comparison in progress

Completed foundations: dataset audit, EDA, deterministic cleaning, leakage-safe
preprocessing, user-aware splitting, baseline evaluation, and automated tests.

The current phase compares class-balanced linear and tree models using user-disjoint
cross-validation. A separate user-disjoint final holdout is sealed before comparison
and is not evaluated or used for candidate selection.

## Reproducible model comparison

Install the focused ML environment:

```bash
python3 -m pip install -r requirements-ml.txt
```

Run the full comparison from the project root:

```bash
python3 -m src.model_comparison
```

The command compares the existing dummy and unweighted logistic references with
balanced Logistic Regression, balanced Random Forest, and XGBoost using balanced
sample weights. The primary selection metric is mean Macro F1 across three
`StratifiedGroupKFold` folds. Results are written to `reports/model_comparison/`
and figures to `reports/figures/model_comparison/`.

Threshold tuning and final-holdout evaluation are deliberately deferred to later
phases.

## Planned ML pipeline

Dataset audit → preprocessing → baselines → model comparison → threshold tuning → explainability → customer insights → FastAPI → React → deployment

## Dataset

The project uses the public Clothing Fit datasets published by UCSD:

- Rent the Runway
- ModCloth

Raw and processed datasets are excluded from this repository.

The active pipeline uses Rent the Runway only. ModCloth is reserved for a later,
explicit integration phase.
