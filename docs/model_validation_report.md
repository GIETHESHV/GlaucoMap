# Model Validation Report

> Generated after training on Rotterdam Optic Nerve Study dataset.
> To be populated by Person A after running `ml/evaluation/metrics.py`.

## Detection Performance

| Method | AUC | Sens @ 80% Spec | Sens @ 85% Spec | Sens @ 90% Spec |
|--------|-----|-----------------|-----------------|-----------------|
| PLR (baseline) | TBD | TBD | TBD | TBD |
| Our model (full) | TBD | TBD | TBD | TBD |
| Ablation: no GNN | TBD | TBD | TBD | TBD |
| Ablation: no time enc | TBD | TBD | TBD | TBD |

## Progression Rate Prediction

| Method | MAE (dB/year) |
|--------|---------------|
| PLR | TBD |
| Our model | TBD |

## Time-to-Detection

| Method | Mean sessions | Median sessions |
|--------|---------------|-----------------|
| PLR | TBD | TBD |
| Our model | TBD | TBD |

## Brier Score (Event Prediction)

| Method | Brier Score |
|--------|-------------|
| PLR | TBD |
| Our model | TBD |

## Figures

- `docs/figures/rnfb_graph_visualization.png` — RNFB anatomical graph
- `docs/figures/roc_curves.png` — ROC curves for all methods
- `docs/figures/progression_rate_scatter.png` — Predicted vs true rates
- `docs/figures/ablation_bar_chart.png` — Ablation study comparison
