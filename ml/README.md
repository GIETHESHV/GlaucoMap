# ML — GlaucoMap Model Engine

## Model Architecture

```
[VF Test Session (54 sensitivity values)]
        ↓
[GNN Anatomical Prior Layer]
  - 3-layer Graph Attention Network (GAT)
  - Nodes: 54 Humphrey 24-2 VF locations
  - Edges: RNFB bundle co-vulnerability (Garway-Heath map)
  - Output: 54-dim anatomically-smoothed latent embedding
        ↓
[Spatiotemporal Model]
  - GRU over T session embeddings (irregular time intervals)
  - Sinusoidal time-delta encoding
  - Output: per-location progression rate (dB/year) + uncertainty
        ↓
[Progression Detector + Alert Engine]
  - PLR baseline comparison
  - Bundle-coherent alert thresholding
  - Confidence-weighted suppression
```

## Setup

```bash
cd ml
pip install -r requirements.txt
```

## Key Files

| File | Description |
|------|-------------|
| `data/anatomical_graph.py` | RNFB graph construction (Garway-Heath) |
| `data/dataset_loader.py` | Multi-format dataset ingestion |
| `data/patient_dataset.py` | PyTorch Dataset wrapper |
| `models/anatomical_gnn.py` | GAT anatomical prior |
| `models/spatiotemporal_model.py` | GRU progression model |
| `models/full_pipeline.py` | End-to-end model |
| `detection/progression_detector.py` | PLR + our method |
| `detection/alert_engine.py` | Alert thresholding |
| `serving/predictor.py` | Clean inference interface |
| `training/train.py` | Training loop with W&B |

## Training

```bash
python training/train.py --config config/base_config.yaml
```

## Inference

```python
from ml.serving.predictor import GlaucomaPredictor
predictor = GlaucomaPredictor("checkpoints/best_model.pt")
result = predictor.predict(sessions)
```

## Owner: Person A
