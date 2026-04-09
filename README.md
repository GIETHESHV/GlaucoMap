# GlaucoMap

A mobile-first, AI-powered glaucoma progression detection system combining self-administered visual field tests with neuroanatomical GNN modeling — deployable in low-resource clinical settings across India and beyond.

## Architecture

```
[Flutter Mobile App]
    ↓  (FDT-adapted visual field test, 3 min)
[FastAPI Backend]
    ↓  (REST / WebSocket)
[Spatiotemporal Model Engine]
    ├── Bayesian Sensitivity Map (per eye, per session)
    ├── GNN Anatomical Prior (retinal nerve fiber bundle graph)
    ├── Progression Detection (scikit-survival / PyTorch)
    └── Alert Engine (imminent vision loss threshold)
    ↓
[Dashboard — Clinician Web UI]
    └── Patient timeline, heatmaps, progression reports
```

## Repository Structure

```
glaucoma-tracker/
├── backend/        # FastAPI server, DB, auth, clinical data pipeline
├── mobile/         # Flutter app + FDT visual field test engine
├── ml/             # GNN model, spatiotemporal model, progression detection
├── dashboard/      # React clinician web dashboard
├── data/           # Datasets, templates, fixtures
└── docs/           # Architecture docs, clinical validation, guides
```

## Quickstart

See each subdirectory's `README.md` for setup instructions:
- [ML](ml/README.md)
- [Backend](backend/README.md)
- [Mobile](mobile/README.md)
- [Dashboard](dashboard/README.md)

## Team

| Person | Role |
|--------|------|
| A | ML Core & GNN Anatomical Model |
| B | Backend API & Data Infrastructure |
| C | Mobile App & Visual Field Test Engine |
| D | Clinician Dashboard, DevOps & Integration |

## Milestones

| Milestone | Owner | Target Week |
|-----------|-------|-------------|
| M1: RNFB graph + dataset loader | A | Week 2 |
| M2: API auth + session endpoints live | B | Week 2 |
| M3: Flutter test engine renders stimulus | C | Week 3 |
| M4: Dashboard scaffold with VF heatmap | D | Week 3 |
| M5: GNN model training on Rotterdam data | A | Week 5 |
| M6: Mobile → API → ML full pipeline | B+C | Week 6 |
| M7: Clinician dashboard connected to live API | D | Week 7 |
| M8: Clinical validation benchmarks done | A+D | Week 9 |
| M9: v1.0.0 tagged, demo package ready | All | Week 10 |
