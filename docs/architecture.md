# System Architecture

See root README.md for the high-level diagram.

## Components

### ML Engine (`ml/`)
- RNFB anatomical graph (Garway-Heath map, 54 nodes)
- Graph Attention Network (GAT) anatomical prior
- GRU-based spatiotemporal progression model
- Progression detection: PLR baseline + our spatiotemporal method
- Alert engine with confidence-weighted thresholding

### Backend (`backend/`)
- FastAPI + PostgreSQL + Redis + Celery
- JWT auth with RBAC (admin / doctor / technician / patient)
- Hospital-scoped data isolation
- Async ML inference via Celery tasks

### Mobile (`mobile/`)
- Flutter (Android API 26+, iOS 14+)
- FDT-adapted visual field test (17 locations, 3 min)
- Offline-first with background sync

### Dashboard (`dashboard/`)
- React 18 + TypeScript + TailwindCSS
- VF heatmap, progression charts, RNFB sector chart
- Real-time alerts via WebSocket
