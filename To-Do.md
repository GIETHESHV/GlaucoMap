# 👁️ Glaucoma Progression Tracker — Full Project Build Plan

> **Mission:** Build a mobile-first, AI-powered glaucoma progression detection system combining self-administered visual field tests with neuroanatomical GNN modeling — deployable in low-resource clinical settings across India and beyond.
>
> **Target:** Production-grade, publishable, and sellable at **$10,000+**
>
> **Team Size:** 4 Engineers | **Stack:** Python · PyTorch · PyTorch Geometric · Flutter · FastAPI · scikit-survival

---

## 📐 Overall System Architecture

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

---

## 🗂️ Work Process Overview

| Phase | Duration | Focus |
|-------|----------|-------|
| **Phase 1 — Foundations** | Weeks 1–2 | Repo setup, data pipelines, mobile scaffold, API skeleton |
| **Phase 2 — Core Models** | Weeks 3–5 | GNN prior, spatiotemporal model, progression engine |
| **Phase 3 — Integration** | Weeks 6–7 | Mobile ↔ API ↔ Model full pipeline, clinician dashboard |
| **Phase 4 — Validation** | Weeks 8–9 | Clinical dataset testing, model benchmarking, accuracy reports |
| **Phase 5 — Polish & Ship** | Week 10 | CI/CD, docs, demo, packaging for sale/deployment |

### GitHub Workflow
- `main` — protected, release-only
- `dev` — integration branch
- `feature/A-*`, `feature/B-*`, `feature/C-*`, `feature/D-*` — personal branches
- All PRs go into `dev` with at least one reviewer
- Weekly sync: merge `dev` → `main` after integration tests pass

---

## 👤 Person A — ML Core & GNN Anatomical Model

**Role:** Lead ML Engineer — Spatiotemporal model, GNN anatomical prior, progression detection engine

---

### A1. Repository & Environment Bootstrap
- Initialize monorepo: `glaucoma-tracker/` with subdirs `backend/`, `mobile/`, `ml/`, `dashboard/`, `data/`, `docs/`
- Create `ml/requirements.txt`: PyTorch, PyTorch Geometric, scikit-survival, numpy, pandas, scipy, matplotlib, seaborn, wandb
- Set up `ml/README.md` with model architecture overview
- Configure `ml/config/base_config.yaml` with all hyperparameters externalized
- Push initial structure to `feature/A-setup` → PR into `dev`

---

### A2. Retinal Nerve Fiber Bundle Graph Construction
- Research and encode the Garway-Heath map: 54 visual field test point locations → retinal nerve fiber bundle sectors (6 sectors: superotemporal, superonasal, nasal, inferonasal, inferotemporal, temporal)
- Build `ml/data/anatomical_graph.py`:
  - Nodes = 54 standard Humphrey VF test locations (24-2 grid)
  - Edges = anatomical connectivity (same RNFB sector = strong edge; adjacent sector = weak edge)
  - Node features: eccentricity (degrees), angular position, expected sensitivity (age-normalized dB)
  - Edge weights: anatomical co-vulnerability score
- Serialize graph as `ml/data/rnfb_graph.pt`
- Write unit tests: `ml/tests/test_anatomical_graph.py` (verify node count, edge symmetry, sector assignment)
- Visualize graph with matplotlib — save as `docs/figures/rnfb_graph_visualization.png`

---

### A3. Dataset Ingestion & Preprocessing Pipeline
- Write `ml/data/dataset_loader.py` supporting:
  - Rotterdam Optic Nerve Study format
  - UKBB visual field CSV format
  - Moorfields format (DICOM metadata + perimetry export)
  - Custom hospital CSV format (template in `data/templates/hospital_export_template.csv`)
- Implement preprocessing:
  - Age-correction of sensitivity values (dB normalization by age)
  - Reliability filtering: false positive rate < 15%, false negative rate < 20%, fixation loss < 20%
  - Missing value imputation using spatial interpolation (kriging over VF grid)
- Build `ml/data/patient_dataset.py` as a `torch.utils.data.Dataset` subclass
  - Each item: `{patient_id, eye, session_index, sensitivity_map [54], age, md_score, psd_score, test_duration, reliability_flags}`
- Write data split utility: `ml/data/splits.py` (stratified by progression status, age, baseline severity)
- Add data validation script: `ml/data/validate_dataset.py`

---

### A4. Graph Neural Network — Anatomical Prior Layer
- Implement `ml/models/anatomical_gnn.py` using PyTorch Geometric:
  - Architecture: 3-layer Graph Attention Network (GAT) with anatomical edge weights
  - Input: 54-node feature vectors (sensitivity + spatial encoding)
  - Output: 54-dimensional anatomically-smoothed latent embedding per test session
  - Skip connections to preserve raw sensitivity signal
  - Learnable anatomical soft-constraint: penalize predictions that violate RNFB bundle co-degradation
- Implement `ml/models/anatomical_loss.py`:
  - Bundle coherence loss: penalize if two nodes in the same bundle deviate by > 3dB in predicted progression rate
  - Asymmetry prior: nasal step patterns should have sharp gradients
- Write ablation config to allow disabling the anatomical prior (for benchmarking purposes)
- Unit tests: `ml/tests/test_gnn.py` — verify forward pass shapes, gradient flow, edge weight influence

---

### A5. Spatiotemporal Progression Model
- Implement `ml/models/spatiotemporal_model.py`:
  - Input: sequence of GNN embeddings across T test sessions per patient
  - Architecture: Temporal component using a GRU over session embeddings (handles irregular time intervals)
  - Time encoding: sinusoidal time-delta encoding (weeks between tests as input)
  - Output: per-location progression rate (dB/year) + uncertainty estimate (heteroscedastic output)
  - Uncertainty: predict both mean and log-variance at each VF location
- Implement `ml/models/measurement_noise_model.py`:
  - Model test-retest variability as a function of sensitivity (empirical Henson curve)
  - Used to distinguish real progression from measurement noise
  - Parameters fit from normative dataset
- Write `ml/models/full_pipeline.py` combining GNN + spatiotemporal model end-to-end
- Training loop: `ml/training/train.py` with W&B logging, checkpoint saving, early stopping

---

### A6. Progression Detection & Alert Engine
- Implement `ml/detection/progression_detector.py`:
  - Method 1: Pointwise Linear Regression on raw sensitivity (PLR baseline — industry standard)
  - Method 2: Spatiotemporal model with anatomical prior (our method)
  - Comparison report generated automatically
- Implement `ml/detection/alert_engine.py`:
  - Progressive threshold: if ≥ 3 VF locations in the same RNFB bundle show ≥ 1 dB/year decline over 3+ sessions → ALERT
  - Rate-based projection: project to 10% sensitivity loss threshold and estimate months remaining
  - Confidence-weighted alerting: suppress alerts when uncertainty > 2× measurement noise
- Implement `ml/detection/event_analysis.py` (pointwise event detection — VFI < 85% threshold)
- Output format: JSON progression report per patient (fed to API and clinician dashboard)

---

### A7. Model Validation & Benchmarking
- Implement `ml/evaluation/metrics.py`:
  - Sensitivity and specificity of progression detection (at 80%, 85%, 90% specificity thresholds)
  - Time-to-detection (how many sessions needed vs PLR baseline)
  - Mean Absolute Error on progression rate prediction
  - Brier score for event prediction
- Run experiments on Rotterdam dataset:
  - Ablation: GNN prior ON vs OFF
  - Ablation: Time encoding ON vs OFF
  - Comparison: our model vs PLR vs AGIS vs CIGTS scoring
- Generate `docs/model_validation_report.md` with tables and figures
- Save all trained model weights to `ml/checkpoints/`

---

### A8. Model Export & Serving Interface
- Export trained model to TorchScript: `ml/export/export_model.py`
- Write `ml/serving/predictor.py`: clean inference interface
  - Input: list of VF test sessions (JSON)
  - Output: progression rates, uncertainty, alert status, projected timeline
- Benchmark inference time (target < 200ms per patient on CPU)
- Write `ml/serving/schema.py` with Pydantic models for all inputs/outputs (shared with Person B)

---

## 👤 Person B — Backend API & Data Infrastructure

**Role:** Backend Engineer — FastAPI server, database, auth, clinical data pipeline, deployment

---

### B1. Backend Project Setup
- Scaffold `backend/` with FastAPI + Uvicorn
- Dependencies: FastAPI, SQLAlchemy, Alembic, Pydantic v2, python-jose (JWT), bcrypt, celery, redis, boto3, psycopg2, httpx
- Set up `docker-compose.yml` with services: `api`, `postgres`, `redis`, `celery_worker`
- Set up `backend/.env.example` with all required environment variables
- Initialize Alembic for migrations: `backend/alembic/`
- Push to `feature/B-setup` → PR into `dev`

---

### B2. Database Schema Design & Migrations
- Design and implement SQLAlchemy models in `backend/models/`:
  - `Patient`: id, mrn, name, dob, eye_affected, diagnosis_date, hospital_id, created_at
  - `Hospital`: id, name, city, state, license_key
  - `Clinician`: id, hospital_id, name, email, role (admin/doctor/technician), hashed_password
  - `TestSession`: id, patient_id, eye (L/R), test_date, duration_seconds, reliability_flags (JSON), raw_sensitivity_values (JSON array[54]), md_score, psd_score, vfi, device_type, app_version
  - `ProgressionReport`: id, patient_id, generated_at, model_version, progression_rates (JSON), uncertainty_map (JSON), alert_level (none/watch/alert/urgent), projected_loss_months, report_pdf_url
  - `Alert`: id, patient_id, clinician_id, created_at, alert_level, acknowledged_at, notes
- Write Alembic migrations for all tables
- Seed script: `backend/scripts/seed_demo_data.py` with synthetic patient data

---

### B3. Authentication & Authorization System
- Implement `backend/auth/`:
  - JWT access tokens (15 min expiry) + refresh tokens (7 days)
  - Role-based access control: `admin`, `doctor`, `technician`, `patient`
  - Hospital-scoped data isolation (clinicians can only see their hospital's patients)
  - API key auth for mobile app (per-device key issued at registration)
- Endpoints:
  - `POST /auth/login` — email + password → tokens
  - `POST /auth/refresh` — refresh token → new access token
  - `POST /auth/register-device` — mobile device registration
  - `GET /auth/me` — current user info
- Write `backend/tests/test_auth.py`

---

### B4. Core Patient & Session API Endpoints
- Implement `backend/routers/patients.py`:
  - `POST /patients/` — register new patient
  - `GET /patients/{id}` — patient profile
  - `GET /patients/` — paginated list with filters (hospital, date range, alert status)
  - `PUT /patients/{id}` — update demographics
- Implement `backend/routers/sessions.py`:
  - `POST /sessions/` — submit new test session (from mobile)
  - `GET /sessions/{id}` — single session detail
  - `GET /patients/{id}/sessions` — all sessions for a patient (timeline)
  - `DELETE /sessions/{id}` — admin only, with audit log
- Input validation: sensitivity array must have exactly 54 values in valid dB range (0–40)
- Write `backend/tests/test_sessions.py`

---

### B5. ML Model Integration & Async Processing
- Implement `backend/services/ml_service.py`:
  - On new session submission, enqueue Celery task: `tasks.run_progression_analysis`
  - Task: load patient's full session history → call `ml/serving/predictor.py` → store ProgressionReport
  - If alert_level ≥ `alert` → trigger notification pipeline
- Implement `backend/tasks/analysis_tasks.py` (Celery):
  - `run_progression_analysis(patient_id)`
  - `generate_pdf_report(report_id)`
  - `send_clinician_alert(alert_id)`
- Implement `backend/routers/reports.py`:
  - `GET /reports/{patient_id}/latest` — most recent progression report
  - `GET /reports/{patient_id}/history` — all reports
  - `POST /reports/{patient_id}/generate` — force regeneration
- Add WebSocket endpoint `ws/patients/{id}/live` — stream real-time test feedback during mobile test

---

### B6. Clinical Data Import Pipeline
- Implement `backend/ingestion/`:
  - `import_csv.py` — bulk import from hospital CSV exports (using A3 format templates)
  - `import_dicom.py` — parse perimetry-relevant DICOM metadata
  - `validate_import.py` — pre-import validation with error report
- Create `POST /admin/import` endpoint (admin-only, multipart file upload)
- Background task for large imports with progress tracking
- `GET /admin/import/{job_id}/status` — import progress endpoint
- Write audit log for all imports: who, when, how many records

---

### B7. Reporting & Export API
- Implement `backend/services/pdf_generator.py` using ReportLab:
  - Clinical PDF report: patient demographics, session timeline, VF sensitivity heatmaps, progression rate chart, alert summary, clinician notes field
  - Clinic aggregate report: active patients, alert counts, progression statistics
- Implement `backend/routers/export.py`:
  - `GET /export/patient/{id}/pdf` — patient PDF report
  - `GET /export/clinic/summary` — clinic-wide CSV export
  - `GET /export/patient/{id}/raw-data` — JSON export for research
- Integrate with S3 or local file storage for PDF persistence

---

### B8. Deployment, CI/CD & Monitoring
- Write `backend/Dockerfile` and `ml/Dockerfile`
- Configure GitHub Actions in `.github/workflows/`:
  - `ci.yml` — lint (ruff), type-check (mypy), pytest on every PR to `dev`
  - `deploy.yml` — build Docker images, push to registry, deploy on merge to `main`
- Set up health check endpoint: `GET /health` and `GET /health/ml`
- Implement structured logging (structlog) with request ID tracing
- Set up Prometheus metrics endpoint: `/metrics` (request count, latency, ML inference time, alert rate)
- Write `docs/deployment_guide.md` and `docs/api_reference.md` (auto-generated from OpenAPI)

---

## 👤 Person C — Mobile App (Flutter) & Visual Field Test Engine

**Role:** Mobile Engineer — Flutter app, FDT-adapted VF test stimulus system, test UX, patient-facing features

---

### C1. Flutter Project Setup
- Scaffold `mobile/` Flutter project (target: Android API 26+, iOS 14+)
- Dependencies: `dio`, `flutter_secure_storage`, `provider`/`riverpod`, `fl_chart`, `camera`, `sensors_plus`, `wakelock_plus`, `sqflite`, `path_provider`, `flutter_local_notifications`
- Set up flavors: `development`, `staging`, `production`
- Set up `mobile/lib/` folder structure: `screens/`, `widgets/`, `services/`, `models/`, `providers/`, `test_engine/`
- Configure CI: Flutter analyze + widget tests on PR
- Push to `feature/C-setup` → PR into `dev`

---

### C2. FDT-Adapted Visual Field Test Engine
- Implement `mobile/lib/test_engine/stimulus_controller.dart`:
  - Frequency-doubling stimulus: alternating sinusoidal grating that appears at double the actual spatial frequency (psychophysical illusion targeting M-cell pathway, same pathway damaged early in glaucoma)
  - Stimulus parameters: 0.5 cpd spatial frequency, 25 Hz temporal frequency, circular aperture 5° diameter
  - Stimulus positions: subset of 24-2 grid (17 locations per eye for 3-min test — selected by information-maximizing algorithm)
  - Adaptive staircase: 4-2 dB staircase procedure starting at age-expected threshold
  - Max test duration: 3 minutes enforced via countdown timer
- Implement `mobile/lib/test_engine/fixation_monitor.dart`:
  - Front camera gaze estimation (MediaPipe face mesh or OpenCV) to detect if patient is fixating centrally
  - Fixation loss: if gaze deviation > 5° for > 500ms during stimulus → mark trial as unreliable
  - Pause test and warn patient if fixation quality drops below threshold
- Implement `mobile/lib/test_engine/response_recorder.dart`:
  - Touch tap = "I saw it" response (large tap target, any screen area)
  - Response time recording (< 700ms expected)
  - False positive catch trials (no stimulus presented, response = false positive)
  - False negative catch trials (suprathreshold stimulus, no response = false negative)

---

### C3. Screen Calibration & Luminance Control
- Implement `mobile/lib/test_engine/screen_calibrator.dart`:
  - Set screen brightness to maximum and lock during test (using `wakelock_plus`)
  - Request user to disable adaptive brightness before test
  - Measure ambient light via light sensor and warn if > 10 lux (room should be dim)
  - Display test distance guidance (33 cm = standard perimetry distance)
  - One-time per-device calibration: display known gray patches, ask user to confirm equal brightness → estimate screen gamma
- Calibration wizard screen with step-by-step visual instructions
- Store calibration parameters in `sqflite` local DB per device
- Show calibration status badge in test screen header

---

### C4. Test Flow UI & Patient Experience
- Implement screens in `mobile/lib/screens/test/`:
  - `PreTestChecklistScreen`: checklist — glasses on, room dark, phone at arm's length, one eye covered
  - `FixationTargetScreen`: practice fixation with animated bull's-eye before real test
  - `TestRunningScreen`: full-screen dark background with fixation cross, countdown timer, progress bar, stimulus rendering using `CustomPainter`
  - `TestPausedScreen`: pause/resume with fixation quality warning
  - `TestCompleteScreen`: animated "Test complete!" with reliability summary and submit button
- Implement `mobile/lib/widgets/test_engine/stimulus_painter.dart`: renders FDT grating using `CustomPainter` with precise timing using `Ticker`
- All test screens must prevent back navigation (accidental interruption prevention)
- Haptic feedback on tap response (light impact)

---

### C5. Patient App — Core Features
- Implement screens in `mobile/lib/screens/`:
  - `OnboardingScreen`: 3-slide explainer (what is glaucoma, why test at home, how this app helps)
  - `LoginScreen` / `RegisterScreen`: email/password + hospital code entry
  - `HomeScreen`: next test due date, streak counter, last MD score, alert banner if flagged
  - `TestHistoryScreen`: timeline list of past tests with date, reliability, MD score
  - `VFHeatmapScreen`: visual field sensitivity heatmap using `fl_chart` or custom `CustomPainter` — color-coded dB values on anatomical VF grid
  - `ProgressionChartScreen`: line chart of MD score over time with trend line
  - `NotificationsScreen`: test reminders, clinician alerts, app updates
- Implement `mobile/lib/services/reminder_service.dart`:
  - Local push notifications for weekly/biweekly test reminders
  - Smart scheduling: don't remind within 2 days of last test
  - Missed test escalation: 3 missed reminders → notify linked clinician

---

### C6. Offline Support & Data Sync
- Implement `mobile/lib/services/local_db_service.dart` (sqflite):
  - Store completed test sessions locally until sync succeeds
  - Store last-known progression report for offline viewing
  - Queue failed API calls for retry
- Implement `mobile/lib/services/sync_service.dart`:
  - Background sync on app resume and network reconnect
  - Conflict resolution: server wins for reports, client wins for test sessions (append-only)
  - Sync status indicator in HomeScreen
- Handle no-internet gracefully: complete test offline, queue for upload, show "Pending sync" badge

---

### C7. Clinician Companion Screen (Lite)
- Implement `mobile/lib/screens/clinician/`:
  - `ClinicianPatientListScreen`: list of assigned patients, filter by alert level
  - `PatientDetailScreen` (mobile): mini summary — last test date, MD trend sparkline, alert status
  - `AlertInboxScreen`: pending alerts with acknowledge button
- Note: Full clinician dashboard is web-based (Person D) — mobile is a companion view only
- Role detection on login: patient vs clinician shows different home screens

---

### C8. App Testing, Accessibility & Release Build
- Write widget tests: `mobile/test/` for all core screens
- Write integration test: full test-taking flow (mock stimulus, real response recording)
- Accessibility: ensure minimum tap target size 48×48dp, high-contrast fixation cross option, font size respects system settings
- Performance: test screen must render at 60 fps — profile and optimize `CustomPainter`
- Build release APK and AAB: `flutter build appbundle --flavor production`
- Write `mobile/README.md`: setup instructions, flavor config, test instructions
- Prepare `docs/mobile_test_protocol.md`: clinical validation protocol for test accuracy

---

## 👤 Person D — Clinician Dashboard, DevOps & Project Integration

**Role:** Full-Stack + DevOps — Clinician web dashboard, infrastructure, integration testing, documentation, demo packaging

---

### D1. Dashboard Project Setup
- Scaffold `dashboard/` as a React + TypeScript + Vite project
- Dependencies: React 18, TypeScript, TailwindCSS, shadcn/ui, Recharts, D3.js, React Query, Zustand, React Router v6, Axios, date-fns
- Set up ESLint + Prettier + Husky pre-commit hooks
- Configure Vite proxy for local API development
- Set up `dashboard/src/` structure: `pages/`, `components/`, `hooks/`, `services/`, `store/`, `types/`
- Push to `feature/D-setup` → PR into `dev`

---

### D2. Authentication & Role-Based UI Shell
- Implement `dashboard/src/pages/auth/`:
  - `LoginPage`: email/password form → JWT storage in httpOnly cookie
  - `ForgotPasswordPage`, `ResetPasswordPage`
- Implement `dashboard/src/components/layout/`:
  - `AppShell`: sidebar + top nav + main content area
  - `Sidebar`: navigation links filtered by role (admin sees all, doctor sees patients + reports, technician sees test queue)
  - `TopNav`: hospital name, user avatar, notifications bell with badge count
- Implement React Query auth wrapper — redirect to login if 401
- Role-based route guards: `<ProtectedRoute requiredRole="doctor" />`

---

### D3. Patient Management Dashboard
- Implement `dashboard/src/pages/patients/`:
  - `PatientListPage`: sortable/filterable table — columns: MRN, name, age, eye, last test date, MD score, alert status
  - Filter panel: by alert level, date range, disease severity
  - `PatientDetailPage`: full patient view with tabbed sections (Overview, Test History, Progression, Reports, Notes)
- Implement `dashboard/src/components/patients/`:
  - `AlertBadge`: color-coded (green/yellow/orange/red) alert level indicator
  - `MdTrendSparkline`: mini inline trend chart
  - `PatientSearchBar`: debounced search across MRN + name
- Bulk actions: download PDFs for multiple patients, export CSV

---

### D4. Visual Field Heatmap & Progression Visualization
- Implement `dashboard/src/components/visualizations/VFHeatmap.tsx`:
  - Render 54-point 24-2 grid with physiological blind spot
  - Color scale: green (35+ dB) → yellow (20 dB) → red (0 dB) — Humphrey-standard palette
  - Overlay options: raw sensitivity, progression rate (dB/year), uncertainty
  - Hover tooltip: dB value, location coordinates, progression rate, RNFB sector
  - Session selector: scrub through historical sessions with animated transition
- Implement `dashboard/src/components/visualizations/ProgressionChart.tsx`:
  - Multi-line chart using Recharts: MD score, VFI, PSD over time
  - Reference lines: clinically significant progression thresholds
  - Shaded confidence band around progression trend
  - Annotate test sessions with reliability dots (filled = reliable, hollow = unreliable)
- Implement `dashboard/src/components/visualizations/RNFBSectorChart.tsx`:
  - Polar sector chart showing RNFB bundle-level damage
  - Matches anatomical GNN sectors from Person A's model

---

### D5. Alert Management & Notification System
- Implement `dashboard/src/pages/alerts/AlertsPage.tsx`:
  - Inbox view: all pending alerts sorted by severity
  - Alert detail: patient info, triggering event, model confidence, recommended action
  - One-click acknowledge with notes field
  - Filter: by hospital, severity, date, doctor
- Implement real-time alert updates via WebSocket (`ws/alerts/live`)
- Browser push notifications (Web Push API) for urgent alerts — even when dashboard tab is inactive
- Implement `dashboard/src/components/alerts/AlertTimeline.tsx`: per-patient alert history with status changes

---

### D6. Clinic Analytics & Admin Panel
- Implement `dashboard/src/pages/analytics/ClinicAnalyticsPage.tsx`:
  - KPI cards: total active patients, tests this week, alert rate %, average MD score
  - Bar chart: tests per week over last 6 months
  - Pie chart: distribution by alert level
  - Heatmap calendar: test activity by day
- Implement `dashboard/src/pages/admin/`:
  - `UserManagementPage`: create/edit/deactivate clinician accounts
  - `ImportPage`: drag-and-drop CSV import UI with column mapping wizard and validation preview
  - `SystemHealthPage`: API status, ML service status, queue depth, last model version
  - `LicenseManagementPage`: hospital license key, feature flags, expiry date

---

### D7. Integration Testing & End-to-End Pipeline Validation
- Write end-to-end tests using Playwright: `e2e/`
  - Login → view patient → trigger test analysis → view report
  - Alert generation → acknowledgment flow
  - CSV import → patient appears in list
- Write backend integration tests: `backend/tests/integration/`
  - Full session submission → ML analysis → report generation
  - Alert threshold triggering
- Write mobile integration test: Detox or Flutter integration test for test-taking flow
- Set up test data fixtures: `data/fixtures/` with 10 synthetic patients and 20+ sessions per patient
- Run full pipeline stress test: 100 concurrent session submissions

---

### D8. Documentation, Demo Package & Commercial Readiness
- Write `README.md` (root): project overview, architecture diagram, quickstart
- Write `docs/clinical_validation_report.md`: model accuracy vs Rotterdam dataset (Person A's results)
- Write `docs/hospital_onboarding_guide.md`: step-by-step setup, staff training, device requirements
- Write `docs/data_privacy_compliance.md`: DISHA (India Digital Health) compliance checklist, data residency, encryption-at-rest, audit log coverage
- Create `demo/` package:
  - `demo/seed_demo_hospital.sh`: script to spin up a demo environment with realistic synthetic patients
  - `demo/demo_walkthrough.md`: scripted demo narrative for sales calls
  - Pre-recorded demo video (screen recording with narration script in `demo/narration_script.md`)
- Write `docs/pricing_and_licensing.md`: suggested SaaS pricing tiers (per-hospital annual license), source code license terms
- Create `CHANGELOG.md`, `CONTRIBUTING.md`, `LICENSE.md`
- Final deliverable: tag `v1.0.0` on `main`, create GitHub Release with compiled APK, Docker images, and documentation bundle

---

## 🗃️ Shared Resources & Conventions

### Naming & Code Standards
- Python: PEP8, type hints everywhere, docstrings on all public functions
- Dart: Effective Dart, null safety enabled
- TypeScript: strict mode, no `any`
- Commit messages: `type(scope): description` — e.g., `feat(gnn): add bundle coherence loss`

### Environment Variables (`.env.example` template)
```
DATABASE_URL=postgresql://user:pass@localhost:5432/glaucoma_db
REDIS_URL=redis://localhost:6379/0
JWT_SECRET=change_me_in_production
ML_MODEL_PATH=ml/checkpoints/best_model.pt
AWS_S3_BUCKET=glaucoma-reports
FIREBASE_SERVER_KEY=...  # for mobile push notifications
ENVIRONMENT=development
```

### Key Milestones & PR Checkpoints
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

---

## 💡 Differentiation & Commercial Value Notes

> These points should be highlighted in the demo and documentation for maximum buyer impact:

1. **Neuroanatomically-constrained model**: Unlike black-box ML, this model's structure mirrors peer-reviewed glaucoma anatomy (Garway-Heath map). Clinicians can trust it.
2. **Sample efficiency**: Detects progression in fewer test sessions than PLR (standard of care) — quantified in validation report.
3. **₹0 hardware cost for patients**: Runs on any Android phone. Replaces ₹5–15 lakh hospital perimeter for monitoring (not diagnosis).
4. **Offline-capable mobile test**: Works in rural clinics with poor connectivity.
5. **Export-ready for research**: Full raw data export for academic partnerships (Rotterdam-compatible format) — increases institutional sales.
6. **India-first compliance**: DISHA-aligned data handling, Hindi/Tamil localization hooks in Flutter app.
