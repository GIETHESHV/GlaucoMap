# Contributing

## Branch Strategy
- `main` — protected, release-only
- `dev` — integration branch
- `feature/A-*`, `feature/B-*`, `feature/C-*`, `feature/D-*` — personal branches
- All PRs target `dev` with at least one reviewer
- Weekly sync: merge `dev` → `main` after integration tests pass

## Commit Convention
`type(scope): description`

Examples:
- `feat(gnn): add bundle coherence loss`
- `fix(api): handle missing sensitivity values`
- `docs(ml): update model architecture overview`

## Code Standards
- Python: PEP8, type hints everywhere, docstrings on all public functions
- Dart: Effective Dart, null safety enabled
- TypeScript: strict mode, no `any`

## PR Checklist
- [ ] Tests pass locally
- [ ] Type hints / types added
- [ ] Docstrings on new public functions
- [ ] No secrets committed
