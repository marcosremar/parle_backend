# Project structure rules

This repository has a strict project root layout. **Do not create new files or directories in the repository root** unless explicitly requested by the user.

## Allowed root entries

Only the following entries should normally exist at the root of the repo:

- `README.md`
- `LICENSE.txt`
- `CONTRIBUTING.md`
- `main.sh`
- `pyproject.toml`
- `.pre-commit-config.yaml`
- `.coveragerc`
- `.gitignore`
- `.gitmodules`
- `.dockerignore`
- `.secrets.baseline`
- `.env`, `.env.example`
- `environment.yml`
- `requirements.txt`
- `requirements-dev.txt`
- `requirements-test.txt`
- `docker/`
- `config/`
- `docs/`
- `scripts/`
- `src/`
- `tests/`

Git internals like `.git/` and CI config like `.github/` are also allowed but should not be modified by tools unless explicitly requested.

## Where to create new files

- Documentation: always under `docs/` (or subdirectories like `docs/status/`).
- Automation scripts: always under `scripts/`.
- Tests: under `tests/` in the appropriate subdirectory.
- Configuration: under `config/` or as updates to existing config files.

If unsure, ask the user where to place a new file instead of using the project root.

## Temporary AI files

- Any temporary, exploratory, or AI-generated files (notes, scratch Markdown, analysis reports,
  throwaway scripts, etc.) must be created under the `tmp/` directory at the repository root.
- Never create such files in the project root.
- Prefer subdirectories inside `tmp/` (for example `tmp/notes/`, `tmp/reports/`,
  `tmp/experiments/`) to keep the folder organized.
