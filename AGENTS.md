# AGENTS.md

Operational guide for coding agents in `D:\NCKH\HR_Robot`.

## 1) Project Overview

- Primary language: Python (3.12 in local venv).
- Domain: face recognition + attendance tracking + FastAPI dashboard API.
- Runtime architecture: camera -> detection/recognition -> attendance manager -> API/WebSocket.
- Database: SQLite by default (`data/hr_robot.db`), configurable via `DATABASE_URL`.
- Dependencies are managed with `requirements.txt` (no Poetry/Pipenv config present).

## 2) Key Paths

- `src/pipeline.py`: CLI entrypoint and live runtime orchestration.
- `src/api_server.py`: REST + WebSocket endpoints for dashboard.
- `src/attendance/attendance_manager.py`: attendance state machine.
- `src/database/models.py`: SQLAlchemy models and session helpers.
- `src/database/repository.py`: persistence/query layer.
- `config/settings.py`: env-based settings and defaults.
- `tests/`: pytest suites for repository and attendance behavior.
- `scripts/`: utility scripts (download models, init/migrate DB, diagnostics).

## 3) Environment Setup

Run from repository root.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_models.py
python scripts/init_database.py
```

Linux/macOS activation alternative:

```bash
source venv/bin/activate
```

## 4) Build, Lint, Test Commands

There is no dedicated compile/build pipeline for this repo.

### Run application

```bash
python src/pipeline.py
python src/pipeline.py --register
python src/pipeline.py --register --name "Nguyen Van A"
python src/pipeline.py --list
python src/pipeline.py --benchmark
```

### Run tests (all)

```bash
pytest -q
pytest -q tests
```

### Run a single test file

```bash
pytest -q tests/test_repository_attendance.py
pytest -q tests/test_attendance_manager.py
```

### Run a single test case (node id)

```bash
pytest -q tests/test_repository_attendance.py::TestLogCheckin::test_creates_open_log
pytest -q tests/test_attendance_manager.py::TestCheckinFlow::test_two_frames_triggers_checkin
```

### Filter tests by keyword

```bash
pytest -q -k checkin
pytest -q -k overload
```

### Useful debugging flags

```bash
pytest -q -x
pytest -q --maxfail=1
pytest -q -vv
```

### Practical quality checks (since no linter config is committed)

```bash
python -m compileall src tests
pytest -q
```

## 5) Tooling Status (Important)

- No `pyproject.toml` found.
- No `setup.cfg`, `tox.ini`, `.flake8`, `mypy.ini`, or `pytest.ini` found.
- No committed lint/format/type-check command is canonical yet.
- Keep edits consistent with surrounding style unless user asks to introduce tooling.

## 6) Code Style Conventions

Use existing code conventions in `src/` and `tests/`.

### Imports

- Order imports as: standard library, third-party, local modules.
- Prefer explicit imports; avoid wildcard imports.
- Keep local imports from `src.*` and `config.*` explicit.
- `sys.path` injection is used in entry scripts; avoid spreading this pattern unnecessarily.

### Formatting and structure

- Use 4-space indentation.
- Keep functions focused and readable.
- Prefer clarity over micro-optimizations.
- Follow existing quote style in file being edited.
- Add comments only for non-obvious logic.

### Type hints

- Add/maintain type hints for new or modified public methods.
- Prefer built-in generics (`list[...]`, `dict[...]`, `tuple[...]`).
- Use `Optional[T]` when `None` is valid.
- Keep return shapes stable for methods used by API/websocket layers.

### Naming

- `snake_case` for functions/variables/modules.
- `PascalCase` for classes.
- `UPPER_SNAKE_CASE` for constants and state labels.
- Use domain names (`member_id`, `headcount`, `check_in_time`) consistently.

### Error handling

- Handle exceptions around IO/DB/network boundaries.
- Keep exception scope narrow.
- Current code logs many runtime errors with `print`; preserve behavior unless refactor is requested.
- Repository methods returning `None` for not-found/already-closed cases are expected behavior.

### Database patterns

- Use `session_scope()` in runtime/service code for transactions.
- Use repository methods for persistence (single source of truth).
- Call `session.flush()` when IDs or immediate DB state are required.
- Avoid duplicating raw query logic across modules.

### Time handling

- Use `datetime.utcnow()` consistently for server-side timestamps.
- Serialize datetime values to ISO at API/WebSocket boundary.
- Be explicit with date-range boundaries (`start <= t < end`).

### Concurrency and shared state

- Respect lock usage in shared runtime objects.
- `AttendanceManager` and websocket client registry are stateful; preserve thread-safe patterns.
- Return copies/snapshots of mutable shared collections when exposing state.

## 7) Testing Conventions

- Framework: pytest.
- Tests use in-memory SQLite for isolation.
- Enable `PRAGMA foreign_keys=ON` in test DB setup.
- Use fixtures from `tests/conftest.py` for reusable setup.
- Keep tests deterministic with explicit timestamps.
- For attendance state-machine changes, run `tests/test_attendance_manager.py`.
- For repository/data changes, run `tests/test_repository_attendance.py`.

## 8) Cursor and Copilot Rules

Checked rule locations in this repo:

- No `.cursorrules` file found.
- No `.cursor/rules/` directory found.
- No `.github/copilot-instructions.md` found.

If these files are added later, treat them as repo instructions and update this document.

## 9) Agent Guardrails

- Make minimal, targeted edits.
- Do not commit secrets or local runtime artifacts (`.env`, `data/*.db*`, model binaries).
- Do not refactor unrelated files during focused fixes.
- Prefer repository-level consistency over personal style.
- Validate with the smallest relevant test command, then broader tests when needed.
