# AGENTS.md — (Image Recognition) Repository Guide

Scope: This file governs the entire repository.

Read this first if you’re contributing, reviewing, or acting as an automated coding agent.

## Intent & Principles

- SOLID, KISS, YAGNI
- Security by default: least privilege access to local hardware (camera feeds).
- Testability: modular boundaries, deterministic components, fast tests first (via `pytest`).
- Clarity: idiomatic Python naming (PEP 8), minimal non‑obvious comments only.

## Expectations for Agents/Contributors

- Skim existing documentation for architecture context before coding.
- Drive all planning via GitHub Issues (no in‑repo trackers).
- Keep changes small and focused; propose ADRs for deviations.
- Add/Update tests for essential behaviors you change or add (e.g., in `.github/test/test_app.py`).
- For each new feature, add both unit and integration tests when feasible. Integration tests are as important as unit tests and should exercise end-to-end behavior without relying on brittle environment assumptions.
- Structured logging only; avoid standard `print()` statements in production code.

## Session Handoff Protocol (GitHub Issues)

- Start: pick a ready P0 issue, self‑assign, post a “Session Start” plan.
- During: post concise updates at milestones; adjust labels as needed.
- End: post “What landed” + “Next steps” and update labels/boards.
- If behavior/architecture changed, update documentation in the same commit.

### Task Tooling (GitHub)

- Windows PowerShell (preferred on Windows):
  - Pick a ready P0 task and mark it in‑progress: `pwsh -f tools/agents/session-start.ps1 [-AssignSelf]`
  - Update status/comment: `pwsh -f tools/agents/session-update.ps1 -Issue <#> -Status <ready|in-progress|blocked|done> [-WhatFile md] [-NextFile md] [-Close] [-AssignSelf]`
  - Quickly show the top ready P0: `pwsh -f tools/agents/pick-task.ps1`

- Bash (legacy WSL2 tooling still available):
  - `bash tools/agents/session-start.sh`
  - `bash tools/agents/session-update.sh --issue <#> --status <...>`
  - `bash tools/agents/pick-task.sh`

- Note: If CRLF line-endings cause issues, prefer the PowerShell versions on Windows.
  All tools read `GITHUB_TOKEN` (or `tools/agents/.env`, or `$HOME/.config/XXXX/agent.env`, or a local token file). On Windows, the scripts also probe `F:\WIN_TOKEN.txt`.

## Code Organization

Solution layout:

- `.github/` — Contains CI/CD GitHub Actions workflows (`main.yml`) and tests (`test_app.py`).
- `Monkey/` — Visual interface assets mapping to inference states (`monkeyknow.png`, `monkeythink.png`).
- `app.py` — Main entry point containing camera initialization, UI routing, and MediaPipe bindings.
- `hand_landmarker.task` — Pre-trained MediaPipe ML model asset.

### File Layout Rules (Vertical Slice)

- One class/functional module per file.
- Vertical slices first: organize code by the architectural pipeline components (Agents):
  - **Vision Perception Agent (VPA):** Handles hardware interfacing and frame normalization.
  - **Landmark Detection Agent (LDA):** Manages spatial tracking using `hand_landmarker.task`.
  - **Monkey Inference Agent (MIA):** Executes gesture logic and determines "Thinking" vs "Knowing" states.
  - **Interaction & Feedback Agent (IFA):** Manages OpenCV visual rendering and overlays.
- Avoid maintaining monolithic scripts. Aim to abstract the aforementioned agents from `app.py` into distinct modular files.

## Workflow & Quality

- Feature toggles/configuration are mandatory for runtime‑conditional behavior.
- Follow Python conventions; keep functions single‑purpose.
- Error handling: Use `try/except` blocks; return typed results where possible; log structured context; never swallow exceptions.
- Source control: push cohesive changes to master after green build/tests.
- Keep the repo clean: do not commit generated artifacts or logs. `.gitignore` must exclude `__pycache__/`, `*.pyc`, `logs/`, and environment files.
- **CRITICAL RESTRICTION:** Under no circumstances should `agent.md` or `AGENTS.md` be committed or pushed to the public GitHub repository. Ensure this file remains explicitly restricted to the local environment.

### Roadmap & Priorities

- **P0:** Stabilize the core OpenCV stream and MediaPipe hand/facial tracking loop.
- **P1:** Refactor the existing logic in `app.py` into the distinct VPA, LDA, MIA, and IFA class abstractions for better testability.
- **P2:** Expand MIA's inference capabilities to recognize more complex multi-hand gestures beyond "index near mouth" and "index up".
- **P3:** Increase coverage in `.github/test/` to include mock camera feeds for isolated CI tests.

## Coding Standards

- Python 3.x enabled; strict type hinting is preferred.
- OpenCV and MediaPipe tasks must release resources cleanly (e.g., `cap.release()`, `hand_tracker.close()`, and `cv2.destroyAllWindows()`).
- No dead code: remove unused fields/methods/imports and scaffolding when no longer used.
- Naming: `snake_case` for variables, locals, and functions. `PascalCase` for classes.
- Logging: structured using Python's standard `logging` module with appropriate log levels; no raw `print()` statements in production code.

## Documentation Rules

- Central doc is the source of truth. Keep it current when architecture shifts.
- All task/progress tracking in GitHub Issues.

## Ambiguity

- Prefer the simplest design that satisfies current requirements.
- If multiple options exist, document a brief rationale and link architectural docs.
- User instructions take precedence over the central doc.
