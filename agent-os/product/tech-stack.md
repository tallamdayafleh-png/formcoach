# Tech Stack: FormCoach

> Technology choices. The coding agent should follow these unless a decision is recorded otherwise.

## Overview

| Category | Technology | Purpose |
|----------|-----------|---------|
| Language | Python 3.11+ | CV pipeline, agent, and backend |
| Pose estimation | MediaPipe Pose | Per-frame body keypoints |
| Video processing | OpenCV | Frame reading, skeleton overlay rendering |
| Numerics | NumPy | Joint-angle math, signal smoothing for rep detection |
| LLM | Hosted LLM API (Claude or similar) via a single `call_llm()` wrapper | Coaching feedback, critique, revision |
| Backend | FastAPI | Upload endpoint, analysis API |
| Frontend | Next.js (React + TypeScript) | Upload page, overlay video, feedback display |
| Database | SQLite (dev) → PostgreSQL (prod) | Users, sessions, results |
| Integrations | Notion API (Phase 2) | Workout logging and scheduling |
| Testing | pytest | Unit tests for angle math and rep detection |
| Evaluation | Custom Python scripts + labeled clip set | Rep accuracy and principle pass rate |
| Packaging | Docker | Reproducible builds |
| Hosting | Render or Railway (backend), Vercel (frontend) | Public demo |
| CI | GitHub Actions | Tests and lint on every PR |

## Conventions

- The LLM never sees raw video, only the structured pose-summary JSON.
- All LLM calls go through `call_llm()` so the provider can be swapped.
- API keys live in `.env` and are never committed.
- Lint/format with `ruff`; type hints required on public functions.

## Repo Layout

```
formcoach/
├── agent-os/product/      # mission.md, roadmap.md, tech-stack.md
├── cv/                    # pose extraction, angles, rep detection, form checks
├── coach/                 # LLM coach, critique loop, coach_principles.md
├── api/                   # FastAPI app
├── web/                   # Next.js frontend
├── eval/                  # labeled clips metadata + eval scripts
└── tests/
```
