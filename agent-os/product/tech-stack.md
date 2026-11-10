# Tech Stack: FormCoach

> Technology choices. The coding agent should follow these unless a decision is recorded otherwise.

## Overview

| Category | Technology | Purpose |
|----------|-----------|---------|
| Language | Python 3.10+ | CV pipeline, agent, and backend |
| Pose estimation | MediaPipe Pose Landmarker (Tasks API, `mediapipe` 1.x) | Per-frame body keypoints |
| Live pose estimation | MediaPipe for Web (`@mediapipe/tasks-vision` PoseLandmarker) | In-browser webcam keypoints for live mode |
| Live cues | Web Audio API / SpeechSynthesis | Audio "depth hit" cues in live mode |
| Shared thresholds | `config/form_thresholds.json` | One source of truth for depth / rep thresholds, read by Python and the browser |
| Video processing | OpenCV (`opencv-contrib-python`, installed by mediapipe) | Frame reading, skeleton overlay rendering |
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
- Lint/format with `ruff`; type hints required on public functions. Ruff config lives in `pyproject.toml`.
- Use the MediaPipe Tasks API (`PoseLandmarker`); the legacy `mp.solutions` API is removed in mediapipe 1.x.
- Don't add `opencv-python` to requirements; it conflicts with the `opencv-contrib-python` that mediapipe installs.
- Dependencies are pinned in `requirements.txt` (runtime) and `requirements-dev.txt` (adds pytest, ruff); develop in a `.venv` virtual environment.
- Model files (`models/`), local clips (`data/`), and generated output (`out/`) are gitignored, never committed.
- Form and rep thresholds (depth angles, rep hysteresis, minimum rep duration) live only in `config/form_thresholds.json`. Python (`cv/`) and the browser (`web/`) both load that file; never hard-code these numbers in either. JSON is used because both languages parse it natively with no extra dependency.
- Signal processing in `cv/` (smoothing, gap handling, rep counting) is causal: each frame uses only past frames, via one-frame-at-a-time `update()` classes (`CausalSmoother`, `RepCounter`), so the browser can port them line for line and live and post-set results match. Smoothing is an EMA with a time constant in seconds (`smooth_tau_s`), not a fixed per-frame alpha, so it behaves the same at any frame rate.
- The camera view (side / front / angled) is classified once from the first seconds (`cv/view.py`), and it decides which checks run: depth needs a side view, knee valgus needs a front view. Checks that can't run are reported in the pose summary as `not_assessed` with a reason, never guessed. Reps are counted from the 2D knee angle in side view and from hip height relative to recent standing height in front view.
- Mirrored (selfie) video: iPhone front-camera clips are mirror images when "Mirror Front Camera" is on, which makes MediaPipe swap left and right. Extraction detects the front camera from the file's metadata (`--mirrored auto`, overridable with `yes`/`no`) and swaps left/right labels, so "left_knee" is always the lifter's real left knee. Positions stay as recorded so the overlay lines up; the keypoints file records `meta.mirrored`. In live mode, frames from `getUserMedia` are not mirrored (only the on-screen preview usually is), so no swap is needed there.
- `python -m cv.analyze <video>` runs the whole CV pipeline and writes the pose-summary JSON (`out/<name>_summary.json`), the coach's only input.
- Live mode runs pose estimation in the browser; webcam video never leaves the device. Only the set summary is sent to the backend for LLM coaching.

## Repo Layout

```
formcoach/
├── agent-os/product/      # mission.md, roadmap.md, tech-stack.md
├── config/                # form_thresholds.json, shared by Python and browser
├── cv/                    # pose extraction, angles, rep detection, form checks
├── coach/                 # LLM coach, critique loop, coach_principles.md
├── api/                   # FastAPI app
├── web/                   # Next.js frontend
├── eval/                  # labeled clips metadata + eval scripts
├── tests/
├── models/                # downloaded MediaPipe .task files (gitignored)
├── data/                  # local video clips (gitignored)
├── out/                   # generated keypoints JSON + overlay videos (gitignored)
├── pyproject.toml         # project metadata + ruff config
├── requirements.txt       # pinned runtime deps
└── requirements-dev.txt   # + pytest, ruff
```
