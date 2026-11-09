# FormCoach

AI gym form coach: computer vision (pose estimation) + an LLM coach that self-checks its feedback against written principles.

## Read first
- Product mission: @agent-os/product/mission.md
- Roadmap (current priorities): @agent-os/product/roadmap.md
- Tech stack and repo layout: @agent-os/product/tech-stack.md
- Coach runtime principles: @coach/coach_principles.md

## How to work in this repo
1. Before coding a roadmap item, write a short plan and wait for approval.
2. Build one roadmap item at a time; tick it off in roadmap.md when done.
3. Write pytest tests for all angle math, rep detection, and form checks.
4. Run `pytest` and `ruff check .` before saying a task is finished.
5. The LLM never receives raw video — only the pose-summary JSON.
6. All LLM calls go through `call_llm()` in coach/coach.py.
7. Never commit API keys; use `.env` (already in .gitignore).

## I'm learning
I'm a CS student building this for my resume and to learn CV + AI engineering. When you make a non-obvious design choice, explain it briefly (and use a diagram when explaining a concept) so I can talk about it in interviews.
