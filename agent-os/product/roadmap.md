# Roadmap: FormCoach

> Development phases. Each phase ends in something demoable.

## Phase 1: MVP (must-have for launch)

1. [x] **Video upload + pose extraction:** upload a squat video, run MediaPipe Pose, export per-frame keypoints. `S`
2. [x] **Skeleton overlay:** render joints and bones on the video for visual verification. `S`
3. [x] **Rep detection:** detect squat reps from hip/knee angle cycles. `M`
4. [x] **Form checks:** flag knee valgus and depth (above parallel) per rep; output a structured pose-summary JSON. `M`
5. [ ] **Labeled eval set:** hand-label 50 clips (rep count + form issues); script to report accuracy. `M`
6. [ ] **Live mode:** browser webcam using MediaPipe for Web, a rep state machine, and on-screen + audio "depth hit" cues; LLM coaching runs on the set summary afterward. Depth thresholds come from the shared config file, so Python and browser use the same numbers. `L`
7. [ ] **LLM coach:** generate feedback from the pose summary only. `S`
8. [ ] **Principle-based critique loop:** draft, critique against `coach_principles.md`, revise (max 2 rounds). `M`
9. [ ] **Feedback eval:** measure principle pass rate with vs. without the critique loop. `M`
10. [ ] **API + simple web UI:** FastAPI endpoint and upload page that shows overlay video + feedback. `M`
11. [ ] **Deploy:** containerize and host a public demo. `S`

## Phase 2: Post-Launch

1. [ ] **More lifts:** deadlift and bench press checks. `L`
2. [ ] **Notion integration:** planner agent logs the workout and schedules the next session. `M`
3. [ ] **Workout history:** per-user history and trend view (e.g., depth improving over weeks). `M`
4. [ ] **Camera-angle guidance:** warn when the angle or lighting is too poor for reliable tracking. `S`
5. [ ] **Mobile capture:** record directly in a mobile-friendly web app. `M`

> Effort: `XS` 1 day · `S` 2–3 days · `M` 1 week · `L` 2+ weeks
