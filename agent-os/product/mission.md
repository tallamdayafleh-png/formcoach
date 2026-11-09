# Mission: FormCoach

> An AI gym coach that watches your set through your phone camera and tells you, in plain English, exactly what to fix on your next one.

## Problem

Most people who lift train without a coach. Bad form (knee cave, rounded back, shallow depth) goes unnoticed until it causes a plateau or injury. Filming yourself helps, but you still have to know what to look for, and existing apps either count reps without explaining anything or give generic, ungrounded advice.

## Target Users

- **Primary:** College students and beginner-to-intermediate lifters training without a coach.
- **Secondary:** Self-coached intermediate lifters who film sets and want objective checks on depth and joint angles.

## Solution

FormCoach pairs computer vision with an LLM agent:

1. **Pose estimation** tracks joints frame by frame and measures angles and reps objectively.
2. **An LLM coach** turns those measurements into one or two specific, actionable cues tied to a rep number and body part.
3. **A planner agent** logs the workout and schedules the next session in Notion.

### Differentiators

- **Grounded feedback:** every cue traces back to measured pose data, never guesswork.
- **Self-checking coach:** feedback is critiqued against a written set of coaching principles (`coach_principles.md`) and revised before the user sees it.
- **Measured quality:** rep-count accuracy and feedback quality are evaluated on a labeled test set.

## Non-Goals

- Diagnosing injuries or giving medical advice.
- Replacing in-person coaching for advanced or competitive lifters.
- Real-time, mid-rep audio cues (post-set feedback only in v1).

## Success Metrics

- ≥95% rep-count accuracy on a 50-clip labeled set.
- ≥90% of feedback outputs pass every coaching principle on the eval set.
- Feedback returned in under 15 seconds for a 30-second clip.
