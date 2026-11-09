"""
Constitutional critique-and-revise loop for the Gym Form Coach.

Plug in any LLM provider by implementing `call_llm(prompt) -> str`.
"""

import json
from pathlib import Path

CONSTITUTION = (Path(__file__).parent / "coach_principles.md").read_text()
MAX_REVISIONS = 2


def call_llm(prompt: str) -> str:
    """Replace with your provider's SDK call (Anthropic, OpenAI, local model, etc.)."""
    raise NotImplementedError


def draft(pose_summary: dict) -> str:
    prompt = f"""You are a strength coach. Using ONLY this pose analysis, give the lifter feedback.

Pose analysis (JSON):
{json.dumps(pose_summary, indent=2)}

Feedback:"""
    return call_llm(prompt)


def critique(feedback: str, pose_summary: dict) -> dict:
    prompt = f"""Check the coaching feedback against every principle in the constitution.

CONSTITUTION:
{CONSTITUTION}

POSE ANALYSIS:
{json.dumps(pose_summary, indent=2)}

FEEDBACK:
{feedback}

Respond with JSON only:
{{"violations": [{{"principle": <number>, "issue": "<what's wrong>"}}], "passes": <true|false>}}"""
    return json.loads(call_llm(prompt))


def revise(feedback: str, violations: list, pose_summary: dict) -> str:
    prompt = f"""Rewrite the feedback to fix these constitution violations. Stay grounded in the pose data.

CONSTITUTION:
{CONSTITUTION}

POSE ANALYSIS:
{json.dumps(pose_summary, indent=2)}

ORIGINAL FEEDBACK:
{feedback}

VIOLATIONS:
{json.dumps(violations, indent=2)}

Revised feedback:"""
    return call_llm(prompt)


def coach(pose_summary: dict) -> dict:
    feedback = draft(pose_summary)
    history = []
    for _ in range(MAX_REVISIONS):
        result = critique(feedback, pose_summary)
        history.append({"feedback": feedback, "critique": result})
        if result["passes"]:
            break
        feedback = revise(feedback, result["violations"], pose_summary)
    # history (each draft + its critique) feeds the eval scripts and the demo
    return {"feedback": feedback, "history": history}


if __name__ == "__main__":
    example = {
        "exercise": "squat",
        "reps_detected": 5,
        "pose_confidence": 0.91,
        "flags": [
            {"rep": 4, "joint": "left_knee", "issue": "valgus", "angle_deg": 12},
            {"rep": 5, "joint": "hip", "issue": "above_parallel", "depth_deg": 95},
        ],
    }
    print(json.dumps(coach(example), indent=2))
