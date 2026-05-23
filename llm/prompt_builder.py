from dataclasses import dataclass

SYSTEM_PROMPT = """You are a scene analysis assistant. You are given a list of objects detected in a scene by a computer vision model, along with relevant background knowledge about those objects, memory of past scenes, and optionally some context about the specific user you are talking to.

Your job is to describe what is likely happening in the scene, reason about the activity and context, and note any interesting changes or patterns compared to past observations.

Be concise — 3 to 5 sentences. Do not list the objects mechanically. Reason like a thoughtful observer."""


@dataclass
class PromptPackage:
    messages: list
    user_prompt: str


def build_prompt(
    detections: list,
    static_context: str = "",
    episodic_context: str = "",
    user_context: str = "",      # new — injected from UserProfile
) -> PromptPackage:
    if not detections:
        detection_line = "No objects detected."
    else:
        parts = [f"{d['label']} ({d['score']:.0%})" for d in sorted(detections, key=lambda x: -x["score"])]
        detection_line = ", ".join(parts)

    sections = [f"Detected objects: {detection_line}"]

    if static_context.strip():
        sections.append(f"Background knowledge:\n{static_context}")

    if episodic_context.strip():
        sections.append(f"Past scene memory:\n{episodic_context}")

    if user_context.strip():
        sections.append(f"User context:\n{user_context}")

    sections.append("Describe what is happening in this scene.")
    user_prompt = "\n\n".join(sections)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]

    return PromptPackage(messages=messages, user_prompt=user_prompt)


def build_followup_prompt(history: list, followup_question: str) -> list:
    return history + [{"role": "user", "content": followup_question}]