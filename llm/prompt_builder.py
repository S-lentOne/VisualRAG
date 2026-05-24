from dataclasses import dataclass

SYSTEM_PROMPT = """You are a scene analysis assistant powered by a computer vision system.

The detected objects listed in each message are GROUND TRUTH — they were physically observed by the camera. Never contradict, omit, or add to this list when asked about what was seen. If asked to list detected objects, reproduce the full detection list exactly.

For scene descriptions, reason naturally about the activity and context in 3 to 5 sentences. For direct factual questions ("were there headphones?", "list all items"), answer strictly from the detection data provided."""

VIDEO_SYSTEM_PROMPT = """You are a scene analysis assistant reviewing a recorded video session.

The detected objects and frequency counts provided are GROUND TRUTH from a computer vision system that analyzed multiple frames. Every object listed was physically observed on camera at least once. Never contradict or omit items from this list.

When asked to list items, reproduce the full detection list. When asked whether a specific object was present, check the detection list — if it appears there, confirm it; if not, say it was not detected."""


@dataclass
class PromptPackage:
    messages: list
    user_prompt: str


def build_prompt(
    detections: list,
    static_context: str = "",
    episodic_context: str = "",
    user_context: str = "",
    is_video_session: bool = False,
) -> PromptPackage:
    system = VIDEO_SYSTEM_PROMPT if is_video_session else SYSTEM_PROMPT

    if not detections:
        detection_line = "No objects detected."
        detection_list = "None"
    else:
        # two formats: inline for the prompt header, numbered list for explicit grounding
        parts = [f"{d['label']} ({d['score']:.0%})" for d in sorted(detections, key=lambda x: -x["score"])]
        detection_line = ", ".join(parts)
        detection_list  = "\n".join(f"{i+1}. {d['label']}" for i, d in enumerate(sorted(detections, key=lambda x: -x["score"])))

    sections = [
        f"Detected objects (complete list — treat as ground truth):\n{detection_list}",
        f"Confidence scores: {detection_line}",
    ]

    if static_context.strip():
        sections.append(f"Background knowledge:\n{static_context}")

    if episodic_context.strip():
        sections.append(f"Scene memory:\n{episodic_context}")

    if user_context.strip():
        sections.append(f"User context:\n{user_context}")

    sections.append("Describe what is happening in this scene.")
    user_prompt = "\n\n".join(sections)

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_prompt},
    ]

    return PromptPackage(messages=messages, user_prompt=user_prompt)


def build_followup_prompt(history: list, followup_question: str) -> list:
    return history + [{"role": "user", "content": followup_question}]