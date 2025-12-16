from typing import Any
import json
import re


def normalize_message_to_str(message: Any) -> str:
    """LLM / LangChain 메시지나 content를 항상 str로 변환."""
    if message is None:
        return ""

    if hasattr(message, "content"):
        return normalize_message_to_str(message.content)

    if isinstance(message, str):
        return message

    if isinstance(message, list):
        parts = []
        for part in message:
            if isinstance(part, dict):
                if part.get("type") == "text" and "text" in part:
                    parts.append(str(part["text"]))
                else:
                    parts.append(str(part))
            else:
                parts.append(str(part))
        return "\n".join(parts)

    if isinstance(message, dict):
        try:
            return json.dumps(message, ensure_ascii=False)
        except TypeError:
            return str(message)

    return str(message)
