from typing import Any
import json
import re


def normalize_message_to_str(message: Any) -> str:
    """LLM / LangChain 메시지나 content를 항상 str로 변환."""
    if message is None:
        return ""

    # BaseMessage 같은 객체면 .content 다시 태워서 처리
    if hasattr(message, "content"):
        return normalize_message_to_str(message.content)

    # 이미 str이면 그대로
    if isinstance(message, str):
        return message

    # 멀티파트: [{"type": "text", "text": "..."}, ...]
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

    # dict인 경우 JSON 문자열로
    if isinstance(message, dict):
        try:
            return json.dumps(message, ensure_ascii=False)
        except TypeError:
            return str(message)

    # 나머지는 일단 문자열 캐스팅
    return str(message)
