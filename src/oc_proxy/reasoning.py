from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import CallTypes

OPENCODE_SESSION_HEADER = "x-opencode-session"

_EXPLICIT_SESSION_HEADERS = (
    OPENCODE_SESSION_HEADER,
    "x-litellm-session-id",
    "x-litellm-trace-id",
    "thread-id",
    "session-id",
    "session_id",
)

_METADATA_SESSION_KEYS = (
    "session_id",
    "sessionId",
    "thread_id",
    "threadId",
    "conversation_id",
    "conversationId",
)


def _patch_litellm_empty_stream_choices() -> None:
    try:
        from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
        from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
            AnthropicStreamWrapper,
        )
        from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
            LiteLLMAnthropicMessagesAdapter,
        )
    except ImportError:
        return

    original_should_start = AnthropicStreamWrapper._should_start_new_content_block
    original_translate = LiteLLMAnthropicMessagesAdapter.translate_streaming_openai_response_to_anthropic
    original_raise_on_repetition = CustomStreamWrapper.raise_on_model_repetition

    if getattr(original_translate, "_oc_proxy_empty_choices_patch", False):
        return

    def patched_should_start(self: Any, chunk: Any) -> bool:
        if _has_empty_choices(chunk):
            return False
        return original_should_start(self, chunk)

    def patched_translate(self: Any, response: Any, current_content_block_index: int) -> dict[str, Any]:
        if _has_empty_choices(response):
            return {
                "type": "content_block_delta",
                "index": current_content_block_index,
                "delta": {"type": "text_delta", "text": ""},
            }
        return original_translate(self, response, current_content_block_index)

    def patched_raise_on_repetition(self: Any) -> None:
        if len(self.chunks) < 2:
            return
        if _has_empty_choices(self.chunks[-1]) or _has_empty_choices(self.chunks[-2]):
            self._repeated_messages_count = 1
            return
        return original_raise_on_repetition(self)

    patched_translate._oc_proxy_empty_choices_patch = True  # type: ignore[attr-defined]
    CustomStreamWrapper.raise_on_model_repetition = patched_raise_on_repetition
    AnthropicStreamWrapper._should_start_new_content_block = patched_should_start
    LiteLLMAnthropicMessagesAdapter.translate_streaming_openai_response_to_anthropic = patched_translate


class DeepSeekReasoningContentCallback(CustomLogger):
    """Preserve DeepSeek reasoning metadata and drop Anthropic-only replay fields."""

    async def async_pre_call_deployment_hook(self, kwargs: dict[str, Any], call_type: CallTypes | None) -> dict[str, Any]:
        if call_type not in {CallTypes.completion, CallTypes.acompletion}:
            return kwargs

        model = str(kwargs.get("model") or "")
        should_preserve_reasoning = model.startswith(("deepseek-v4", "kimi-k2"))
        messages = kwargs.get("messages")
        if not isinstance(messages, list):
            return kwargs

        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            thinking_blocks = message.pop("thinking_blocks", None)
            reasoning_content = _thinking_blocks_to_reasoning_content(thinking_blocks) if should_preserve_reasoning else None
            if reasoning_content:
                message["reasoning_content"] = reasoning_content
            if message.get("tool_calls") and message.get("content") is None:
                message["content"] = ""
        return kwargs

    async def async_post_call_streaming_iterator_hook(self, user_api_key_dict: Any, response: Any, request_data: dict) -> Any:
        model = str(request_data.get("model") or "")
        async for item in response:
            if model == "minimax-m2.7" and _has_empty_choices(item):
                continue
            yield item


def _has_empty_choices(item: Any) -> bool:
    choices = getattr(item, "choices", None)
    return isinstance(choices, list) and not choices


def _thinking_blocks_to_reasoning_content(thinking_blocks: Any) -> str | None:
    if not isinstance(thinking_blocks, list):
        return None

    parts: list[str] = []
    for block in thinking_blocks:
        if isinstance(block, dict) and block.get("type") == "thinking" and block.get("thinking"):
            parts.append(str(block["thinking"]))
    return "\n".join(parts) or None


def _normalized_headers(headers: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    if isinstance(headers, dict):
        for key, value in headers.items():
            text = str(value).strip()
            if text:
                result[str(key).lower()] = text
    return result


def _header_session_id(headers: Any) -> str | None:
    normalized = _normalized_headers(headers)
    for name in _EXPLICIT_SESSION_HEADERS:
        value = normalized.get(name.lower())
        if value:
            return value
    for key, value in normalized.items():
        if key.startswith("x-") and key.endswith("-session-id") and value:
            return value
    return None


def _metadata_session_id(metadata: Any, _depth: int = 0) -> str | None:
    if not isinstance(metadata, dict) or _depth > 3:
        return None
    for key, value in metadata.items():
        if key in _METADATA_SESSION_KEYS and isinstance(value, str) and value.strip():
            return value.strip()
    for value in metadata.values():
        found = _metadata_session_id(value, _depth + 1)
        if found:
            return found
    return None


def _conversation_anchor(messages: Any) -> str:
    if not isinstance(messages, list):
        return "[]"
    anchor_messages: list[Any] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") == "tool":
            continue
        anchor_messages.append(message)
        if len(anchor_messages) >= 2:
            break
    return json.dumps(anchor_messages, sort_keys=True, separators=(",", ":"), default=str)


def _derive_session_id(messages: Any) -> str:
    digest = hashlib.sha256(_conversation_anchor(messages).encode("utf-8")).digest()[:16]
    return str(uuid.UUID(bytes=digest))


def resolve_opencode_session(kwargs: dict[str, Any]) -> str:
    metadata = kwargs.get("metadata")
    metadata_headers = metadata.get("headers") if isinstance(metadata, dict) else None
    session_id = (
        _header_session_id(metadata_headers)
        or _header_session_id(kwargs.get("headers"))
        or _header_session_id(kwargs.get("extra_headers"))
        or _metadata_session_id(metadata)
    )
    return session_id or _derive_session_id(kwargs.get("messages"))


class OpenCodeSessionHeaderCallback(CustomLogger):
    """Attach a stable OpenCode Go session identifier to every upstream request."""

    async def async_pre_call_deployment_hook(self, kwargs: dict[str, Any], call_type: CallTypes | None) -> dict[str, Any]:
        if call_type not in {CallTypes.completion, CallTypes.acompletion}:
            return kwargs

        session_id = resolve_opencode_session(kwargs)
        extra_headers = kwargs.get("extra_headers")
        if not isinstance(extra_headers, dict):
            extra_headers = {}
            kwargs["extra_headers"] = extra_headers
        extra_headers.setdefault(OPENCODE_SESSION_HEADER, session_id)
        return kwargs


deepseek_reasoning_content_callback = DeepSeekReasoningContentCallback()
opencode_session_header_callback = OpenCodeSessionHeaderCallback()
_patch_litellm_empty_stream_choices()
