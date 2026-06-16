from sglang.srt.entrypoints.openai.serving_chat import (
    DSTAR_VL_DEFAULT_SYSTEM_PROMPT,
    _maybe_prepend_dstar_default_system_message,
)


def test_dstar_default_system_prompt_is_prepended_when_missing():
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]

    normalized = _maybe_prepend_dstar_default_system_message(messages, "dstar_vl")

    assert normalized[0] == {
        "role": "system",
        "content": DSTAR_VL_DEFAULT_SYSTEM_PROMPT,
    }
    assert normalized[1:] == messages


def test_dstar_default_system_prompt_is_not_duplicated():
    messages = [
        {"role": "system", "content": "custom"},
        {"role": "user", "content": "Hello"},
    ]

    normalized = _maybe_prepend_dstar_default_system_message(messages, "dstar_vl")

    assert normalized == messages


def test_non_dstar_models_are_unchanged():
    messages = [{"role": "user", "content": "Hello"}]

    normalized = _maybe_prepend_dstar_default_system_message(messages, "qwen3_vl")

    assert normalized == messages
