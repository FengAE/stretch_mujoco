import json
import os

import pytest

from stretch_mujoco.agents import (
    LLMConfigError,
    LLMProviderConfig,
    LLMRequest,
    LLMTrigger,
    OpenAICompatibleProvider,
)


def write_config(tmp_path, payload, mode=0o600):
    path = tmp_path / "office_llm.local.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(mode)
    return path


def enabled_payload(**overrides):
    payload = {
        "enabled": True,
        "provider": "openai_compatible",
        "api_key": "dummy-test-key",
        "base_url": "https://llm.example.test/v1",
        "model": "office-test-model",
        "api_mode": "responses",
        "timeout_seconds": 12,
        "max_retries": 0,
    }
    payload.update(overrides)
    return payload


def test_local_config_loads_and_redacts_api_key(tmp_path) -> None:
    config = LLMProviderConfig.from_json(write_config(tmp_path, enabled_payload()))

    assert config.api_key == "dummy-test-key"
    assert config.safe_summary()["api_key"] == "***"
    assert "dummy-test-key" not in repr(config)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission check")
def test_enabled_config_requires_private_file_permissions(tmp_path) -> None:
    path = write_config(tmp_path, enabled_payload(), mode=0o644)

    with pytest.raises(LLMConfigError, match="chmod 600"):
        LLMProviderConfig.from_json(path)


def test_provider_builds_responses_request_without_network(tmp_path) -> None:
    captured = {}

    def transport(url, headers, payload, timeout):
        captured.update(
            url=url,
            headers=headers,
            payload=payload,
            timeout=timeout,
        )
        return {"output_text": '{"dialogue": "I will prepare the report."}'}

    config = LLMProviderConfig.from_json(write_config(tmp_path, enabled_payload()))
    provider = OpenAICompatibleProvider(config, transport=transport)
    response = provider(
        LLMRequest(
            LLMTrigger.NEW_TASK,
            "employee_01",
            day=0,
            minute_of_day=550.0,
            context={"task": "prepare_report"},
        )
    )

    assert response == {"dialogue": "I will prepare the report."}
    assert captured["url"] == "https://llm.example.test/v1/responses"
    assert captured["headers"]["Authorization"] == "Bearer dummy-test-key"
    assert captured["payload"]["model"] == "office-test-model"
    assert "request_robot" in captured["payload"]["input"]
    assert captured["timeout"] == 12


def test_provider_parses_chat_completions_json(tmp_path) -> None:
    def transport(url, headers, payload, timeout):
        assert url.endswith("/chat/completions")
        assert payload["response_format"] == {"type": "json_object"}
        assert "json object" in payload["messages"][0]["content"].lower()
        return {"choices": [{"message": {"content": '```json\n{"dialogue": "Hello"}\n```'}}]}

    config = LLMProviderConfig.from_json(
        write_config(tmp_path, enabled_payload(api_mode="chat_completions"))
    )
    provider = OpenAICompatibleProvider(config, transport=transport)

    response = provider(LLMRequest(LLMTrigger.DIALOGUE, "employee_01", 0, 600.0))

    assert response == {"dialogue": "Hello"}


def test_day_start_prompt_contains_schedule_contract_and_context(tmp_path) -> None:
    captured = {}

    def transport(url, headers, payload, timeout):
        captured.update(payload)
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"schedule": [{"id": "work", "start": '
                        '"09:00", "end": "18:00", "activity": "work", '
                        '"location": "workstation_right"}]}'
                    }
                }
            ]
        }

    config = LLMProviderConfig.from_json(
        write_config(tmp_path, enabled_payload(api_mode="chat_completions"))
    )
    provider = OpenAICompatibleProvider(config, transport=transport)

    provider(
        LLMRequest(
            LLMTrigger.DAY_START,
            "employee_01",
            0,
            540.0,
            {"valid_objects": {"Workstation": ["workstation_right"]}},
        )
    )

    prompt = captured["messages"][1]["content"]
    assert "09:00-18:00" in prompt
    assert "workstation_right" in prompt
    assert '"schedule"' in prompt
