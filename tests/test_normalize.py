"""Tests for normalize module."""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from normalize import _parse_llm_corrections, llm_normalize, DEFAULT_PROMPT, MODEL


# --- _parse_llm_corrections tests ---


def test_parse_corrections_from_json():
    """Standard JSON with corrections list is parsed."""
    response = '{"corrections": [{"transcribed": "fondos", "correct": "Pandavas"}]}'
    result = _parse_llm_corrections(response)
    assert len(result) == 1
    assert result[0]["transcribed"] == "fondos"
    assert result[0]["correct"] == "Pandavas"


def test_parse_empty_corrections():
    """Empty corrections list returns []."""
    response = '{"corrections": []}'
    result = _parse_llm_corrections(response)
    assert result == []


def test_parse_empty_object():
    """Empty object (no corrections key) returns []."""
    response = '{}'
    result = _parse_llm_corrections(response)
    assert result == []


def test_parse_json_in_thinking_output():
    """JSON buried in thinking tags is extracted."""
    response = (
        '<think>\nLet me analyze the text for mishearings...\n</think>\n'
        '{"corrections": [{"transcribed": "goros", "correct": "Kauravas"}]}'
    )
    result = _parse_llm_corrections(response)
    assert len(result) == 1
    assert result[0]["transcribed"] == "goros"


def test_parse_json_in_code_block():
    """JSON in ```json code block is extracted."""
    response = (
        'Here are the corrections:\n'
        '```json\n'
        '{"corrections": [{"transcribed": "yudister", "correct": "Yudhishthira"}]}\n'
        '```'
    )
    result = _parse_llm_corrections(response)
    assert len(result) == 1
    assert result[0]["correct"] == "Yudhishthira"


def test_parse_plain_code_block():
    """JSON in plain ``` code block is extracted."""
    response = (
        'Result:\n'
        '```\n'
        '{"corrections": [{"transcribed": "dhrashtra", "correct": "Dhritarashtra"}]}\n'
        '```'
    )
    result = _parse_llm_corrections(response)
    assert len(result) == 1
    assert result[0]["correct"] == "Dhritarashtra"


def test_parse_drops_corrections_missing_keys():
    """Corrections missing 'transcribed' or 'correct' are dropped."""
    response = '{"corrections": [{"transcribed": "fondos"}, {"transcribed": "kuru", "correct": "Kuru"}]}'
    result = _parse_llm_corrections(response)
    assert len(result) == 1
    assert result[0]["correct"] == "Kuru"


def test_parse_drops_non_string_values():
    """Corrections with non-string values are dropped."""
    response = '{"corrections": [{"transcribed": "fondos", "correct": null}, {"transcribed": "kuru", "correct": "Kuru"}]}'
    result = _parse_llm_corrections(response)
    assert len(result) == 1
    assert result[0]["correct"] == "Kuru"


def test_parse_non_list_corrections_returns_empty():
    """Non-list corrections value returns []."""
    response = '{"corrections": "none"}'
    result = _parse_llm_corrections(response)
    assert result == []


def test_parse_garbage_raises():
    """Unparseable response raises ValueError."""
    import pytest
    with pytest.raises(ValueError, match="Could not parse LLM response"):
        _parse_llm_corrections("I don't understand the question, sorry!")


# --- llm_normalize tests ---


@patch("normalize._call_mlx")
def test_llm_normalize_returns_corrections(mock_call_mlx):
    """Successful end-to-end with mocked MLX backend."""
    mock_call_mlx.return_value = '{"corrections": [{"transcribed": "fondos", "correct": "Pandavas"}]}'
    corrections, entry = llm_normalize("The fondos went to war")
    assert len(corrections) == 1
    assert corrections[0]["transcribed"] == "fondos"


@patch("normalize._call_mlx")
def test_llm_normalize_empty_corrections(mock_call_mlx):
    """Model returns no corrections."""
    mock_call_mlx.return_value = '{"corrections": []}'
    corrections, entry = llm_normalize("Once upon a time")
    assert corrections == []


@patch("normalize._call_mlx")
def test_llm_normalize_timeout_raises(mock_call_mlx):
    """TimeoutError propagates when inference times out."""
    mock_call_mlx.side_effect = TimeoutError("timed out")
    import pytest
    with pytest.raises(TimeoutError):
        llm_normalize("some text")


@patch("normalize._call_mlx")
def test_llm_normalize_default_prompt(mock_call_mlx):
    """DEFAULT_PROMPT is used when none provided."""
    mock_call_mlx.return_value = '{"corrections": []}'
    _, _ = llm_normalize("test text")
    prompt_arg = mock_call_mlx.call_args[0][0]
    assert "Mahabharata" in prompt_arg
    assert "test text" in prompt_arg


@patch("normalize._call_mlx")
def test_llm_normalize_custom_prompt(mock_call_mlx):
    """Custom prompt passed through to MLX backend."""
    mock_call_mlx.return_value = '{"corrections": []}'
    custom = "Find errors in: {text}"
    _, _ = llm_normalize("hello", prompt=custom)
    prompt_arg = mock_call_mlx.call_args[0][0]
    assert prompt_arg == "Find errors in: hello"


@patch("normalize._call_mlx")
def test_llm_normalize_custom_model(mock_call_mlx):
    """Model parameter passed to MLX backend."""
    mock_call_mlx.return_value = '{"corrections": []}'
    _, _ = llm_normalize("text", model="mlx-community/some-other-model")
    model_arg = mock_call_mlx.call_args[0][1]
    assert model_arg == "mlx-community/some-other-model"


@patch("normalize._call_mlx")
def test_llm_normalize_returns_processing_entry(mock_call_mlx):
    """Processing entry has expected stage, model, status, and timestamp."""
    mock_call_mlx.return_value = '{"corrections": []}'
    _, entry = llm_normalize("some text")
    assert entry["stage"] == "llm_normalization"
    assert entry["model"] == MODEL
    assert entry["status"] == "success"
    assert "timestamp" in entry
