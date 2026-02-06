"""Tests for normalize module."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from normalize import _parse_response, normalize, DEFAULT_PROMPT


# --- _parse_response tests ---


def test_parse_corrections_from_json():
    """Standard JSON with corrections list is parsed."""
    response = '{"corrections": [{"transcribed": "fondos", "correct": "Pandavas"}]}'
    result = _parse_response(response)
    assert len(result) == 1
    assert result[0]["transcribed"] == "fondos"
    assert result[0]["correct"] == "Pandavas"


def test_parse_empty_corrections():
    """Empty corrections list returns []."""
    response = '{"corrections": []}'
    result = _parse_response(response)
    assert result == []


def test_parse_empty_object():
    """Empty object (no corrections key) returns []."""
    response = '{}'
    result = _parse_response(response)
    assert result == []


def test_parse_json_in_thinking_output():
    """JSON buried in thinking tags is extracted."""
    response = (
        '<think>\nLet me analyze the text for mishearings...\n</think>\n'
        '{"corrections": [{"transcribed": "goros", "correct": "Kauravas"}]}'
    )
    result = _parse_response(response)
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
    result = _parse_response(response)
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
    result = _parse_response(response)
    assert len(result) == 1
    assert result[0]["correct"] == "Dhritarashtra"


def test_parse_garbage_raises():
    """Unparseable response raises ValueError."""
    import pytest
    with pytest.raises(ValueError, match="Could not parse LLM response"):
        _parse_response("I don't understand the question, sorry!")


# --- normalize tests ---


@patch("normalize.subprocess.run")
def test_normalize_returns_corrections(mock_run):
    """Successful end-to-end with mocked subprocess."""
    mock_run.return_value = MagicMock(
        stdout='{"corrections": [{"transcribed": "fondos", "correct": "Pandavas"}]}',
        returncode=0,
    )
    result = normalize("The fondos went to war")
    assert len(result) == 1
    assert result[0]["transcribed"] == "fondos"


@patch("normalize.subprocess.run")
def test_normalize_empty_corrections(mock_run):
    """Model returns no corrections."""
    mock_run.return_value = MagicMock(
        stdout='{"corrections": []}',
        returncode=0,
    )
    result = normalize("Once upon a time")
    assert result == []


@patch("normalize.subprocess.run")
def test_normalize_timeout_raises(mock_run):
    """subprocess.TimeoutExpired propagates."""
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="ollama", timeout=300)
    import pytest
    with pytest.raises(subprocess.TimeoutExpired):
        normalize("some text")


@patch("normalize.subprocess.run")
def test_normalize_default_prompt(mock_run):
    """DEFAULT_PROMPT is used when none provided."""
    mock_run.return_value = MagicMock(
        stdout='{"corrections": []}',
        returncode=0,
    )
    normalize("test text")
    call_args = mock_run.call_args[0][0]
    # The prompt arg (index 3) should contain DEFAULT_PROMPT content
    assert "Mahabharata" in call_args[3]
    assert "test text" in call_args[3]


@patch("normalize.subprocess.run")
def test_normalize_custom_prompt(mock_run):
    """Custom prompt passed through to subprocess."""
    mock_run.return_value = MagicMock(
        stdout='{"corrections": []}',
        returncode=0,
    )
    custom = "Find errors in: {text}"
    normalize("hello", prompt=custom)
    call_args = mock_run.call_args[0][0]
    assert call_args[3] == "Find errors in: hello"


@patch("normalize.subprocess.run")
def test_normalize_custom_model(mock_run):
    """Model parameter passed to ollama command."""
    mock_run.return_value = MagicMock(
        stdout='{"corrections": []}',
        returncode=0,
    )
    normalize("text", model="llama3:8b")
    call_args = mock_run.call_args[0][0]
    assert call_args[1] == "run"
    assert call_args[2] == "llama3:8b"
