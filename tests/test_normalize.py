"""Tests for normalize module."""

import json
import sys
import urllib.error
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from normalize import _parse_llm_corrections, llm_normalize, DEFAULT_PROMPT, MODEL


def _make_ollama_response(text: str):
    """Create a mock urlopen context manager returning an Ollama JSON response."""
    body = json.dumps({"response": text}).encode()
    resp = BytesIO(body)
    resp.__enter__ = lambda self: self
    resp.__exit__ = lambda self, *a: None
    return resp


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


def test_parse_garbage_raises():
    """Unparseable response raises ValueError."""
    import pytest
    with pytest.raises(ValueError, match="Could not parse LLM response"):
        _parse_llm_corrections("I don't understand the question, sorry!")


# --- llm_normalize tests ---


@patch("normalize.urllib.request.urlopen")
def test_llm_normalize_returns_corrections(mock_urlopen):
    """Successful end-to-end with mocked Ollama API."""
    mock_urlopen.return_value = _make_ollama_response(
        '{"corrections": [{"transcribed": "fondos", "correct": "Pandavas"}]}'
    )
    corrections, entry = llm_normalize("The fondos went to war")
    assert len(corrections) == 1
    assert corrections[0]["transcribed"] == "fondos"


@patch("normalize.urllib.request.urlopen")
def test_llm_normalize_empty_corrections(mock_urlopen):
    """Model returns no corrections."""
    mock_urlopen.return_value = _make_ollama_response('{"corrections": []}')
    corrections, entry = llm_normalize("Once upon a time")
    assert corrections == []


@patch("normalize.urllib.request.urlopen")
def test_llm_normalize_timeout_raises(mock_urlopen):
    """TimeoutError propagates when Ollama request times out."""
    mock_urlopen.side_effect = urllib.error.URLError(TimeoutError("timed out"))
    import pytest
    with pytest.raises(TimeoutError):
        llm_normalize("some text")


@patch("normalize.urllib.request.urlopen")
def test_llm_normalize_default_prompt(mock_urlopen):
    """DEFAULT_PROMPT is used when none provided."""
    mock_urlopen.return_value = _make_ollama_response('{"corrections": []}')
    _, _ = llm_normalize("test text")
    req = mock_urlopen.call_args[0][0]
    payload = json.loads(req.data)
    assert "Mahabharata" in payload["prompt"]
    assert "test text" in payload["prompt"]


@patch("normalize.urllib.request.urlopen")
def test_llm_normalize_custom_prompt(mock_urlopen):
    """Custom prompt passed through to Ollama API."""
    mock_urlopen.return_value = _make_ollama_response('{"corrections": []}')
    custom = "Find errors in: {text}"
    _, _ = llm_normalize("hello", prompt=custom)
    req = mock_urlopen.call_args[0][0]
    payload = json.loads(req.data)
    assert payload["prompt"] == "Find errors in: hello"


@patch("normalize.urllib.request.urlopen")
def test_llm_normalize_custom_model(mock_urlopen):
    """Model parameter passed to Ollama API."""
    mock_urlopen.return_value = _make_ollama_response('{"corrections": []}')
    _, _ = llm_normalize("text", model="llama3:8b")
    req = mock_urlopen.call_args[0][0]
    payload = json.loads(req.data)
    assert payload["model"] == "llama3:8b"


@patch("normalize.urllib.request.urlopen")
def test_llm_normalize_returns_processing_entry(mock_urlopen):
    """Processing entry has expected stage, model, status, and timestamp."""
    mock_urlopen.return_value = _make_ollama_response('{"corrections": []}')
    _, entry = llm_normalize("some text")
    assert entry["stage"] == "llm_normalization"
    assert entry["model"] == MODEL
    assert entry["status"] == "success"
    assert "timestamp" in entry
