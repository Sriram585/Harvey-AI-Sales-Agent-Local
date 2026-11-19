"""Tests for local AI package (sellbird.local_ai)."""

import json
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from pydantic import BaseModel

from sellbird.local_ai.hardware import (
    HardwareProfile,
    get_hardware_profile,
    _detect_gpu,
)
from sellbird.local_ai.network_guard import (
    ExternalNetworkBlockedError,
    StrictOfflineSandbox,
    STRICT_OFFLINE_MODE,
)
from sellbird.local_ai.embeddings import (
    embed_text,
    cosine_similarity,
    LOCAL_MODEL_PATH,
)
from sellbird.local_ai.ollama_client import (
    OllamaClient,
    OllamaUnavailableError,
    INFERENCE_LOG_FILE,
    _strip_markdown_fences,
)
from sellbird.brain import Brain


class MockOutputModel(BaseModel):
    intent: str
    confidence: float
    summary: str


# ── 1. Hardware Profile Tests ──

def test_hardware_profile_real_system():
    """Verify hardware profile returns valid host specs and classification."""
    profile = get_hardware_profile()
    assert isinstance(profile, HardwareProfile)
    assert profile.tier in ("standard_laptop", "high_performance")
    assert profile.recommended_model in ("qwen3:1.7b", "qwen3:4b", "qwen3:8b", "qwen3:14b")
    assert profile.cpu_cores >= 1
    assert profile.ram_gb > 0
    assert profile.vram_gb >= 0.0
    assert len(profile.fallback_models) >= 1
    assert len(profile.rationale) > 10


def test_hardware_profile_high_performance_tier():
    """Verify tier switch to high_performance when GPU VRAM >= 8GB or RAM >= 32GB."""
    # Test high VRAM condition
    with patch("sellbird.local_ai.hardware._detect_gpu", return_value=("NVIDIA RTX 4090", 24.0)):
        with patch("psutil.virtual_memory", return_value=SimpleNamespace(total=16 * (1024 ** 3), available=8 * (1024 ** 3))):
            profile = get_hardware_profile()
            assert profile.tier == "high_performance"
            assert profile.recommended_model == "qwen3:8b"
            assert "qwen3:14b" in profile.fallback_models

    # Test high RAM condition (e.g. Mac/Server with 64GB)
    with patch("sellbird.local_ai.hardware._detect_gpu", return_value=(None, 0.0)):
        with patch("psutil.virtual_memory", return_value=SimpleNamespace(total=64 * (1024 ** 3), available=40 * (1024 ** 3))):
            profile = get_hardware_profile()
            assert profile.tier == "high_performance"
            assert profile.recommended_model == "qwen3:8b"


def test_hardware_profile_standard_laptop_tier():
    """Verify standard_laptop tier when system has low RAM and no dedicated GPU."""
    # Low RAM (< 16GB)
    with patch("sellbird.local_ai.hardware._detect_gpu", return_value=(None, 0.0)):
        with patch("psutil.virtual_memory", return_value=SimpleNamespace(total=8 * (1024 ** 3), available=4 * (1024 ** 3))):
            profile = get_hardware_profile()
            assert profile.tier == "standard_laptop"
            assert profile.recommended_model == "qwen3:1.7b"
            assert "qwen3:4b" in profile.fallback_models


# ── 2. Network Guard Tests ──

def test_strict_offline_sandbox_blocks_external():
    """Verify StrictOfflineSandbox blocks non-local network attempts."""
    with StrictOfflineSandbox():
        # Block external hostname
        with pytest.raises(ExternalNetworkBlockedError) as exc_info:
            socket.create_connection(("example.com", 80), timeout=1)
        assert "example.com" in str(exc_info.value)

        # Block external IP
        with pytest.raises(ExternalNetworkBlockedError) as exc_info2:
            socket.create_connection(("93.184.216.34", 80), timeout=1)
        assert "93.184.216.34" in str(exc_info2.value)

        # Direct socket connect check
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with pytest.raises(ExternalNetworkBlockedError):
            s.connect(("8.8.8.8", 53))
        s.close()


def test_strict_offline_sandbox_allows_loopback():
    """Verify StrictOfflineSandbox allows 127.0.0.1 and localhost connections."""
    with StrictOfflineSandbox():
        # Connecting to a closed local port raises ConnectionRefusedError or OSError from OS,
        # but NEVER ExternalNetworkBlockedError
        try:
            socket.create_connection(("127.0.0.1", 11434), timeout=0.1)
        except ExternalNetworkBlockedError:
            pytest.fail("StrictOfflineSandbox blocked 127.0.0.1 unexpectedly!")
        except (ConnectionRefusedError, OSError):
            pass

        try:
            socket.create_connection(("localhost", 11434), timeout=0.1)
        except ExternalNetworkBlockedError:
            pytest.fail("StrictOfflineSandbox blocked localhost unexpectedly!")
        except (ConnectionRefusedError, OSError):
            pass


def test_strict_offline_sandbox_restores_socket():
    """Verify socket methods are restored after exiting context."""
    orig_create = socket.create_connection
    orig_connect = socket.socket.connect

    with StrictOfflineSandbox():
        assert socket.create_connection != orig_create

    assert socket.create_connection == orig_create
    assert socket.socket.connect == orig_connect


# ── 3. Embeddings Tests ──

def test_embeddings_computed_inside_offline_sandbox():
    """Verify sentence-transformers runs and computes cosine similarity completely offline."""
    assert LOCAL_MODEL_PATH.exists(), f"Local model must be cached at {LOCAL_MODEL_PATH}"

    with StrictOfflineSandbox():
        v1 = embed_text("AI autonomous sales agent for outreach")
        v2 = embed_text("Automated outbound email sales development representative")
        v3 = embed_text("A delicious chocolate fudge brownie recipe with walnuts")

        # all-MiniLM-L6-v2 outputs 384 dimensional vectors
        assert len(v1) == 384
        assert len(v2) == 384
        assert len(v3) == 384

        sim_related = cosine_similarity(v1, v2)
        sim_unrelated = cosine_similarity(v1, v3)

        assert 0.0 <= sim_related <= 1.0
        assert 0.0 <= sim_unrelated <= 1.0
        assert sim_related > sim_unrelated, f"Expected {sim_related} > {sim_unrelated}"


# ── 4. Ollama Client Tests ──

@pytest.mark.asyncio
async def test_ollama_client_json_parsing_and_logging(tmp_path, monkeypatch):
    """Verify JSON generation, markdown stripping, and inference logging."""
    import httpx
    log_file = tmp_path / "local_inference.jsonl"
    monkeypatch.setattr("sellbird.local_ai.ollama_client.INFERENCE_LOG_FILE", log_file)

    client = OllamaClient()

    mock_resp_json = {
        "model": "qwen3:4b",
        "created_at": "2026-11-19T10:00:00Z",
        "message": {
            "role": "assistant",
            "content": '```json\n{"intent": "interested", "confidence": 0.95, "summary": "Wants a demo"}\n```',
        },
        "done": True,
        "eval_count": 42,
        "eval_duration": 1000000000,
    }

    mock_response = httpx.Response(
        200,
        json=mock_resp_json,
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/chat"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        result = await client.generate_json(
            prompt="Analyze this reply: 'Sounds great, send me a demo.'",
            system_prompt="Classify buyer intent.",
            response_model=MockOutputModel,
            workflow_name="handler.classify",
        )

        assert isinstance(result, MockOutputModel)
        assert result.intent == "interested"
        assert result.confidence == 0.95
        assert result.summary == "Wants a demo"

    # Verify log entry in jsonl
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    log_entry = json.loads(lines[0])
    assert log_entry["workflow_name"] == "handler.classify"
    assert log_entry["model_name"] == "qwen3:4b"
    assert log_entry["eval_count"] == 42
    assert log_entry["tokens_per_sec"] > 0
    assert log_entry["json_repaired"] is False
    assert log_entry["status"] == "success"


@pytest.mark.asyncio
async def test_ollama_client_automatic_repair_retry(tmp_path, monkeypatch):
    """Verify automatic 1x repair retry when model outputs malformed JSON on attempt 1."""
    import httpx
    log_file = tmp_path / "local_inference.jsonl"
    monkeypatch.setattr("sellbird.local_ai.ollama_client.INFERENCE_LOG_FILE", log_file)

    client = OllamaClient()

    # Attempt 1 returns broken JSON (trailing comma / unclosed brace)
    resp1 = httpx.Response(
        200,
        json={
            "model": "qwen3:4b",
            "message": {"role": "assistant", "content": '{"intent": "interested", "confidence": 0.9,'},
            "eval_count": 20,
            "eval_duration": 500000000,
        },
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/chat"),
    )

    # Attempt 2 (repair retry) returns valid JSON
    resp2 = httpx.Response(
        200,
        json={
            "model": "qwen3:4b",
            "message": {
                "role": "assistant",
                "content": '{"intent": "interested", "confidence": 0.9, "summary": "Repaired successfully"}',
            },
            "eval_count": 25,
            "eval_duration": 500000000,
        },
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/chat"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = [resp1, resp2]

        result = await client.generate_json(
            prompt="Analyze reply",
            system_prompt="Classify buyer intent.",
            response_model=MockOutputModel,
            workflow_name="handler.repair_test",
        )

        assert isinstance(result, MockOutputModel)
        assert result.summary == "Repaired successfully"
        assert mock_post.call_count == 2

    # Verify log entry shows repaired
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    log_entry = json.loads(lines[0])
    assert log_entry["json_repaired"] is True
    assert log_entry["status"] == "success"
    assert log_entry["eval_count"] == 45


@pytest.mark.asyncio
async def test_ollama_client_unavailable_error():
    """Verify OllamaUnavailableError is raised cleanly when Ollama is unreachable."""
    client = OllamaClient(base_url="http://127.0.0.1:19999", timeout=1.0)

    with pytest.raises(OllamaUnavailableError) as exc_health:
        await client.health_check()
    assert "Ollama is unreachable" in str(exc_health.value)

    with pytest.raises(OllamaUnavailableError) as exc_gen:
        await client.generate_json(
            prompt="Test",
            system_prompt="Test",
            response_model=MockOutputModel,
            workflow_name="test.unavailable",
        )
    assert "Failed to connect to Ollama" in str(exc_gen.value)


# ── 5. Brain Wiring Tests ──

@pytest.mark.asyncio
async def test_brain_wired_to_ollama_client():
    """Verify Brain initializes OllamaClient and uses local engine."""
    brain = Brain()
    assert brain.provider == "local"
    assert isinstance(brain.ollama, OllamaClient)
    assert brain.model in ("qwen3:1.7b", "qwen3:4b", "qwen3:8b", "qwen3:14b", "local-llm")

    # Mock health check
    with patch.object(brain.ollama, "health_check", new_callable=AsyncMock) as mock_health:
        mock_health.return_value = ["qwen3:4b", "llama3.2:3b"]
        problem = await brain.preflight()
        assert problem == ""

    # Mock health check failure
    with patch.object(brain.ollama, "health_check", new_callable=AsyncMock) as mock_fail:
        mock_fail.side_effect = OllamaUnavailableError("Ollama is not running")
        problem = await brain.preflight()
        assert "Ollama is not running" in problem
