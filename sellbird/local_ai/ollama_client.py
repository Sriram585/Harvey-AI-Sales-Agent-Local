"""Async HTTP client for local Ollama API."""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar
import httpx
from pydantic import BaseModel, ValidationError

logger = logging.getLogger("sellbird.local_ai.ollama")

T = TypeVar("T", bound=BaseModel)

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_KEEP_ALIVE = "30m"
DEFAULT_RECOMMENDED_MODEL = "qwen2.5:1.5b"
INFERENCE_LOG_FILE = Path("logs") / "local_inference.jsonl"


class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama daemon cannot be reached."""
    pass


def _strip_markdown_fences(content: str) -> str:
    """Clean markdown code block wrappers from JSON responses."""
    text = content.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        else:
            text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _clean_and_unwrap_json(text: str, response_model: type[T]) -> T:
    """Validate JSON text against response_model, automatically unwrapping common LLM wrapper structures."""
    clean_text = _strip_markdown_fences(text)

    # 1. Direct validation attempt
    try:
        return response_model.model_validate_json(clean_text)
    except Exception:
        pass

    # 2. Parse as python object and handle container wrappers
    try:
        data = json.loads(clean_text)

        # Handle list when container expects list inside a key
        if isinstance(data, list):
            for field_name in ("signals", "items", "data", "results"):
                if hasattr(response_model, "model_fields") and field_name in response_model.model_fields:
                    try:
                        return response_model.model_validate({field_name: data})
                    except Exception:
                        pass

        if isinstance(data, dict):
            # Check if fields are wrapped in 'properties', 'data', 'result', etc.
            wrapper_keys = (
                "properties",
                "data",
                "response",
                "result",
                "output",
                "profile",
                "company_profile",
                "icp",
                "signals",
                "research",
            )
            for key in wrapper_keys:
                if key in data and isinstance(data[key], (dict, list)):
                    try:
                        return response_model.model_validate(data[key])
                    except Exception:
                        pass
                    # In case data[key] is a list and model expects {key: [...]}
                    try:
                        return response_model.model_validate({key: data[key]})
                    except Exception:
                        pass

            # Direct dict validation
            try:
                return response_model.model_validate(data)
            except Exception:
                pass
    except Exception:
        pass

    # 3. Final attempt to raise informative Pydantic ValidationError
    return response_model.model_validate_json(clean_text)


class OllamaClient:
    """Client for native Ollama API endpoints strictly on localhost."""

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        keep_alive: str = DEFAULT_KEEP_ALIVE,
        timeout: float = 60.0,
        db_log_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.keep_alive = keep_alive
        self.timeout = timeout
        self.db_log_callback = db_log_callback

    async def _log_inference(self, record: dict[str, Any]):
        """Persist structured inference record to jsonl file and optional DB callback."""
        try:
            INFERENCE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(INFERENCE_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.warning(f"Failed to append to {INFERENCE_LOG_FILE}: {e}")

        if self.db_log_callback is not None:
            try:
                await self.db_log_callback(record)
            except Exception as e:
                logger.warning(f"Error executing db_log_callback: {e}")

    async def health_check(self) -> list[str]:
        """Query GET /api/tags to verify Ollama is running and list available models."""
        url = f"{self.base_url}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                if resp.status_code != 200:
                    raise OllamaUnavailableError(
                        f"Ollama returned HTTP {resp.status_code} at {url}"
                    )
                data = resp.json()
                models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
                return models
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError) as e:
            raise OllamaUnavailableError(
                f"Ollama is unreachable at {self.base_url}. Ensure 'ollama serve' is running. Error: {e}"
            ) from e

    async def generate_json(
        self,
        prompt: str,
        system_prompt: str,
        response_model: type[T],
        workflow_name: str,
        model: str | None = None,
    ) -> T:
        """Call POST /api/chat with format='json', stream=False, validate response, and retry once on malformed JSON."""
        endpoint = f"{self.base_url}/api/chat"
        chosen_model = model or "qwen3:4b"

        # Inject schema guidance into system prompt if not present
        schema_json = json.dumps(response_model.model_json_schema())
        full_system = (
            f"{system_prompt}\n\n"
            f"You must output valid JSON adhering strictly to this JSON schema:\n{schema_json}"
        )

        messages = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": prompt},
        ]

        payload = {
            "model": chosen_model,
            "messages": messages,
            "format": "json",
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "keep_alive": self.keep_alive,
            },
        }

        t0 = time.perf_counter()
        raw_text = ""
        eval_count = 0
        eval_duration = 0
        status = "pending"
        json_repaired = False

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                try:
                    resp = await client.post(endpoint, json=payload)
                except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError) as e:
                    raise OllamaUnavailableError(
                        f"Failed to connect to Ollama at {endpoint}: {e}"
                    ) from e

                if resp.status_code != 200:
                    status = f"http_{resp.status_code}"
                    raise OllamaUnavailableError(
                        f"Ollama returned HTTP {resp.status_code}: {resp.text}"
                    )

                data = resp.json()
                raw_text = data.get("message", {}).get("content", "")
                eval_count = data.get("eval_count", 0)
                eval_duration = data.get("eval_duration", 0)

                # Attempt 1 validation
                clean_text = _strip_markdown_fences(raw_text)
                try:
                    parsed = _clean_and_unwrap_json(clean_text, response_model)
                    status = "success"
                    return parsed
                except (ValidationError, json.JSONDecodeError) as parse_err:
                    logger.warning(
                        f"Ollama JSON validation failed on first attempt for {workflow_name}: {parse_err}. Retrying JSON repair..."
                    )

                    # Automatic 1x repair retry
                    repair_messages = [
                        {"role": "system", "content": full_system},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": raw_text},
                        {
                            "role": "user",
                            "content": (
                                f"The previous response failed schema validation: {parse_err}.\n"
                                "Please fix the JSON formatting error and respond with ONLY the corrected flat JSON object."
                            ),
                        },
                    ]
                    repair_payload = {
                        "model": chosen_model,
                        "messages": repair_messages,
                        "format": "json",
                        "stream": False,
                        "keep_alive": self.keep_alive,
                        "options": {
                            "keep_alive": self.keep_alive,
                        },
                    }

                    repair_resp = await client.post(endpoint, json=repair_payload)
                    if repair_resp.status_code == 200:
                        repair_data = repair_resp.json()
                        raw_repair_text = repair_data.get("message", {}).get("content", "")
                        clean_repair = _strip_markdown_fences(raw_repair_text)
                        parsed = _clean_and_unwrap_json(clean_repair, response_model)
                        json_repaired = True
                        status = "success"
                        eval_count += repair_data.get("eval_count", 0)
                        return parsed
                    else:
                        status = f"repair_http_{repair_resp.status_code}"
                        raise parse_err

        except Exception as exc:
            if status == "pending":
                status = "error"
            raise exc
        finally:
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            if eval_duration > 0:
                tokens_per_sec = round(eval_count / (eval_duration / 1e9), 2)
            elif latency_ms > 0 and eval_count > 0:
                tokens_per_sec = round(eval_count / (latency_ms / 1000), 2)
            else:
                tokens_per_sec = 0.0

            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "workflow_name": workflow_name,
                "endpoint_url": endpoint,
                "model_name": chosen_model,
                "latency_ms": latency_ms,
                "eval_count": eval_count,
                "tokens_per_sec": tokens_per_sec,
                "json_repaired": json_repaired,
                "status": status,
            }
            await self._log_inference(record)
