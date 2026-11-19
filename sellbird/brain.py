"""Brain — Local Sales Agent thinking engine.

All external AI APIs and external CLIs have been stripped.
Local execution routes strictly through local Ollama or an optional local transport.
Zero external AI calls are permitted.
"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sellbird.paths import PROJECT_ROOT
from sellbird.state import StateManager
from sellbird.textio import read_user_text
from sellbird.workspace_home import current_home
from sellbird.local_ai.ollama_client import (
    DEFAULT_KEEP_ALIVE,
    DEFAULT_OLLAMA_URL,
    OllamaClient,
    OllamaUnavailableError,
    _strip_markdown_fences,
)
from sellbird.local_ai.hardware import get_hardware_profile

logger = logging.getLogger("sellbird.brain")

PROMPTS_DIR = PROJECT_ROOT / "prompts"
SKILLS_DIR = PROJECT_ROOT / "skills"

DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_MAX_RETRIES = 2

PROVIDER_LABELS = {"local": "Local Ollama"}
BRAIN_STATUS_KEY = "brain_status"


def parse_brain_status(raw: str) -> dict:
    """The stored BRAIN_STATUS_KEY value as a dict ({} if absent or bad)."""
    try:
        value = json.loads(raw) if raw else {}
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def active_brain_status(raw: str, provider: str) -> dict:
    status = parse_brain_status(raw)
    return status if status.get("provider") == provider else {}


class BrainUnavailable(Exception):
    """The provider can't be used until the operator acts."""


def _read_knowledge(folder: str, name: str) -> str:
    """Read prompt or skill markdown file."""
    own = getattr(current_home(), f"{folder}_dir") / f"{name}.md"
    try:
        return read_user_text(own)
    except FileNotFoundError:
        from sellbird.home import bundled_defaults
        shipped = bundled_defaults()
        if shipped is None or (shipped / folder / f"{name}.md") == own:
            raise
        return read_user_text(shipped / folder / f"{name}.md")


class Brain:
    def __init__(self, state: StateManager = None, brain_config=None, transport=None):
        self.state = state
        self.transport = transport
        self.config = brain_config
        self.provider = "local"
        
        # Resolve model name: config override > hardware profile recommendation > default
        if brain_config and getattr(brain_config, "model", None) and brain_config.model not in ("local", "local-llm", ""):
            self.model = brain_config.model
        else:
            try:
                profile = get_hardware_profile()
                self.model = profile.recommended_model
            except Exception:
                self.model = "qwen3:4b"

        self.ollama = OllamaClient()
        self.last_reply_empty = False
        self.unavailable = ""
        self.quota = None

    @property
    def provider_label(self) -> str:
        return PROVIDER_LABELS.get(self.provider, "Local Ollama")

    @property
    def paused_reason(self) -> str:
        transport_paused = str(getattr(self.transport, "paused_reason", "") or "") \
            if self.transport is not None else ""
        if transport_paused:
            return transport_paused
        return self.unavailable

    async def preflight(self) -> str:
        """Preflight verification. Returns empty string if ready, or error description."""
        try:
            models = await self.ollama.health_check()
            if not models:
                return f"Ollama is running at {self.ollama.base_url}, but no models are loaded. Run 'ollama pull {self.model}'."
            return ""
        except OllamaUnavailableError as e:
            return str(e)
        except Exception as e:
            return f"Ollama preflight failed: {e}"

    async def think(
        self,
        prompt: str,
        *,
        expect_json: bool = False,
        agent: str = "agent",
        task: str = "task",
        label: str = "",
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> str:
        """Execute an inference call against local Ollama or optional transport."""
        self.last_reply_empty = False

        if self.transport is not None:
            return await self._think_transport(
                prompt,
                expect_json=expect_json,
                timeout=timeout,
                agent=agent,
                task=task,
            )

        # Native Ollama call
        import httpx
        endpoint = f"{self.ollama.base_url}/api/chat"
        system_prompt = (
            f"You are Harvey/Sellbird, an autonomous local AI sales and outreach agent. "
            f"Executing as agent '{agent}' on task '{task}'."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.ollama.keep_alive,
            "options": {
                "keep_alive": self.ollama.keep_alive,
            },
        }
        if expect_json:
            payload["format"] = "json"

        t0 = time.perf_counter()
        raw_text = ""
        eval_count = 0
        eval_duration = 0
        status = "pending"

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(endpoint, json=payload)
                if resp.status_code != 200:
                    status = f"http_{resp.status_code}"
                    logger.warning(f"Ollama returned HTTP {resp.status_code}: {resp.text}")
                    self.last_reply_empty = True
                    return ""

                data = resp.json()
                raw_text = data.get("message", {}).get("content", "")
                eval_count = data.get("eval_count", 0)
                eval_duration = data.get("eval_duration", 0)
                status = "success"

                if expect_json:
                    return _strip_markdown_fences(raw_text)
                return raw_text.strip()

        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError) as e:
            status = "connection_error"
            self.unavailable = f"Local Ollama unreachable: {e}"
            logger.warning(f"Brain: {self.unavailable}")
            self.last_reply_empty = True
            return ""
        except Exception as e:
            status = "error"
            logger.error(f"Brain think unexpected error ({agent}/{task}): {e}")
            self.last_reply_empty = True
            return ""
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
                "workflow_name": f"{agent}.{task}",
                "endpoint_url": endpoint,
                "model_name": self.model,
                "latency_ms": latency_ms,
                "eval_count": eval_count,
                "tokens_per_sec": tokens_per_sec,
                "json_repaired": False,
                "status": status,
            }
            await self.ollama._log_inference(record)

    async def _think_transport(
        self, prompt: str, *, expect_json: bool, timeout: float, agent: str, task: str
    ) -> str:
        """Run through configured local transport."""
        if self.paused_reason:
            return ""
        try:
            text = await asyncio.wait_for(
                self.transport(
                    prompt,
                    agent=agent,
                    task=task,
                    timeout=timeout,
                    expect_json=expect_json,
                ),
                timeout=timeout,
            )
        except Exception as e:
            logger.warning(f"Brain transport call failed ({agent}/{task}): {type(e).__name__}: {e}")
            return ""
        return str(text or "")

    async def think_json(
        self,
        prompt: str,
        *,
        agent: str = "agent",
        task: str = "task",
        label: str = "",
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> dict:
        """Think and parse JSON response."""
        raw = await self.think(
            prompt,
            expect_json=True,
            agent=agent,
            task=task,
            label=label,
            timeout=timeout,
        )
        if not raw or not raw.strip():
            self.last_reply_empty = True
            return {}

        text = _strip_markdown_fences(raw)
        if "```" in text:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
            if match:
                text = match.group(1).strip()

        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
            logger.warning(f"Brain {agent}/{task}: Expected dict, got {type(data).__name__}")
            return {}
        except json.JSONDecodeError as e:
            logger.warning(f"Brain {agent}/{task}: Invalid JSON: {e}")
            return {}

    async def check_usage(self) -> int:
        return 0

    async def can_call(self, max_daily_calls: int = 1000) -> bool:
        return True

    def load_prompt(self, prompt_name: str, **kwargs) -> str:
        """Load a prompt template from prompts/ directory."""
        try:
            template = _read_knowledge("prompts", prompt_name)
        except FileNotFoundError:
            logger.warning(f"Prompt file not found: {prompt_name}.md")
            return ""
        except OSError as e:
            logger.error(f"Could not read prompt file {prompt_name}.md: {e}")
            return ""
        for key, value in kwargs.items():
            template = template.replace(f"{{{{{key}}}}}", str(value))
        leftover = re.findall(r"\{\{(\w+)\}\}", template)
        if leftover:
            logger.warning(
                f"Prompt '{prompt_name}' has unfilled variables: {sorted(set(leftover))}"
            )
        return template

    def load_skill(self, skill_name: str) -> str:
        """Load a skill knowledge file from skills/ directory."""
        try:
            return _read_knowledge("skills", skill_name)
        except FileNotFoundError:
            logger.warning(f"Skill file not found: {skill_name}.md")
            return ""
        except OSError as e:
            logger.error(f"Could not read skill file {skill_name}.md: {e}")
            return ""

    def load_skills_for_agent(self, agent_name: str) -> str:
        """Load foundational knowledge for a given agent."""
        skill_map = {
            "scout": ["prospecting_tactics", "lead_qualification", "account_navigation", "signal_playbook", "product_knowledge"],
            "writer": ["product_knowledge", "competitive_intel"],
            "handler": ["objection_handling", "sales_methodology", "offer_strategy", "product_knowledge", "competitive_intel"],
            "sender": ["email_frameworks", "product_knowledge"],
            "linkedin": ["linkedin_outreach", "prospecting_tactics", "product_knowledge"],
        }
        skill_names = skill_map.get(agent_name, [])
        if not skill_names:
            return ""

        sections = []
        for name in skill_names:
            content = self.load_skill(name)
            if content:
                sections.append(content)

        if not sections:
            return ""

        return (
            "\n\n---\n## FOUNDATIONAL KNOWLEDGE\n"
            "Use the following frameworks and best practices to guide your work:\n\n"
            + "\n\n---\n\n".join(sections)
        )
