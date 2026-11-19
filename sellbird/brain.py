"""Brain — Local Sales Agent thinking engine.

All external AI APIs and external CLIs have been stripped.
Local execution routes through an optional local transport or local model engine.
Zero external AI calls are permitted.
"""

import asyncio
import json
import logging
import re
from pathlib import Path

from sellbird.paths import PROJECT_ROOT
from sellbird.state import StateManager
from sellbird.textio import read_user_text
from sellbird.workspace_home import current_home

logger = logging.getLogger("sellbird.brain")

PROMPTS_DIR = PROJECT_ROOT / "prompts"
SKILLS_DIR = PROJECT_ROOT / "skills"

DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_MAX_RETRIES = 2

PROVIDER_LABELS = {"local": "Local"}
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
        self.model = "local-llm"
        self.last_reply_empty = False
        self.unavailable = ""
        self.quota = None

    @property
    def provider_label(self) -> str:
        return PROVIDER_LABELS.get(self.provider, "Local")

    @property
    def paused_reason(self) -> str:
        transport_paused = str(getattr(self.transport, "paused_reason", "") or "") \
            if self.transport is not None else ""
        if transport_paused:
            return transport_paused
        return self.unavailable

    async def preflight(self) -> str:
        """Preflight verification. Returns empty string if ready."""
        return ""

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
        """Execute a thought turn. External AI is disabled; uses transport if configured."""
        self.last_reply_empty = False

        if self.transport is not None:
            return await self._think_transport(
                prompt,
                expect_json=expect_json,
                timeout=timeout,
                agent=agent,
                task=task,
            )

        logger.info(f"External AI disabled. Local brain invoked for {agent}/{task} without active local transport.")
        self.last_reply_empty = True
        return ""

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

        text = raw.strip()
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
