"""Disposable Sellbird test harness for end-to-end tests."""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

from sellbird.config import EnvConfig, SellbirdConfig
from sellbird import main as main_mod
from sellbird.state import StateManager
from sellbird.time_utils import now_iso as _now_iso
from tests.test_outbox_native import FakeProvider, seed_prospect

BASE = {
    "persona": {"name": "Sellbird", "company": "Acme Tools", "role": "BD",
                "email": "sellbird@acme.example", "linkedin": "", "tone": "direct"},
    "product": {"name": "Widget", "description": "Widgets for teams", "pricing": "$10",
                "key_benefits": ["fast"], "objection_responses": {}},
    "icp": {"industries": ["software"], "company_size": "10-50", "titles": ["VP"],
            "geography": ["US"]},
    "channels": {"email": {"provider": "smtp", "require_approval": True}},
}


async def _approved_item(state, email="bob@beta.com"):
    pid = await seed_prospect(state, email=email, status="contacted")
    return await state.add_outbox_item(
        prospect_id=pid, to_email=email, subject="following up",
        body="Hi Bob, any thoughts on this?", send_at=_now_iso(), status="approved",
        campaign_id="c-1", step=2,
    )


class Harness:
    """One disposable Sellbird install wired to fixtures."""

    def __init__(self, monkeypatch, tmp_path, provider="local", model="local", db=None):
        self.tmp = tmp_path
        self.monkeypatch = monkeypatch
        self.log = tmp_path / "brain.log"
        self.scenario = "smart"
        self.state = StateManager(str(db or tmp_path / "sellbird.db"))
        self.mail = FakeProvider()
        self.cycles = 0
        self.stop_after = 1
        self.between = {}
        self.scout_runs = 0
        self.set_config(provider, model)

        runtimes = tmp_path / "runtimes"
        runtimes.mkdir(exist_ok=True)
        monkeypatch.setattr(tempfile, "tempdir", str(runtimes))

        import sellbird.agents.handler as handler_mod
        import sellbird.agents.scout as scout_mod
        import sellbird.agents.sender as sender_mod

        monkeypatch.setattr(handler_mod, "get_mail_provider", lambda c, e: self.mail)
        monkeypatch.setattr(sender_mod, "get_mail_provider", lambda c, e: self.mail)

        async def scout_run(_self):
            self.scout_runs += 1

        async def no_profile(state, limit=25, **_):
            return None

        monkeypatch.setattr(scout_mod.Scout, "run", scout_run)
        monkeypatch.setattr(main_mod, "run_profile_stage", no_profile)
        monkeypatch.setattr(main_mod, "load_config", lambda: self.config)
        monkeypatch.setattr(main_mod, "load_env", lambda: EnvConfig())
        monkeypatch.setattr(main_mod, "StateManager", lambda: self.state)
        monkeypatch.setattr(main_mod, "in_quiet_hours", lambda config: False)

        async def one_cycle_sleep(seconds, stop_event):
            self.cycles += 1
            step = self.between.get(self.cycles)
            if step is not None:
                result = step()
                if asyncio.iscoroutine(result):
                    await result
            return self.cycles >= self.stop_after

        monkeypatch.setattr(main_mod, "_interruptible_sleep", one_cycle_sleep)

        real_init = sender_mod.Sender.__init__

        def init_no_pacing(s, *a, **k):
            real_init(s, *a, **k)
            s.pacing_seconds = lambda: 0

        monkeypatch.setattr(sender_mod.Sender, "__init__", init_no_pacing)

    def set_config(self, provider="local", model="local"):
        brain = {"provider": provider}
        if model:
            brain["model"] = model
        self.config = SellbirdConfig(**BASE, brain=brain)

    async def run_cycles(self, n=1, between=None):
        self.cycles = 0
        self.stop_after = n
        self.between = between or {}
        await asyncio.wait_for(main_mod.heartbeat(asyncio.Event()), 120)
