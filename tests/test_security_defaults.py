"""Approval requirements apply; LinkedIn needs opt-in."""

from types import SimpleNamespace

import pytest
import yaml

from sellbird.config import BrainConfig, LinkedInChannelConfig
from sellbird.paths import PROJECT_ROOT
from sellbird.trainer import Trainer


def test_linkedin_is_disabled_in_model_and_shipped_template():
    assert LinkedInChannelConfig().enabled is False
    template = yaml.safe_load((PROJECT_ROOT / "sellbird.yaml").read_text())
    assert template["channels"]["linkedin"]["enabled"] is False
    assert LinkedInChannelConfig(enabled=True).enabled is True


def test_product_training_does_not_enable_linkedin():
    trainer = Trainer.__new__(Trainer)
    trainer.brain_config = BrainConfig()
    config = trainer._build_config({}, {}, {}, "acme.test")
    assert config["channels"]["linkedin"]["enabled"] is False


@pytest.mark.parametrize("enabled", [False, True])
def test_setup_saves_explicit_linkedin_choice_for_existing_config(tmp_path, monkeypatch, enabled):
    from sellbird import setup
    path = tmp_path / "sellbird.yaml"
    path.write_text(yaml.safe_dump({"channels": {"linkedin": {"enabled": not enabled}}}))
    monkeypatch.setattr(setup, "CONFIG_FILE", path)
    wizard = setup.SetupWizard()
    wizard.env_vars["LINKEDIN_EMAIL"] = "user@example.test" if enabled else ""
    wizard.config = {"channels": {"linkedin": {"enabled": enabled}}}
    wizard._write_config()
    assert yaml.safe_load(path.read_text())["channels"]["linkedin"]["enabled"] is enabled


# ── a held Instantly setup stays visible and never stalls the pipeline ──

@pytest.mark.asyncio
async def test_training_on_a_local_url_says_why_it_was_refused():
    from sellbird.trainer import FallbackCrawler
    crawler = FallbackCrawler()
    assert await crawler.crawl("http://localhost:3000", max_pages=3) == {}
    assert crawler.blocked is True


def test_status_still_works_when_the_config_is_invalid(tmp_path, monkeypatch, capsys):
    # `sellbird status` is what people run when something's wrong, so an
    # invalid sellbird.yaml must not crash the sending-hold check.
    import sellbird.config
    import sellbird.state
    from sellbird.cli import cmd_status
    from sellbird.config import SellbirdConfig
    from pydantic import ValidationError

    def invalid():
        SellbirdConfig(channels={"email": {"max_daily_sends": "lots"}})

    try:
        invalid()
    except ValidationError:
        pass
    real_state = sellbird.state.StateManager
    monkeypatch.setattr(sellbird.state, "StateManager", lambda: real_state(str(tmp_path / "s.db")))
    monkeypatch.setattr(sellbird.config, "load_config", invalid)
    cmd_status(SimpleNamespace(check=False))
    assert "Prospects:" in capsys.readouterr().out
