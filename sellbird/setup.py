"""Interactive setup wizard — walks you through everything Sellbird needs."""

import asyncio
import getpass
import logging
import os
import shutil
from pathlib import Path

import yaml
from pydantic import ValidationError

from sellbird.brain import Brain
from sellbird.config import COUNTRY_ALIASES, BrainConfig
from sellbird.state import StateManager

logger = logging.getLogger("sellbird.setup")

from sellbird.paths import PROJECT_ROOT
from sellbird.textio import read_user_text
ENV_FILE = PROJECT_ROOT / ".env"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"
CONFIG_FILE = PROJECT_ROOT / "sellbird.yaml"


def _print_sellbird(message: str):
    """Print a message as Sellbird."""
    print(f"\n  Sellbird: {message}")


def _print_step(step: int, total: int, title: str):
    """Print a step header."""
    print(f"\n{'─'*60}")
    print(f"  Step {step}/{total}: {title}")
    print(f"{'─'*60}")


def _ask(prompt: str, default: str = "", required: bool = True, secret: bool = False) -> str:
    """Ask the user a question. secret=True hides input (passwords, API keys)."""
    suffix = f" [{default}]" if default else ""
    reader = getpass.getpass if secret else input
    while True:
        try:
            answer = reader(f"\n  → {prompt}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Setup cancelled.")
            raise SystemExit(1)

        if not answer and default:
            return default
        if not answer and required:
            print("    (required — please enter a value)")
            continue
        return answer


def _ask_yes_no(prompt: str, default: bool = True) -> bool:
    """Ask a yes/no question."""
    suffix = " [Y/n]" if default else " [y/N]"
    try:
        answer = input(f"\n  → {prompt}{suffix}: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n\n  Setup cancelled.")
        raise SystemExit(1)

    if not answer:
        return default
    return answer in ("y", "yes")


def _read_env_file() -> dict[str, str]:
    """The KEY=value pairs already in .env (comments and blank lines skipped)."""
    existing: dict[str, str] = {}
    if ENV_FILE.exists():
        for line in read_user_text(ENV_FILE).splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                existing[key.strip()] = value.strip()
    return existing


def _config_file() -> Path:
    """The config Sellbird loads: sellbird.local.yaml (what `sellbird train` writes)
    wins over sellbird.yaml when it exists, as in config._find_config_file.
    Setup reads and writes this file, so its answers are the ones Sellbird uses."""
    local = CONFIG_FILE.with_name("sellbird.local.yaml")
    return local if local.exists() else CONFIG_FILE


def _saved_timezone() -> str:
    """usage.quiet_hours.timezone in the file setup writes to, or ""."""
    try:
        data = yaml.safe_load(read_user_text(_config_file()))
    except (OSError, yaml.YAMLError):
        return ""
    usage = data.get("usage") if isinstance(data, dict) else None
    quiet = usage.get("quiet_hours") if isinstance(usage, dict) else None
    zone = quiet.get("timezone") if isinstance(quiet, dict) else None
    return str(zone or "").strip()


def _merge_answers(existing: dict, answers: dict) -> None:
    """Lay manual setup answers over existing config, skipping blank answers so
    they never wipe values already in the file."""
    for key, value in answers.items():
        if isinstance(value, dict) and isinstance(existing.get(key), dict):
            _merge_answers(existing[key], value)
        elif value not in ("", [], {}, None):
            existing[key] = value


def _product_configured() -> bool:
    """Whether the config names a real company and product, not the shipped
    placeholders (the same test the dashboard uses)."""
    try:
        data = yaml.safe_load(read_user_text(_config_file()))
    except (OSError, yaml.YAMLError):
        return False
    if not isinstance(data, dict):
        return False
    persona = data.get("persona") if isinstance(data.get("persona"), dict) else {}
    product = data.get("product") if isinstance(data.get("product"), dict) else {}
    return (persona.get("company") not in (None, "", "Your Company")
            and product.get("name") not in (None, "", "Your Product"))


def _ask_brain_config() -> BrainConfig:
    """Configure local brain."""
    _print_sellbird("I think with a local AI model (no external API calls):")
    model = _ask("Model (leave blank for local default)", default="local", required=False)
    return BrainConfig(provider="local", model=model or "local")


class SetupWizard:
    """Interactive wizard to configure Sellbird from scratch."""

    def __init__(self):
        self.brain_config: BrainConfig = BrainConfig(provider="local", model="local")
        self.env_vars: dict[str, str] = {}
        self.config: dict = {}
        self.manual_product: bool = False
        self.config_saved: bool = False

    async def run(self):
        """Run the full setup wizard."""
        print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   Sellbird Setup                                         ║
║   The open-source outreach platform                      ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")
        _print_sellbird("Welcome! I'm going to get everything set up so I can start")
        _print_sellbird("finding leads and closing deals for you. This takes about 5 minutes.")

        total_steps = 5

        # Step 1: Check prerequisites
        _print_step(1, total_steps, "Checking Prerequisites")
        await self._check_prerequisites()

        # Step 2: Connect email platform
        _print_step(2, total_steps, "Email Platform (Gmail / SMTP / Microsoft)")
        await self._setup_email_platform()

        # Step 3: LinkedIn (optional)
        _print_step(3, total_steps, "LinkedIn (Optional)")
        self._setup_linkedin()

        # Step 4: Train on product
        _print_step(4, total_steps, "Train Sellbird on Your Product")
        await self._setup_product()

        # Step 5: Configure behavior
        _print_step(5, total_steps, "Configure Behavior")
        self._setup_behavior()

        # Write everything
        self._write_env()
        self._write_config()

        # Final summary
        await self._print_summary()

    async def _check_prerequisites(self):
        """Pick the brain provider and verify local brain is ready."""
        self.brain_config = _ask_brain_config()
        print("\n  Checking Local Brain...", end=" ")
        brain = Brain(None, self.brain_config)
        problem = await brain.preflight()
        if problem:
            print(f"✗ Notice: {problem}")
        else:
            print("✓ Ready")

    async def _setup_email_platform(self):
        """Configure email outreach provider."""
        _print_sellbird("Sellbird sends emails via Gmail, Microsoft 365, or any SMTP server.")
        provider = _ask("Email provider (gmail / smtp / microsoft)", default="gmail")
        self.email_provider = provider.lower().strip() if provider else "gmail"
        if self.email_provider == "smtp":
            self.env_vars["SMTP_HOST"] = _ask("SMTP Host", default="smtp.mail.com")
            self.env_vars["SMTP_USERNAME"] = _ask("SMTP Username", default="")
            self.env_vars["SMTP_PASSWORD"] = _ask("SMTP Password", secret=True)

    def _setup_linkedin(self):
        """Configure LinkedIn credentials (optional)."""
        _print_sellbird("I can also do LinkedIn outreach — finding profiles,")
        _print_sellbird("sending connection requests, and messaging connections.")
        _print_sellbird("This requires your LinkedIn login credentials. Totally optional.")

        use_linkedin = _ask_yes_no("Set up LinkedIn outreach?", default=False)

        if use_linkedin:
            email = _ask("LinkedIn email")
            password = _ask("LinkedIn password (input hidden)", secret=True)
            self.env_vars["LINKEDIN_EMAIL"] = email
            self.env_vars["LINKEDIN_PASSWORD"] = password
            _print_sellbird("LinkedIn credentials saved. I'll use these to log in")
            _print_sellbird("via a headless browser session.")
        else:
            _print_sellbird("No problem — email outreach works great on its own.")
            self.env_vars["LINKEDIN_EMAIL"] = ""
            self.env_vars["LINKEDIN_PASSWORD"] = ""

    async def _setup_product(self):
        """Train Sellbird on the product — either via URL or manual entry."""
        _print_sellbird("Now the important part — I need to learn about what you're selling.")
        _print_sellbird("I can either crawl your website and figure it out myself,")
        _print_sellbird("or you can tell me the basics manually.\n")

        method = _ask(
            "Train from website URL or enter manually? (url/manual)",
            default="url",
        ).lower()

        if method in ("url", "u", "website"):
            url = _ask("Your product's website URL (e.g. https://yourcompany.com)")

            _print_sellbird(f"Give me a minute — I'm going to crawl {url}")
            _print_sellbird("and learn everything I can about your product.\n")

            from sellbird.trainer import Trainer
            trainer = Trainer(brain_config=self.brain_config)
            self.config = await trainer.train(url, str(_config_file())) or {}

            if self.config:
                _print_sellbird("Training complete! I've got a solid understanding")
                _print_sellbird("of your product now. Check sellbird.yaml to review")
                _print_sellbird("what I learned — tweak anything that looks off.")
                return
            else:
                _print_sellbird("Couldn't crawl that URL. Let's enter the details manually.")

        # Manual entry fallback
        self._setup_product_manual()

    def _setup_product_manual(self):
        """Collect product details through interactive prompts."""
        _print_sellbird("Tell me about what you're selling:\n")

        company = _ask("Your company name")
        name = _ask("Product name")
        description = _ask("What does it do? (1-2 sentences)")
        value_prop = _ask("Main value proposition (the #1 benefit to customers)")

        _print_sellbird("\nNow tell me about your ideal customer:\n")

        industries_raw = _ask(
            "Target industries (comma-separated, e.g. SaaS, Fintech, Healthcare)",
            default="B2B SaaS",
        )
        industries = [i.strip() for i in industries_raw.split(",") if i.strip()]

        titles_raw = _ask(
            "Target job titles (comma-separated, e.g. VP Sales, Head of Marketing)",
            default="Founder, CEO, VP Sales",
        )
        titles = [t.strip() for t in titles_raw.split(",") if t.strip()]

        geography_raw = _ask(
            "Target geography (comma-separated, e.g. United States, Canada)",
            default="United States",
        )
        geography = [g.strip() for g in geography_raw.split(",") if g.strip()]

        self.config = {
            "persona": {
                "company": company,
                "sender_name": "Sales Team",
                "sender_title": "Account Executive",
            },
            "product": {
                "name": name,
                "description": description,
                "value_proposition": value_prop,
            },
            "icp": {
                "industries": industries or ["B2B SaaS"],
                "titles": titles or ["VP", "Director"],
                "geography": geography,
            },
            "channels": {
                "email": {
                    "enabled": True,
                    "provider": getattr(self, "email_provider", "gmail"),
                    "max_daily_sends": 50,
                },
                "linkedin": {
                    "enabled": bool(self.env_vars.get("LINKEDIN_EMAIL")),
                    "max_daily_connections": 20,
                    "max_daily_messages": 10,
                },
            },
            "usage": {
                "max_daily_percent": 80,
                "heartbeat_interval_minutes": 15,
                "quiet_hours": {
                    "start": "22:00",
                    "end": "07:00",
                    "timezone": "America/New_York",
                },
            },
        }

        self.manual_product = True
        _print_sellbird("Got it. I have everything I need about your product.")

    def _setup_behavior(self):
        """Configure Sellbird's operational behavior."""
        _print_sellbird("Last thing — a few settings for how I operate.\n")

        # Usage limit
        print("  I track my own usage so I don't eat your whole daily capacity.")
        usage_pct = _ask(
            "Max % of daily usage I can consume",
            default="80",
        )
        try:
            usage_pct = min(100.0, max(1.0, float(usage_pct)))
        except ValueError:
            print("    (didn't catch that — using 80)")
            usage_pct = 80.0

        # Heartbeat interval
        print("\n  I wake up on a schedule to check for work.")
        interval = _ask(
            "How often should I check for work (minutes)?",
            default="15",
        )
        try:
            interval = max(1, int(interval))
        except ValueError:
            print("    (didn't catch that — using 15)")
            interval = 15

        # Quiet hours
        print("\n  I won't send emails or do outreach during quiet hours.")
        quiet_start = _ask("Quiet hours start (24h format)", default="22:00")
        quiet_end = _ask("Quiet hours end (24h format)", default="07:00")

        # Timezone
        saved_zone = _saved_timezone()
        tz_hint = f" [{saved_zone}]" if saved_zone else ""
        tz_prompt = f"Your timezone (e.g. America/New_York, Europe/London){tz_hint}"
        import zoneinfo
        while True:
            timezone = _ask(tz_prompt, default=saved_zone or "America/New_York", required=False)
            try:
                zoneinfo.ZoneInfo(timezone)
                break
            except (zoneinfo.ZoneInfoNotFoundError, ValueError):
                print(f"    '{timezone}' isn't a known timezone. Try America/New_York, "
                      "America/Los_Angeles, Europe/London, etc.")

        # Email sending limits
        print("\n  Email sending limits:")
        daily_sends = _ask("Max emails to send per day", default="50")
        try:
            daily_sends = max(1, int(daily_sends))
        except ValueError:
            daily_sends = 50

        # Update config dict
        if not self.config:
            self.config = {}

        self.config.setdefault("usage", {})
        self.config["usage"].update({
            "max_daily_percent": usage_pct,
            "heartbeat_interval_minutes": interval,
            "quiet_hours": {
                "start": quiet_start,
                "end": quiet_end,
                "timezone": timezone,
            },
        })

        self.config.setdefault("channels", {}).setdefault("email", {})
        self.config["channels"]["email"]["max_daily_sends"] = daily_sends

    def _write_env(self):
        """Write the .env file, preserving any existing variables we didn't touch."""
        existing = _read_env_file()
        merged = dict(existing)
        for key, value in self.env_vars.items():
            if value or key not in merged:
                merged[key] = value

        lines = [f"{key}={value}" for key, value in merged.items()]
        ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            ENV_FILE.chmod(0o600)
        except OSError:
            pass
        logger.info(f"Wrote {ENV_FILE}")

    def _write_config(self):
        """Write sellbird.yaml (only if we have manual config — trainer writes its own)."""
        from sellbird.trainer import brain_section

        config_file = _config_file()
        if self.config:
            self.config["brain"] = brain_section(self.brain_config)
        if self.config and not config_file.exists():
            with open(config_file, "w", encoding="utf-8") as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Wrote {config_file}")
            self.config_saved = True
        elif self.config:
            try:
                existing = yaml.safe_load(read_user_text(config_file)) or {}
                if self.manual_product:
                    for section in ("persona", "product", "icp"):
                        if section in self.config:
                            if not isinstance(existing.get(section), dict):
                                existing[section] = {}
                            _merge_answers(existing[section], self.config[section])
                existing["usage"] = self.config.get("usage", existing.get("usage", {}))
                existing["brain"] = self.config["brain"]
                if "LINKEDIN_EMAIL" in self.env_vars:
                    existing.setdefault("channels", {}).setdefault("linkedin", {})["enabled"] = bool(
                        self.env_vars["LINKEDIN_EMAIL"]
                    )
                if "channels" in self.config:
                    existing.setdefault("channels", {})
                    if "email" in self.config["channels"]:
                        existing["channels"].setdefault("email", {})
                        existing["channels"]["email"]["max_daily_sends"] = (
                            self.config["channels"]["email"].get("max_daily_sends", 50)
                        )
                text = yaml.dump(existing, default_flow_style=False, sort_keys=False)
                with open(config_file, "w", encoding="utf-8") as f:
                    f.write(text)
                self.config_saved = True
            except Exception as e:
                _print_sellbird(f"Couldn't save your settings to {config_file.name} ({e}).")
                _print_sellbird("Nothing in it was changed. Fix the file and re-run setup.")

    async def _print_summary(self):
        """Print final summary and next steps."""
        print(f"""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   Setup Complete!                                        ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")

        # Show what's configured
        checks = {
            "Local Brain": True,
            "LinkedIn": bool(self.env_vars.get("LINKEDIN_EMAIL")),
            "Product trained": _product_configured(),
        }

        for name, ok in checks.items():
            status = "✓" if ok else "–"
            print(f"  {status} {name}")

        print(f"""
  Files created:
    - .env (your credentials)
    - {_config_file().name} (your configuration)

  To start Sellbird:
    sellbird run          (or: python -m sellbird)

  To re-train on a different product:
    sellbird train https://newproduct.com

  To re-run this setup:
    sellbird setup

  Before you send a single email:
    1. Use a dedicated sending domain (not your main domain) with
       SPF, DKIM, and DMARC configured.
    2. Warm it up 2-4 weeks before real volume.
    3. Make sure your campaigns include an unsubscribe
       option and your physical mailing address (CAN-SPAM).
    See docs/deliverability.md.
""")
        _print_sellbird("I'm ready. Run 'sellbird run' and I'll start closing.")
        _print_sellbird("Make the first move.\n")


async def run_setup():
    wizard = SetupWizard()
    await wizard.run()


def main():
    asyncio.run(run_setup())


if __name__ == "__main__":
    from sellbird.textio import utf8_stdio
    utf8_stdio()
    main()
