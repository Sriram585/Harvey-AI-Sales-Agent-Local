"""Email discovery — pattern-first, verify-once, honest about confidence.

Local direct SMTP verification and pattern inference without external APIs.
1. Classifies the domain by its MX host (google / microsoft / gateway / other).
2. Derives the domain's email PATTERN once — from cache, scraped addresses,
   or defaults — collapsing N guesses to one candidate.
3. Verifies candidate addresses via direct local SMTP where supported.
4. Detects catch-all domains and buckets them as risky.

Every result carries an honest status: verified / risky / guess / invalid.
"""

import asyncio
import logging
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import dns.resolver
import aiosmtplib
import httpx

logger = logging.getLogger("sellbird.email_finder")

# Cache MX host + type per domain to avoid repeated lookups.
_mx_cache: dict[str, Optional[str]] = {}

API_TIMEOUT = httpx.Timeout(connect=10.0, read=20.0, write=10.0, pool=10.0)

# Named email-format patterns → builder. {f}=first, {l}=last, {fi}=first initial,
# {li}=last initial. Ordered by real-world prevalence for the default fallback.
PATTERN_BUILDERS = {
    "{f}.{l}": lambda f, l: f"{f}.{l}",
    "{f}{l}": lambda f, l: f"{f}{l}",
    "{fi}{l}": lambda f, l: f"{f[0]}{l}",
    "{f}": lambda f, l: f"{f}",
    "{fi}.{l}": lambda f, l: f"{f[0]}.{l}",
    "{f}_{l}": lambda f, l: f"{f}_{l}",
    "{l}": lambda f, l: f"{l}",
    "{l}.{f}": lambda f, l: f"{l}.{f}",
    "{f}-{l}": lambda f, l: f"{f}-{l}",
    "{l}{fi}": lambda f, l: f"{l}{f[0]}",
}
DEFAULT_PATTERN = "{f}.{l}"
# A business with a handful of people and no evidence of its format: the
# owner of a five-person roofing company is lou@, far more often than
# lou.smith@. Still only a guess — never sent until a verifier confirms it.
SMALL_BUSINESS_PATTERN = "{f}"
# Pattern-cache sources that are only a fallback, not evidence.
DEFAULT_SOURCES = ("default",)
WEAK_SOURCES = DEFAULT_SOURCES + ("site_address_guess",)


@dataclass
class EmailResult:
    """Outcome of an email lookup. status is the source of truth."""
    email: str
    status: str          # verified / risky / guess / invalid
    pattern: str = ""
    mx_type: str = ""

    @property
    def verified(self) -> bool:
        return self.status == "verified"

    @property
    def sendable(self) -> bool:
        """Safe to send cold: confirmed mailbox only (risky is opt-in elsewhere)."""
        return self.status == "verified"


def _clean_domain(domain: str) -> str:
    domain = (domain or "").strip().lower()
    domain = re.sub(r"^https?://", "", domain)
    domain = domain.split("/")[0].split("?")[0]
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def _clean_name_part(value: str) -> str:
    return re.sub(r"[^a-z]", "", (value or "").lower())


def build_email(pattern: str, first: str, last: str, domain: str) -> str:
    """Render a named pattern into an address, or '' if it can't be built."""
    first = _clean_name_part(first)
    last = _clean_name_part(last)
    domain = _clean_domain(domain)
    if not first or not last or not domain:
        return ""
    builder = PATTERN_BUILDERS.get(pattern)
    if not builder:
        return ""
    try:
        local = builder(first, last)
    except IndexError:
        return ""
    return f"{local}@{domain}" if local else ""


def generate_patterns(first_name: str, last_name: str, domain: str) -> list[str]:
    """All candidate addresses for a person, most-likely first (deduped)."""
    out = []
    for pattern in PATTERN_BUILDERS:
        email = build_email(pattern, first_name, last_name, domain)
        if email:
            out.append(email)
    return list(dict.fromkeys(out))


# ── MX classification ──

def classify_mx(mx_host: str) -> str:
    """Bucket a domain by its mail host — decides the verification strategy."""
    if not mx_host:
        return "none"
    h = mx_host.lower()
    if "google" in h or "googlemail" in h or "aspmx.l" in h:
        return "google"
    if "outlook" in h or "protection.outlook" in h or "microsoft" in h:
        return "microsoft"
    if any(g in h for g in ("pphosted", "mimecast", "barracuda", "proofpoint",
                            "messagelabs", "cisco", "fortinet", "trendmicro")):
        return "gateway"
    return "other"


async def get_mx_host(domain: str) -> Optional[str]:
    """Primary MX host for a domain (cached; None on failure)."""
    domain = _clean_domain(domain)
    if not domain:
        return None
    if domain in _mx_cache:
        return _mx_cache[domain]
    try:
        loop = asyncio.get_event_loop()
        answers = await loop.run_in_executor(
            None, lambda: dns.resolver.resolve(domain, "MX")
        )
        records = sorted(answers, key=lambda x: x.preference)
        mx_host = str(records[0].exchange).rstrip(".") if records else None
        _mx_cache[domain] = mx_host or None
        return _mx_cache[domain]
    except Exception as e:
        logger.debug(f"MX lookup failed for {domain}: {e}")
        _mx_cache[domain] = None
        return None


# ── Pattern derivation ──

def infer_pattern_from_email(email: str, first: str, last: str) -> Optional[str]:
    """Given one known address, work out which named pattern produced it."""
    email = (email or "").strip().lower()
    if "@" not in email:
        return None
    local = email.split("@", 1)[0]
    first = _clean_name_part(first)
    last = _clean_name_part(last)
    if not first or not last:
        return None
    for pattern, builder in PATTERN_BUILDERS.items():
        try:
            if builder(first, last) == local:
                return pattern
        except IndexError:
            continue
    return None


# Mailboxes that are a function, not a person: they never reveal how the
# business spells its people's addresses.
ROLE_LOCALS = frozenset("""
info office contact contactus hello hi team admin administrator support help sales
service services estimates estimate quotes quote inquiries inquiry enquiries enquiry
billing accounts accounting accountspayable ap ar invoices payments orders order
careers career jobs job hr recruiting employment apply marketing media press pr news
newsletter webmaster web website postmaster hostmaster abuse noreply no-reply donotreply
reception frontdesk front appointments appointment schedule scheduling dispatch
customerservice customercare care clients client mail email general main owner
manager management operations ops legal privacy compliance security it tech
feedback reviews referrals warranty claims insurance parts shop store
""".split())

def pattern_from_local(local: str) -> Optional[tuple[str, float]]:
    """The likely pattern behind one published address when nobody on the
    site is tied to it: corwin@ -> {f}, jsmith@ -> {fi}{l}, j.smith@ ->
    {fi}.{l}, john.smith@ -> {f}.{l}. None for role mailboxes and junk.

    Dotless addresses (corwin@, jsmith@) return None: alone they are
    ambiguous. The confidence is deliberately modest; an address tied to a
    named person (infer_pattern_from_email) is the stronger evidence.
    """
    local = (local or "").strip().lower()
    if not local or local in ROLE_LOCALS or any(ch.isdigit() for ch in local):
        return None
    m = re.fullmatch(r"([a-z]+)([._-])([a-z]+)", local)
    if m:
        first, sep, last = m.groups()
        if sep == "." and len(first) == 1 and len(last) >= 2:
            return "{fi}.{l}", 0.6
        if len(first) >= 2 and len(last) >= 2 and first not in ROLE_LOCALS and last not in ROLE_LOCALS:
            return {".": "{f}.{l}", "_": "{f}_{l}", "-": "{f}-{l}"}[sep], 0.6
        return None
    # A lone dotless address is ambiguous: kmorrow@ may be a first name or
    # K. Morrow, sven@ a first name or S. Ven. Only a named person on the site
    # can tell (see site_email_evidence), so on its own it teaches nothing.
    return None


def _disambiguate(local: str, people) -> Optional[tuple[str, float]]:
    """A dotless address no named person owns, read with the names the site
    does list: jsmith@ where an Amy Smith works is an initial plus a
    surname. Anything else stays ambiguous."""
    if not re.fullmatch(r"[a-z]{3,20}", local or ""):
        return None
    lasts = {_clean_name_part(last) for _, last in people or []} - {""}
    firsts = {_clean_name_part(first) for first, _ in people or []} - {""}
    if local in firsts:
        return "{f}", 0.6
    if any(len(last) >= 3 and local[1:] == last for last in lasts):
        return "{fi}{l}", 0.6
    if local in lasts:
        return "{l}", 0.55
    return None


def site_email_evidence(
    addresses: list[str], people: list[tuple[str, str]], domain: str,
) -> tuple[list[tuple[str, str, str]], str, float]:
    """What a site's own published addresses say about its email format.

    Returns (known, pattern, confidence): ``known`` is every on-domain
    address that belongs to a named person on the site (nvale@ for Nora
    Vale), ready for find_email(known_emails=...). ``pattern`` is the
    format those addresses agree on; failing that, the majority format of
    addresses that show a format without being one person's (j.smith@,
    jsmith@ when John and Jane Smith are both named, or jsmith@ where an Amy
    Smith works); else "". An address two named people could own is never
    returned as either one's.
    """
    domain = _clean_domain(domain)
    known: list[tuple[str, str, str]] = []
    tied_votes: dict[str, int] = {}
    loose_votes: dict[str, int] = {}
    for addr in addresses or []:
        addr = (addr or "").strip().lower()
        local, _, adomain = addr.partition("@")
        if adomain != domain or local in ROLE_LOCALS:
            continue
        owners = []
        for first, last in people or []:
            pattern = infer_pattern_from_email(addr, first, last)
            if pattern:
                owners.append((first, last, pattern))
        if len(owners) == 1:
            first, last, pattern = owners[0]
            known.append((addr, first, last))
            tied_votes[pattern] = tied_votes.get(pattern, 0) + 1
            continue
        if owners:
            # jsmith@ with John and Jane Smith both named: the format is
            # clear, whose mailbox it is isn't. Format evidence only.
            patterns = {o[2] for o in owners}
            if len(patterns) == 1:
                loose_votes[owners[0][2]] = loose_votes.get(owners[0][2], 0) + 2
            continue
        guess = pattern_from_local(local) or _disambiguate(local, people)
        if guess:
            loose_votes[guess[0]] = loose_votes.get(guess[0], 0) + 1
    if tied_votes:
        pattern = max(tied_votes, key=tied_votes.get)
        return known, pattern, 0.95 if tied_votes[pattern] >= 2 else 0.9
    if loose_votes:
        pattern = max(loose_votes, key=loose_votes.get)
        n = loose_votes[pattern]
        return known, pattern, min(0.55 + 0.05 * n, 0.7)
    return known, "", 0.0


async def derive_pattern(
    domain: str,
    known_emails: Optional[list[tuple[str, str, str]]] = None,
    state=None,
    default_pattern: str = DEFAULT_PATTERN,
) -> tuple[str, str, float]:
    """Determine a domain's email pattern. Returns (pattern, source, confidence)."""
    domain = _clean_domain(domain)

    for email, first, last in known_emails or []:
        if email.split("@")[-1].lower() == domain:
            inferred = infer_pattern_from_email(email, first, last)
            if inferred:
                return inferred, "known_email", 0.9

    cached = None
    if state is not None:
        try:
            cached = await state.get_email_pattern(domain)
        except Exception as e:
            logger.debug(f"Pattern cache read failed for {domain}: {e}")
    source = (cached or {}).get("source") or ""
    if cached and cached.get("pattern") and source not in WEAK_SOURCES:
        return cached["pattern"], source or "cache", cached.get("confidence", 0.6)

    site_guess = cached if cached and cached.get("pattern") and source.startswith("site_address_guess") else None
    if site_guess:
        return site_guess["pattern"], source, site_guess.get("confidence", 0.6)
    return default_pattern, "default", 0.3


async def verify_email_smtp(email: str) -> Optional[bool]:
    """SMTP RCPT check — only trustworthy for small, self-hosted mail servers.

    Returns True (accepted), False (rejected), or None (couldn't tell).
    Port 25 is blocked on most cloud hosts, so this frequently returns None.
    """
    if "@" not in email:
        return False
    domain = email.rsplit("@", 1)[1]
    mx_host = await get_mx_host(domain)
    if not mx_host:
        return None
    smtp = aiosmtplib.SMTP(hostname=mx_host, port=25, timeout=10)
    try:
        await smtp.connect()
        await smtp.ehlo()
        await smtp.mail("verify@example.com")
        code, _ = await smtp.rcpt(email)
        if code == 250:
            return True
        if code in (550, 551, 553):
            return False
        return None
    except Exception as e:
        logger.debug(f"SMTP verification failed for {email}: {e}")
        return None
    finally:
        try:
            await smtp.quit()
        except Exception:
            pass


async def detect_catch_all(domain: str) -> Optional[bool]:
    """Probe a definitely-fake address; if accepted, the domain is catch-all.

    Only works where SMTP probing works at all (small hosts). Returns None
    when we can't tell.
    """
    fake = f"zz-no-such-user-{random.randint(10000, 99999)}@{_clean_domain(domain)}"
    result = await verify_email_smtp(fake)
    if result is True:
        return True   # accepted a fake address → catch-all
    if result is False:
        return False  # rejected a fake address → not catch-all
    return None


def _status_from_verdict(verdict: dict) -> Optional[str]:
    """Map a provider verdict dict to our status, or None if inconclusive."""
    if verdict.get("catch_all"):
        return "risky"
    status = verdict.get("status", "")
    if status in ("valid", "deliverable", "safe"):
        return "verified"
    if status in ("invalid", "undeliverable", "disabled", "disposable", "spamtrap"):
        return "invalid"
    if status in ("catch_all", "catch-all", "accept_all"):
        return "risky"
    return None  # unknown / risky-unknown — inconclusive


async def _verify_candidate(email: str, mx_type: str, env) -> tuple[Optional[str], bool]:
    """Run local verifier for this domain type."""
    catch_all_seen = False
    if mx_type in ("google", "microsoft", "gateway"):
        return None, catch_all_seen

    smtp = await verify_email_smtp(email)
    if smtp is True:
        if await detect_catch_all(email.split("@")[1]) is True:
            return "risky", True
        return "verified", catch_all_seen
    if smtp is False:
        return "invalid", catch_all_seen

    return None, catch_all_seen



