"""DISCOVER — who exists, and how visible they are.

The only stage that costs money, so it is the one with a provider menu, a
mandatory cost estimate, and a budget check inside the loop.

Two shapes of discovery, and the difference matters:

* **Listings** ask "which businesses of this type are in this place?" and come
  back with a phone, a category, a rating and — crucially — whether there is a
  website at all. For local SMB verticals this is the primary path.
* **SERP** asks "who ranks for this search?" and comes back with a rank
  position. You are buying the *signal*, not the list: positions 11-30 are
  businesses visible enough to be trying and not winning, which is the most
  useful prospecting signal there is.

Cost discipline lives here, not in the database. A table can only record spend
after the fact; only the collector can decline to spend it. Every run
estimates first, checks a budget between batches, reads the kill switch, and
flushes observations per batch — a long run that dies must not lose what it
already learned.
"""

import asyncio
import time
import base64
import logging
import math
import re
from html import unescape
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from urllib.parse import urlparse

import httpx

from sellbird.models import Company

logger = logging.getLogger("sellbird.collectors.discover")

TIMEOUT = httpx.Timeout(connect=10.0, read=45.0, write=10.0, pool=10.0)


# ── Junk filtering ────────────────────────────────────────────────────
# Discovery fills a database with directories, hospital systems, media and
# lead-gen middlemen unless you filter hard. Match PATTERNS, not a literal
# list: a literal list is exactly how `whatclinic.com` and `ufhealth.org` get
# through on the first run.

# Anchored on a dot boundary, not the string start: SERP results arrive as
# `en.wikipedia.org` and `m.facebook.com` at least as often as bare domains,
# and `^(www\.)?` catches neither.
JUNK_DOMAIN_PATTERNS = tuple(re.compile(p, re.I) for p in (
    # directories & marketplaces
    r"(^|\.)(yelp|yellowpages|superpages|manta|bbb|angi|angieslist|homeadvisor)\.",
    r"(^|\.)(thumbtack|houzz|porch|buildzoom|networx|modernize|nextdoor)\.",
    r"(^|\.)(facebook|instagram|twitter|x|linkedin|youtube|tiktok|pinterest)\.",
    r"(^|\.)(meetup|eventbrite)\.",
    r"(^|\.)(indeed|glassdoor|ziprecruiter|craigslist|zillow|realtor)\.",
    r"(^|\.)(mapquest|foursquare|tripadvisor|trustpilot|birdeye|expertise)\.",
    r"(directory|listings?|find-?a-|top\d+|best-?of)\.",
    # Job boards, directories, and marketplaces.
    r"(^|\.)(naukri|jobrapido|jobsora|empleos|shine|foundit|timesjobs|freelancer|upwork)\.",
    # Indian directories & marketplaces
    r"(^|\.)(indiamart|justdial|tradeindia|exportersindia|sulekha|urbanpro|yet5)\.",
    r"(^|\.)(olx|quikr|flipkart|snapdeal|myntra|shopclues|etsy|ebay|alibaba|aliexpress)\.",
    # hobby communities & model repositories — never a prospect
    r"(^|\.)(boardgamegeek|team-bhp|hackaday|instructables|thingiverse|printables|cults3d|thangs|grabcad)\.",
    # media & reference
    r"(^|\.)(wikipedia|reddit|quora|medium|substack|forbes|inc|entrepreneur)\.",
    r"(news|magazine|journal|gazette|tribune|herald)\.(com|org|net)$",
    # institutional
    r"\.(edu|gov|mil)$",
    r"(hospital|healthsystem|medicalcenter)\.",
    r"health\.org$",
    # startup databases, launch sites and tech press (the YC preset's noise)
    r"(^|\.)(ycombinator|workatastartup|startupschool|yclist|ycdb|seedtable)\.",
    r"(^|\.)(crunchbase|techcrunch|wellfound|angel|tracxn|pitchbook)\.",
    r"(^|\.)(cbinsights|producthunt|f6s|dealroom|growjo|owler|zoominfo)\.",
    r"(^|\.)github\.com$",
    # hosted job boards: the company is whoever's board it is, not the host
    r"(^|\.)(job-)?boards\.(eu\.)?greenhouse\.io$",
    r"(^|\.)jobs\.(eu\.)?lever\.co$",
    r"(^|\.)jobs\.ashbyhq\.com$",
    r"(^|\.)(apply|jobs)\.workable\.com$",
    r"(^|\.)(jobs|careers)\.smartrecruiters\.com$",
    r"(^|\.)jobs\.jobvite\.com$",
    r"(^|\.)ats\.rippling\.com$",
    r"(^|\.)(myworkdayjobs|applytojob)\.com$",
    # a company's board on these lives on a subdomain; the bare domain is
    # the vendor, which may be someone's prospect
    r"\.(icims|bamboohr|recruitee|teamtailor)\.com$",
    r"\.breezy\.hr$",
    r"\.jobs\.personio\.(de|com)$",
    # job sites named for the job: jobs.com, careers.co.uk
    r"^(jobs|careers)\.((co|com|org|net|ac)\.)?[a-z]{2,}$",
    # accelerators' own sites (the accelerator preset's noise)
    r"(^|\.)(techstars|antler|joinef|southparkcommons|aigrant|a16z|pioneer|plugandplaytechcenter|plugandplay)\.",
    r"(^|\.)(500\.co|neo\.com)$",
    # funding news, wires and round trackers (the funding preset's noise)
    r"(^|\.)(prnewswire|businesswire|globenewswire|accesswire|einpresswire|newswire|finsmes|axios)\.",
    r"(^|\.)(venturebeat|siliconangle|eu-startups|sifted|geekwire|thesaasnews|fortune|bloomberg)\.",
    r"(^|\.)(reuters|wsj|cnbc|businessinsider|privco|recentfunding|pulse2|startupfundraising)\.",
    r"(^|\.)(startupintros|openvc|siliconcanals|startupdaily|e27|similarweb|vcbacked|roundos)\.",
    r"(^|\.)(techfundingnews|startups\.gallery)",
    r"(^|\.)tech\.eu$",
    # infrastructure that is never a prospect
    r"(^|\.)(google|bing|yahoo|duckduckgo|amazon|apple|microsoft)\.",
    r"(wordpress|wix|squarespace|shopify|godaddy|weebly|webflow)\.com$",
))

# Names that mean the row is a category page or an aggregator, not a business.
JUNK_NAME_PATTERNS = tuple(re.compile(p, re.I) for p in (
    r"^\s*(top|best|\d+)\s+\d*\s*(best\s+)?\w+.*\b(in|near|of)\b",
    r"\b(directory|listings?|reviews? of|compare|find a)\b",
    # "Ohio HOA Management Company List", "List of Plumbers in Austin"
    r"\b(compan(y|ies)|firms?|contractors?|businesses|providers?)\s+list\b",
    r"\blist of\b",
))


def normalize_domain(url_or_domain: str) -> str:
    """A bare, lowercase, www-less hostname — the entity-resolution key.

    Entity resolution decides whether any of this is real. Get it wrong and
    every trend built on top is garbage.
    """
    if not url_or_domain:
        return ""
    raw = url_or_domain.strip().lower()
    if "://" not in raw:
        raw = "http://" + raw
    host = (urlparse(raw).hostname or "").strip(".")
    return host[4:] if host.startswith("www.") else host


# ── Business names ────────────────────────────────────────────────────
# SERP providers return a page TITLE, not a business name: "Denver, CO | HOA
# Management Firm", "Home Page - NVCM | Community Association Management".
# Stored as-is it lands in an email as "...at Denver, CO | HOA Management
# Firm". The business is the title segment that shares a DISTINCTIVE word with
# the domain — "management" matches managementharbor.com but says nothing.

_TITLE_SEPARATORS = re.compile(r"\s+[|\-–—·•]\s+|\s*[|•·]\s*|:\s+")
_GENERIC_NAME_WORDS = frozenset("""
home page welcome official site website hoa hoas management manager managers
mgmt mgt company companies co inc llc ltd corp firm group services service
property properties community communities association associations condo
condominium residential homeowners owners real estate the and of in for best
top leading trusted local full arizona phoenix az texas tx florida fl
california ca nevada nv colorado co georgia ga carolina nc usa us america
roofing roof roofers roofer plumbing plumber plumbers contact about
""".split())


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w]


def _domain_stem(domain: str) -> str:
    host = normalize_domain(domain)
    return re.sub(r"[^a-z0-9]", "", host.rsplit(".", 1)[0]) if host else ""


def _segment_matches_domain(segment: str, stem: str) -> bool:
    # Navigation and service/location labels can share an incidental word
    # with a domain without identifying the business.
    if re.fullmatch(r"(?i)(contact(?: us)?|about(?: us)?|home(?: page)?)", segment.strip()):
        return False
    if re.search(r"(?i)\bservices?\s+(?:in|near)\s+\S", segment):
        return False
    words = _words(segment)
    distinctive = [w for w in words if w not in _GENERIC_NAME_WORDS]
    if any(len(w) >= 3 and w in stem for w in distinctive):
        return True
    # "Northvale Association of Community Managers" ↔ nacm.org
    initials = "".join(w[0] for w in words if w not in ("of", "the", "and", "&"))
    return len(initials) >= 3 and stem.startswith(initials)


def _segment_domain_evidence(segment: str, stem: str) -> int:
    """Amount of distinctive domain evidence, shared by SERP and homepage names."""
    if not _segment_matches_domain(segment, stem):
        return 0
    return sum(len(w) for w in set(_words(segment))
               if w not in _GENERIC_NAME_WORDS and len(w) >= 3 and w in stem)


_SPLIT_WORDS = sorted((w for w in _GENERIC_NAME_WORDS | {"with", "your"}
                       if len(w) >= 4), key=len, reverse=True)


def _split_stem(stem: str) -> list[str]:
    """managementharbor → [management, harbor]; kqresidential → [kq, residential]."""
    # Find every occurrence of every split word in one sweep; scanning the
    # words longest-first keeps the longest word for each start position.
    starts: dict[int, str] = {}
    for w in _SPLIT_WORDS:
        i = stem.find(w)
        while i >= 0:
            starts.setdefault(i, w)
            i = stem.find(w, i + 1)
    parts, start = [], 0
    for i in sorted(starts):
        if i >= start:
            if start < i:
                parts.append(stem[start:i])
            word = starts[i]
            parts.append(word)
            start = i + len(word)
    if start < len(stem):
        parts.append(stem[start:])
    return parts


def name_from_domain(domain: str) -> str:
    stem = normalize_domain(domain).rsplit(".", 1)[0]
    words = [w for chunk in re.split(r"[-_.]+", stem) if chunk for w in _split_stem(chunk)]
    # Short vowel-less leftovers are initials: "kq" → "KQ".
    return " ".join(w.upper() if len(w) <= 3 and not re.search(r"[aeiouy]", w)
                    else w.capitalize() for w in words)


def name_matches_domain(name: str, domain: str) -> bool:
    """True when the stored name demonstrably belongs to this domain."""
    stem = _domain_stem(domain)
    return bool(stem and name and _segment_matches_domain(name, stem))


def site_name_from_html(html: str, domain: str) -> str:
    """The business name a homepage gives itself (og:site_name, then <title>),
    or "" when neither names it in a way the domain backs up."""
    stem = _domain_stem(domain)
    candidates = []
    for pat in (r'<meta[^>]+property=["\']og:site_name["\'][^>]*content=["\']([^"\']+)',
                r'<meta[^>]+content=["\']([^"\']+)["\'][^>]*property=["\']og:site_name',
                r"(?is)<title[^>]*>(.*?)</title>"):
        m = re.search(pat, html or "", re.I)
        if m:
            candidates.append(unescape(re.sub(r"\s+", " ", m.group(1))).strip())
    # Distinctive brand evidence wins; equally backed names prefer the fullest
    # form: "The Management Harbor" over og's "Management Harbor".
    matches = [seg for text in candidates
               for seg in (x.strip(" ,.-") for x in _TITLE_SEPARATORS.split(text))
               if seg and len(seg) <= 60 and _segment_matches_domain(seg, stem)]
    return max(matches, key=lambda seg: (_segment_domain_evidence(seg, stem), len(seg)),
               default="")


def business_name(title: str, domain: str) -> str:
    """The business's name from a SERP title, or one derived from its domain."""
    stem = _domain_stem(domain)
    title = re.sub(r"\s*(\.\.\.|…)\s*$", "", (title or "").strip())
    if stem and title:
        segments = [segment.strip(" ,.-") for segment in _TITLE_SEPARATORS.split(title)]
        matches = [seg for seg in segments
                   if seg and len(seg) <= 60 and _segment_matches_domain(seg, stem)]
        # Prefer the segment with the most distinctive domain evidence rather
        # than the first incidental substring. Stable ties preserve title order.
        if matches:
            return max(matches, key=lambda seg: _segment_domain_evidence(seg, stem))
    return name_from_domain(domain) if stem else title


def is_junk(domain: str, name: str = "") -> bool:
    """True when this row would pollute the database rather than fill it."""
    if domain:
        if any(p.search(domain) for p in JUNK_DOMAIN_PATTERNS):
            return True
        # A bare TLD or an IP address is never a business we can profile.
        if "." not in domain or re.fullmatch(r"[\d.]+", domain):
            return True
    if name and any(p.search(name) for p in JUNK_NAME_PATTERNS):
        return True
    return False


# ── Geocoding ─────────────────────────────────────────────────────────
# Listings providers search a radius, not a place name, so "Denver, CO" has to
# become coordinates. Nominatim allows exactly this (single place lookups at
# <=1 req/s) and forbids bulk enumeration, so every result is cached forever in
# the settings table: a given city is looked up once, ever.

NOMINATIM = "https://nominatim.openstreetmap.org/search"
GEOCODE_UA = "Sellbird/0.1 (open-source sales agent; github.com/ethanplusai/sellbird)"


async def geocode_city(state, city: str, radius_km: int = 40,
                       client: httpx.AsyncClient | None = None) -> str:
    """"Denver, CO" -> "39.7392,-104.9903,40". Cached permanently."""
    if not city:
        return ""
    cache_key = f"geo:{city.strip().lower()}"
    cached = await state.get_setting(cache_key)
    if cached:
        # A previous miss is cached as "-" so we don't re-ask every run.
        return "" if cached == "-" else f"{cached},{radius_km}"

    owned = client is None
    client = client or httpx.AsyncClient(timeout=TIMEOUT)
    try:
        r = await client.get(
            NOMINATIM,
            params={"q": city, "format": "json", "limit": 1},
            headers={"User-Agent": GEOCODE_UA},
        )
        r.raise_for_status()
        hits = r.json()
    except Exception as e:
        logger.warning("geocode failed for %r: %s", city, e)
        return ""
    finally:
        if owned:
            await client.aclose()

    if not hits:
        await state.set_setting(cache_key, "-")
        logger.warning("geocode: no match for %r", city)
        return ""

    point = f"{float(hits[0]['lat']):.4f},{float(hits[0]['lon']):.4f}"
    await state.set_setting(cache_key, point)
    return f"{point},{radius_km}"


async def resolve_coordinates(state, queries: list["DiscoveryQuery"],
                              radius_km: int = 40,
                              client: httpx.AsyncClient | None = None
                              ) -> list["DiscoveryQuery"]:
    """Fill in coordinates for any query that lacks them.

    Configured ``icp.geo_coordinates`` always wins; this only covers what the
    user did not spell out. Distinct cities are geocoded once each, however
    many queries share them.
    """
    needed = {q.location for q in queries if q.location and not q.coordinate}
    points: dict[str, str] = {}
    for i, city in enumerate(sorted(needed)):
        if i:
            await asyncio.sleep(1.1)     # Nominatim: max 1 request/second
        points[city] = await geocode_city(state, city, radius_km, client)

    for query in queries:
        if not query.coordinate:
            query.coordinate = points.get(query.location, "")
    return queries


# ── The record every provider returns ─────────────────────────────────


@dataclass
class Business:
    """One discovered business, normalised across providers."""

    name: str = ""
    domain: str = ""
    website: str = ""
    phone: str = ""
    address: str = ""
    location: str = ""
    category: str = ""
    rating: float | None = None
    review_count: int | None = None
    is_claimed: bool | None = None
    has_website: bool | None = None
    rank: int | None = None          # organic rank, when the provider gives one
    local_pack_rank: int | None = None   # slot in Google's local 3-pack (ads excluded)
    maps_rank: int | None = None     # position in Google Maps results
    keyword: str = ""                # the search that produced the ranks above
    external_id: str = ""            # "<provider>:<stable id>"
    alt_ids: tuple[str, ...] = ()    # other ids the same business was stored under
    source_url: str = ""
    provider: str = ""
    raw_title: str = ""             # SERP evidence for junk filtering before normalization

    def identity(self) -> str:
        return self.domain or self.external_id


@dataclass
class DiscoveryQuery:
    """What to look for, and where."""

    term: str                        # "roofing contractor"
    location: str = ""               # "Denver, CO"
    category: str = ""               # provider-native category slug, optional
    coordinate: str = ""             # "lat,lng,radius_km" for listings providers
    depth: int = 30                  # SERP depth; ignored by listings providers
    limit: int = 100                 # max records to take from one query
    preset: str = ""                 # PRESETS key that built it, if any

    def keyword(self) -> str:
        return f"{self.term} {self.location}".strip()


# ── Provider adapters ─────────────────────────────────────────────────


# Match whole placeholder values, never substrings of opaque credentials.
_PLACEHOLDER_RE = re.compile(
    r"(?i)(placeholder|change[-_]?me|your[_-](?:api[_-])?key(?:[_-]here)?"
    r"|paste[_-](?:api[_-])?key(?:[_-]here)?|x{4,}|<[a-z_ -]+>"
    r"|todo|fixme|example[_-](?:api[_-])?key)"
)


class ProviderExhausted(RuntimeError):
    """This provider can take no more searches in this run."""


class DiscoveryProvider:
    """One way of answering "who exists"."""

    key: str = ""
    label: str = ""
    kind: str = "listings"           # listings | serp
    blurb: str = ""
    cost_note: str = ""
    free_tier: str = ""
    signup_url: str = ""
    env_keys: tuple[str, ...] = ()
    caveat: str = ""
    needs_coordinates: bool = False
    rank_codes: tuple[str, ...] = ()

    def missing_keys(self, env: dict) -> list[str]:
        def _missing(value) -> bool:
            if not isinstance(value, str):
                return not value
            value = value.strip()
            return (not value) or bool(_PLACEHOLDER_RE.fullmatch(value))

        return [k for k in self.env_keys if _missing(env.get(k))]

    def configured(self, env: dict) -> bool:
        return not self.missing_keys(env)

    def estimate(self, queries: list[DiscoveryQuery]) -> float:
        return 0.0

    async def fetch(self, client: httpx.AsyncClient, env: dict,
                    query: DiscoveryQuery) -> list[Business]:
        raise NotImplementedError

    last_cost: float = 0.0
    last_error: str = ""
    stop_reason: str = ""
    complete: bool = True
    remaining_budget: float = math.inf


class OpenStreetMap(DiscoveryProvider):
    """Genuinely free, no account, no key. Honest about what that costs you.

    Good enough to try the whole pipeline end to end without signing up for
    anything, and a useful cross-reference layer afterwards. Not good enough
    to be your only source: OSM maps physical premises, so service-area
    businesses (roofers, HVAC, plumbers) are heavily under-represented.
    """

    key = "osm"
    label = "OpenStreetMap (Overpass)"
    kind = "listings"
    needs_coordinates = True
    blurb = ("Free public map data. No signup, no key, no card. Try the whole "
             "pipeline for nothing before you pay anyone.")
    cost_note = "free"
    free_tier = "unlimited, within community rate etiquette"
    signup_url = "https://www.openstreetmap.org"
    env_keys = ()
    caveat = ("Coverage is thin for any trade without a storefront — OSM maps "
              "premises, so it holds under 2,000 roofers for the entire US. "
              "Great for a free trial run or as a cross-reference; not a "
              "complete list.")

    # Several public Overpass instances run the same API. The main one
    # throttles readily, so try the mirrors before giving up — a free source
    # that fails half the time is not usable as a default.
    # Two entries, two conditions, both measured — see
    # scripts/diagnose_overpass.py, which re-checks them.
    #
    # Each must be its OWN instance: overpass.private.coffee sat here and
    # resolved to the same address as overpass.kumi.systems, so a 502 on the
    # first mirror bought two attempts at one machine and none at a second.
    #
    # Each must also hold the WHOLE planet. The community instances answer in
    # a fraction of a second and are independent, which makes them tempting:
    # overpass.osm.ch tallied ZERO cafes within a kilometre of downtown Denver
    # in 0.21s, so it serves a regional extract, and overpass.osm.jp is the
    # same kind of instance. A mirror like that answers "roofers in Denver"
    # with an empty list, and an empty list here is indistinguishable from a
    # market with nobody in it — the one failure this provider must never
    # produce. Vet a candidate with
    # `python scripts/diagnose_overpass.py --coverage <url>` before adding it;
    # independence at the DNS level is not enough.
    ENDPOINTS = (
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    )
    # Overpass etiquette for a shipped tool: modest volume, no parallelism.
    MIN_INTERVAL = 2.0
    BACKOFF = 5.0
    # A free instance queues before it computes: node(1) — the smallest query
    # Overpass accepts — measured 38s of pure waiting on a busy afternoon. The
    # module-wide 45s read budget leaves nothing for a real query over a 40km
    # radius, so Overpass gets its own and asks the server for a matching
    # slice of execution time. Slow beats empty-handed, and a run that cannot
    # afford the wait stops on MAX_PROSPECTS_PER_CYCLE, not on a timeout.
    HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=10.0,
                                 pool=10.0)
    SERVER_TIMEOUT = 90
    # Statuses that say "this mirror, right now" rather than "this query":
    # throttling, a gateway that lost its backend, an overloaded instance, a
    # timed-out one. Every one of them is another mirror's problem to solve, so
    # walk on. A 400 (bad query) or 404 is ours and fails here and now.
    RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

    # A friendly term maps to the OSM tags that actually carry those businesses.
    TAG_MAP = {
        "roof": ['craft=roofer', 'shop=roofing'],
        "plumb": ['craft=plumber'],
        "hvac": ['craft=hvac'],
        "electric": ['craft=electrician'],
        "dentist": ['amenity=dentist'],
        "lawyer": ['office=lawyer'],
        "attorney": ['office=lawyer'],
        "contractor": ['craft=builder', 'office=construction_company'],
        "landscap": ['craft=gardener', 'shop=garden_centre'],
        # Spanish terms
        "ferreter": ['shop=hardware', 'shop=doityourself'],
        "supermercado": ['shop=supermarket'],
        "abasto": ['shop=convenience', 'shop=supermarket'],
        "distribuidora": ['shop=wholesale'],
        "al mayor": ['shop=wholesale'],
        "por mayor": ['shop=wholesale'],
        "mayorista": ['shop=wholesale'],
        "contad": ['office=accountant'],
        "contable": ['office=accountant'],
        "farmacia": ['amenity=pharmacy'],
        "repuesto": ['shop=car_parts'],
        "electrodom": ['shop=electronics', 'shop=appliance'],
        "tienda de ropa": ['shop=clothes'],
        "panader": ['shop=bakery'],
        "clinica": ['amenity=clinic'],
        "clínica": ['amenity=clinic'],
    }

    def _tags(self, term: str) -> list[str]:
        """The OSM tags that actually carry this kind of business, or none.

        There is deliberately no catch-all fallback. An earlier version fell
        back to `office=company`, which answers "roofers in Denver" with every
        registered office in Denver — coffee roasters, churches, photography
        studios. A search this source cannot answer has to say so; filling the
        database with plausible-looking junk is far worse than returning
        nothing.
        """
        low = term.lower()
        for needle, tags in self.TAG_MAP.items():
            if needle in low:
                return tags
        return []

    async def fetch(self, client, env, query):
        if not query.coordinate:
            logger.warning("osm: no coordinate for %r — Overpass needs a "
                           "lat,lng,radius to search", query.keyword())
            return []
        tags = self._tags(query.term)
        if not tags:
            raise RuntimeError(
                f"OpenStreetMap has no category for {query.term!r}. It can only "
                f"search mapped trades ({', '.join(sorted(self.TAG_MAP))}). Use a "
                f"listings or SERP provider for this one.")
        lat, lng, radius_km = (query.coordinate.split(",") + ["", "", ""])[:3]
        radius_m = int(float(radius_km or 25) * 1000)
        clauses = "".join(
            f'node[{tag.split("=")[0]}={tag.split("=")[1]}](around:{radius_m},{lat},{lng});'
            f'way[{tag.split("=")[0]}={tag.split("=")[1]}](around:{radius_m},{lat},{lng});'
            for tag in tags
        )
        body = (f"[out:json][timeout:{self.SERVER_TIMEOUT}];({clauses});"
                f"out center {query.limit};")

        # Overpass is a free volunteer service and throttles, 502s and times
        # out readily under load. Walk the mirrors, then RAISE: returning []
        # would make a throttled run look exactly like a market with no
        # businesses in it.
        headers = {"User-Agent": "Sellbird/0.1 (open-source sales agent; "
                                 "github.com/ethanplusai/sellbird)"}
        r = None
        for i, endpoint in enumerate(self.ENDPOINTS):
            if i:
                await asyncio.sleep(self.BACKOFF)
            try:
                r = await client.post(endpoint, content=body.encode(),
                                      headers=headers,
                                      timeout=self.HTTP_TIMEOUT)
            except httpx.HTTPError as e:
                logger.warning("osm: %s unreachable (%s)", endpoint, e)
                continue
            if r.status_code not in self.RETRY_STATUSES:
                break
            logger.warning("osm: %s is throttling or down (%s) — trying the "
                           "next mirror", endpoint, r.status_code)
        else:
            raise RuntimeError(
                "Every public Overpass mirror is unavailable or throttling "
                "right now. This is free volunteer infrastructure — wait a few "
                "minutes and re-run, or use a paid provider for bulk work.")
        if r is None:
            raise RuntimeError("Could not reach any Overpass mirror.")
        r.raise_for_status()

        out = []
        for el in r.json().get("elements") or []:
            tags = el.get("tags") or {}
            name = tags.get("name") or ""
            if not name:
                continue
            site = tags.get("website") or tags.get("contact:website") or ""
            city = ", ".join(x for x in (tags.get("addr:city"),
                                         tags.get("addr:state")) if x)
            out.append(Business(
                name=name,
                domain=normalize_domain(site),
                website=site,
                phone=tags.get("phone") or tags.get("contact:phone") or "",
                address=" ".join(x for x in (tags.get("addr:housenumber"),
                                             tags.get("addr:street")) if x),
                location=city or query.location,
                category=tags.get("craft") or tags.get("shop") or tags.get("office") or "",
                has_website=bool(site),
                external_id=f"osm:{el.get('type')}/{el.get('id')}",
                source_url=f"https://www.openstreetmap.org/{el.get('type')}/{el.get('id')}",
                provider=self.key,
            ))
        return out


PROVIDERS: dict[str, DiscoveryProvider] = {
    p.key: p for p in (
        OpenStreetMap(),
    )
}

DEFAULT_PROVIDER = "osm"


# ── The run ───────────────────────────────────────────────────────────


@dataclass
class DiscoveryReport:
    """What a run did — the honest version, including what it skipped."""

    provider: str = ""
    queries: int = 0
    queries_ran: int = 0     # attempted provider requests
    queries_succeeded: int = 0
    estimated_cost: float = 0.0
    actual_cost: float = 0.0
    found: int = 0
    junk: int = 0
    # Counted apart from `junk` on purpose: junk is a judgement about whether
    # a result is a business at all, this is a company the operator has
    # forbidden. Folding them together would hide a do-not-contact list that
    # has grown too broad inside a number nobody questions.
    excluded: int = 0
    new_companies: int = 0
    known_companies: int = 0
    observations: int = 0
    run_id: str = ""
    errors: list[str] = field(default_factory=list)
    stopped: str = ""

    def as_dict(self) -> dict:
        return {
            "provider": self.provider, "queries": self.queries,
            "queries_ran": self.queries_ran,
            "queries_succeeded": self.queries_succeeded,
            "estimated_cost": round(self.estimated_cost, 4),
            "actual_cost": round(self.actual_cost, 4),
            "found": self.found, "junk": self.junk,
            "excluded": self.excluded,
            "new_companies": self.new_companies,
            "known_companies": self.known_companies,
            "observations": self.observations,
            "run_id": self.run_id, "errors": self.errors,
            "stopped": self.stopped,
        }


# Built-in searches beyond "your ICP". Each says how to turn an industry into
# a search, and which kind of provider can run it.
PRESETS = {
    # Startups that say on their own site they're YC-backed. Never
    # ycombinator.com itself, whose terms forbid scraping and harvesting
    # contacts; its pages and the startup databases are excluded too.
    "yc": {
        "label": "YC startups",
        "kinds": ("serp",),
        "term": ('"backed by Y Combinator" {industry} -site:ycombinator.com '
                 "-site:workatastartup.com -site:startupschool.org "
                 "-site:linkedin.com -site:crunchbase.com"),
        "local": False,
    },
    # Companies hiring for one of your hiring_signals roles (else ICP titles),
    # found by their OWN careers pages. The hosted boards are excluded: a
    # boards.greenhouse.io hit names a board, not a website to profile.
    # Profiling then reads the board their site links to and records the
    # role (HIRING_ROLE); the search hit alone records nothing.
    "hiring": {
        "label": "Companies hiring for your roles",
        "kinds": ("serp",),
        "term": ('"{role}" careers {industry} -site:greenhouse.io -site:lever.co '
                 "-site:ashbyhq.com -site:workable.com -site:smartrecruiters.com "
                 "-site:jobvite.com -site:icims.com -site:bamboohr.com -site:recruitee.com "
                 "-site:teamtailor.com -site:personio.de -site:breezy.hr -site:applytojob.com "
                 "-site:myworkdayjobs.com -site:ats.rippling.com -site:linkedin.com "
                 "-site:indeed.com -site:glassdoor.com -site:ziprecruiter.com"),
        "local": False,
    },
    # Startups that say on their own site an accelerator backed them. Two
    # searches per industry, because Google reads only the first 32 words.
    "accelerator": {
        "label": "Accelerator startups",
        "kinds": ("serp",),
        "terms": (
            '{industry} ("backed by Techstars" OR "backed by Antler" OR "backed by Entrepreneur First" '
            'OR "backed by South Park Commons" OR "backed by 500 Global") '
            "-site:linkedin.com -site:crunchbase.com",
            # Not "backed by Pioneer": it finds Pioneer Square Labs and Pioneer
            # Fund (the profile still reads a Pioneer claim on the page).
            '{industry} ("backed by Neo" OR "backed by AI Grant" OR "a16z speedrun" '
            'OR "backed by Plug and Play") '
            "-site:linkedin.com -site:crunchbase.com -site:a16z.com -site:neo.com",
        ),
        "local": False,
    },
    # Companies announcing their own raise, from the last ~18 months. The
    # news sites and databases are excluded here and filtered as junk.
    "funding": {
        "label": "Recently funded",
        "kinds": ("serp",),
        "term": ('{industry} ("we raised" OR "we\'ve raised" OR announces OR raises) '
                 '("seed round" OR "Series A" OR "Series B") after:{since} '
                 "-site:techcrunch.com -site:crunchbase.com -site:pitchbook.com -site:linkedin.com "
                 "-site:prnewswire.com -site:businesswire.com -site:globenewswire.com"),
        "local": False,
    },
}
FUNDING_LOOKBACK_DAYS = 548       # ~18 months: older raises aren't news


def preset_templates(spec: dict) -> tuple[str, ...]:
    """A preset's search templates: one ``term``, or several ``terms``."""
    return tuple(spec.get("terms") or (spec["term"],))

# A hit on careers.acme.com is acme.com's careers page.
_CAREERS_HOST = re.compile(r"^(careers|jobs|join|work|apply|hiring)\.", re.I)
_SECOND_LEVEL_SUFFIX = {"co", "com", "org", "net", "ac", "gov", "edu", "ltd", "plc", "or", "ne", "go"}


def company_host(domain: str) -> str:
    """careers.acme.com → acme.com. Never down to a public suffix: jobs.co.uk
    stays jobs.co.uk (and is junk), not co.uk."""
    stripped = _CAREERS_HOST.sub("", domain)
    labels = stripped.split(".")
    if stripped == domain or len(labels) < 2:
        return domain
    if len(labels) == 2 and labels[0] in _SECOND_LEVEL_SUFFIX and len(labels[1]) == 2:
        return domain
    return stripped


def preset_problem(preset: str, provider_key: str, config=None) -> str | None:
    """Why ``preset`` can't run on ``provider_key`` (with ``config``), or None."""
    if not preset:
        return None
    spec = PRESETS.get(preset)
    if spec is None:
        return f"Unknown preset {preset!r}. Presets: {', '.join(PRESETS)}."
    provider = PROVIDERS.get(provider_key)
    if provider is None or provider.kind not in spec["kinds"]:
        names = ", ".join(k for k, p in PROVIDERS.items() if p.kind in spec["kinds"])
        return (f"The {spec['label']} preset is a web search, so it needs a search "
                f"provider ({names}), not {provider_key}.")
    if any("{role}" in tpl for tpl in preset_templates(spec)) and config is not None:
        from sellbird.collectors.ats import role_keywords
        if not role_keywords(getattr(config, "icp", None)):
            return (f"The {spec['label']} preset searches for roles, and sellbird.yaml has "
                    "none: set icp.hiring_signals (or icp.titles).")
    return None


def build_queries(config, cities: list[str] | None = None,
                  depth: int = 30, limit: int = 100, preset: str = "") -> list[DiscoveryQuery]:
    """Turn the configured ICP into a concrete query list.

    One query per (industry or keyword variant x city). Coordinates come from
    ``icp.geo_coordinates`` when present, because listings providers need a
    radius, not a place name. A ``preset`` (PRESETS) rewrites each industry
    into its search; a non-local one only uses cities you pass.
    """
    icp = config.icp
    terms = list(icp.industries) or [config.product.name]
    spec = PRESETS.get(preset) if preset else None
    if spec:
        templates = preset_templates(spec)
        since = (date.today() - timedelta(days=FUNDING_LOOKBACK_DAYS)).isoformat()
        from sellbird.collectors.ats import role_keywords
        roles = role_keywords(icp) if any("{role}" in tpl for tpl in templates) else [""]
        terms = [tpl.format(industry=t, role=r, since=since)
                 for r in roles for t in terms for tpl in templates]
        places = cities if cities is not None else ([""] if not spec["local"] else list(icp.geography) or [""])
    else:
        # "roofing contractor" is one way to search for a roofer. Each
        # configured variant is another search in every city — ranks differ
        # by phrasing, and the email should quote the phrase that was used.
        variants = getattr(icp, "keyword_variants", {}) or {}
        terms = list(dict.fromkeys(
            v.strip() for t in terms for v in [t, *variants.get(t, [])] if v.strip()))
        places = cities if cities is not None else list(icp.geography) or [""]
    coords = getattr(icp, "geo_coordinates", {}) or {}

    return [
        DiscoveryQuery(term=term, location=place, depth=depth, limit=limit,
                       coordinate=coords.get(place, ""), preset=preset if spec else "")
        for term in terms
        for place in places
    ]


def estimate_cost(provider_key: str, queries: list[DiscoveryQuery]) -> float:
    """Projected spend. Always call this before a run — and print it."""
    provider = PROVIDERS.get(provider_key)
    return provider.estimate(queries) if provider else 0.0


async def run_discovery(
    state,
    config,
    provider_key: str = DEFAULT_PROVIDER,
    queries: list[DiscoveryQuery] | None = None,
    max_spend: float = 1.0,
    dry_run: bool = False,
) -> DiscoveryReport:
    """Discover businesses and record what was learned about them.

    ``dry_run`` estimates and returns without calling anything. Run it first,
    every time: the first small trial is what exposes the data-quality
    problems that would otherwise poison every market you scale to.
    """
    from sellbird.config import live_do_not_contact, load_env
    from sellbird.gate import do_not_contact_domain_match
    from sellbird.signals import seed_signal_catalog

    provider = PROVIDERS.get(provider_key)
    if provider is None:
        raise ValueError(f"unknown discovery provider: {provider_key!r}")

    queries = queries or build_queries(config)
    if provider.needs_coordinates:
        queries = await resolve_coordinates(state, queries)
    report = DiscoveryReport(provider=provider_key, queries=len(queries))
    report.estimated_cost = provider.estimate(queries)

    if dry_run:
        return report

    env = load_env().model_dump()
    missing = provider.missing_keys(env)
    if missing:
        names = ", ".join(k.upper() for k in missing)
        report.errors.append(
            f"{provider.label} is not ready: {names} is empty or still the "
            f"template placeholder in .env")
        report.stopped = "not configured"
        return report

    if report.estimated_cost > max_spend:
        report.errors.append(
            f"estimated ${report.estimated_cost:.2f} exceeds the ${max_spend:.2f} "
            f"cap — raise the cap or run fewer queries")
        report.stopped = "over budget"
        return report

    await seed_signal_catalog(state)
    confirmed = await state.confirmed_signal_codes()
    if not confirmed:
        report.errors.append(
            "no signals confirmed — confirm signals first (dashboard → Signals, "
            "or `sellbird signals --confirm free`) or discovery has nothing to record")
        report.stopped = "no confirmed signals"
        return report

    report.run_id = await state.start_run(
        "discover", provider=provider_key,
        params={"queries": len(queries), "estimate": report.estimated_cost},
    )

    # Resolved once per run, not per result: the list may live in a file, and
    # re-reading it a few hundred times would turn a config lookup into I/O
    # inside the hot loop.
    do_not_contact = live_do_not_contact(config)

    # Each business's base facts are written once per run; its ranks once per
    # (business, keyword), and again only if a repeat search ranks it better.
    seen: set[str] = set()
    best_ranks: dict[tuple[str, str], tuple] = {}
    company_ids: dict[str, str] = {}
    ranked_in_run: dict[tuple[str, str], set[str]] = {}
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as client:
            for i, query in enumerate(queries):
                # The kill switch is read between batches, not once at the
                # top: a long run has to be stoppable while it is running.
                if await state.get_setting("discovery_paused"):
                    report.stopped = "paused"
                    break
                if report.actual_cost > max_spend:
                    report.stopped = "budget reached"
                    break
                if i and provider.key == "osm":
                    await asyncio.sleep(OpenStreetMap.MIN_INTERVAL)

                try:
                    report.queries_ran += 1
                    provider.last_cost = 0.0
                    provider.last_error = provider.stop_reason = ""
                    provider.complete = True
                    provider.remaining_budget = max(0.0, max_spend - report.actual_cost)
                    found = await provider.fetch(client, env, query)
                    report.queries_succeeded += 1
                except ProviderExhausted as e:
                    logger.warning("discover: %s stopped: %s", provider_key, e)
                    report.stopped = str(e)
                    break
                except Exception as e:
                    # A failure is an observation about the run, not a reason
                    # to lose the queries that already succeeded.
                    logger.warning("discover: %s failed on %r: %s",
                                   provider_key, query.keyword(), e)
                    report.errors.append(f"{query.keyword()}: {e}")
                    continue
                finally:
                    report.actual_cost += provider.last_cost

                if provider.last_error:
                    report.errors.append(f"{query.keyword()}: {provider.last_error}")
                report.found += len(found)
                batch = []
                for biz in found:
                    if query.preset == "hiring" and biz.domain and not is_junk(biz.domain):
                        # Judged on the host as found first: jobs.lever.co
                        # stripped would be lever.co, a vendor, not junk.
                        biz.domain = company_host(biz.domain)
                    if is_junk(biz.domain, biz.raw_title or biz.name):
                        report.junk += 1
                        continue
                    # Before the upsert, so a forbidden company never becomes a
                    # row and never reaches profiling. Checking it at send time
                    # alone still lets one through discovery, where it costs
                    # credits to enrich and sits in every cohort until somebody
                    # notices it by eye.
                    if do_not_contact_domain_match(biz.domain, do_not_contact):
                        report.excluded += 1
                        continue
                    ident = biz.identity()
                    if not ident:
                        continue
                    first = ident not in seen
                    if first:
                        seen.add(ident)
                        company_id, is_new = await _upsert(state, biz, query)
                        if is_new:
                            report.new_companies += 1
                        else:
                            report.known_companies += 1
                        company_ids[ident] = company_id
                    company_id = company_ids[ident]
                    phrase = biz.keyword or query.keyword()
                    for code, value in zip(RANK_CODES, _rank_tuple(biz)):
                        if value is not None:
                            ranked_in_run.setdefault((code, phrase), set()).add(company_id)

                    rank_key = (ident, phrase.lower())
                    ranks = _rank_tuple(biz)
                    previous = best_ranks.get(rank_key)
                    improved = previous is None or any(
                        new is not None and (old is None or new < old)
                        for new, old in zip(ranks, previous))
                    if not first and not improved:
                        continue
                    if previous is not None:
                        # A repeat of the same search: record only the ranks
                        # it improved, so a worse one never becomes "latest".
                        biz = replace(biz, **{
                            name: (new if new is not None and (old is None or new < old)
                                   else None)
                            for name, new, old in zip(RANK_FIELDS, ranks, previous)})
                    if improved:
                        best_ranks[rank_key] = tuple(
                            _best(old, new) for new, old in
                            zip(ranks, previous or (None,) * len(ranks)))
                    batch.extend(_observations(biz, company_id, confirmed,
                                               report.run_id, query=query,
                                               ranks_only=not first))

                # A business this phrase ranked before, and that no search for
                # it this run has found, is no longer ranked (within depth) or
                # no longer in the pack. Say so, so the latest reading is true.
                phrase = query.keyword()
                for code in provider.rank_codes if provider.complete else ():
                    if code not in confirmed:
                        continue
                    found_now = ranked_in_run.setdefault((code, phrase), set())
                    for gone in await state.ranked_company_ids(code, phrase) - found_now:
                        batch.append({
                            "company_id": gone, "signal_code": code,
                            "collector": "discover", "value_num": 0.0,
                            "value_text": phrase, "confidence": 1.0,
                            "evidence_url": "", "run_id": report.run_id,
                        })

                # Flush every batch. The observations are the asset; buffering
                # them until some threshold means a run that dies loses them.
                if batch:
                    report.observations += await state.add_observations(
                        batch, run_id=report.run_id)
                if provider.stop_reason:
                    report.stopped = provider.stop_reason
                    break

        await state.finish_run(report.run_id, status="completed",
                               records=report.new_companies + report.known_companies,
                               cost_usd=report.actual_cost)
    except Exception as e:
        logger.exception("discover run failed")
        report.errors.append(str(e))
        await state.finish_run(report.run_id, status="failed",
                               records=report.new_companies,
                               cost_usd=report.actual_cost, error=str(e))
    return report


async def _upsert(state, biz: Business, query: DiscoveryQuery) -> tuple[str, bool]:
    """Record the business, returning (id, was_new)."""
    # Look before inserting, so "already knew this one" is a fact we report
    # rather than something inferred from an insert's return value.
    existing = None
    if biz.domain:
        existing = await state.get_company_by_domain(biz.domain)
    for external in (biz.external_id, *biz.alt_ids):
        if existing is None and external:
            existing = await state.get_company_by_external_id(external)
    if existing:
        return existing.id, False

    company = Company(
        name=biz.name,
        domain=biz.domain,
        website=biz.website,
        phone=biz.phone,
        industry=biz.category or query.term,
        location=biz.location or query.location,
        source=f"discover:{biz.provider}",
        source_url=biz.source_url,
        external_id=biz.external_id,
    )
    return await state.add_company(company), True


RANK_FIELDS = ("rank", "local_pack_rank", "maps_rank")
RANK_CODES = ("SERP_RANK", "IN_LOCAL_PACK", "MAPS_RANK")      # same order


def _rank_tuple(biz: Business) -> tuple:
    return tuple(getattr(biz, name) for name in RANK_FIELDS)


def _observations(biz: Business, company_id: str, confirmed: set[str],
                  run_id: str, query: DiscoveryQuery | None = None,
                  ranks_only: bool = False) -> list[dict]:
    """Every fact this business gave us, as rows — and only the confirmed ones.

    Rank rows carry the search phrase in ``value_text``: "#14 for 'roofer
    Denver, CO'" is a fact; "#14" on its own isn't. ``ranks_only`` is for a
    business already recorded this run, found again by another search.
    """
    rows: list[dict] = []

    def add(code, *, num=None, text="", conf=1.0):
        if code in confirmed:
            rows.append({
                "company_id": company_id, "signal_code": code,
                "collector": "discover", "value_num": num, "value_text": text,
                "confidence": conf, "evidence_url": biz.source_url,
                "run_id": run_id,
            })

    keyword = biz.keyword or (query.keyword() if query is not None else "")
    if biz.rank is not None:
        add("SERP_RANK", num=float(biz.rank), text=keyword)
    if biz.local_pack_rank is not None:
        add("IN_LOCAL_PACK", num=float(biz.local_pack_rank), text=keyword)
    if biz.maps_rank is not None:
        add("MAPS_RANK", num=float(biz.maps_rank), text=keyword)
    if ranks_only:
        return rows

    add("FOUND_IN_SERP", text=biz.provider)
    if biz.has_website is False:
        add("NO_WEBSITE", num=1.0)
    if biz.is_claimed is False:
        add("UNCLAIMED_LISTING", num=1.0)
    if biz.rating is not None:
        add("REVIEW_RATING", num=float(biz.rating))
    if biz.review_count is not None:
        add("REVIEW_COUNT", num=float(biz.review_count))
    if query is not None and query.preset == "yc":
        # The search matched "backed by Y Combinator" on a page of their own
        # site (third-party sites are filtered out above). Short of reading
        # that page ourselves, so below the homepage's own claim.
        add("YC_BACKED", num=1.0, conf=0.7)
    if query is not None and query.preset == "accelerator":
        # Same, for the other programs; which one isn't known until the
        # profile reads their pages. A funding hit isn't recorded: the
        # raise's date matters, so the profile reads the hit page itself.
        add("ACCELERATOR", num=1.0, conf=0.6)
    return rows


def provider_menu(env: dict | None = None) -> list[dict]:
    """The menu the dashboard renders: what each provider does, and its cost.

    The CLI calls this bare; without a load, every keyed provider would show
    "needs setup" even with a working key in .env.
    """
    if env is None:
        from sellbird.config import load_env
        env = load_env().model_dump()
    return [
        {
            "key": p.key, "label": p.label, "kind": p.kind, "blurb": p.blurb,
            "cost_note": p.cost_note, "free_tier": p.free_tier,
            "signup_url": p.signup_url, "caveat": p.caveat,
            "env_keys": [k.upper() for k in p.env_keys],
            "configured": p.configured(env),
            "needs_key": bool(p.env_keys),
        }
        for p in PROVIDERS.values()
    ]

