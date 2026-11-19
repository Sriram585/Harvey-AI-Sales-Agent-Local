"""Isolated web scraping utility for prospect research.

Zero external AI/LLM calls. All scraping logic is isolated here.
Uses httpx + BeautifulSoup with a Playwright Chromium fallback for JS-rendered pages.
Runs Playwright sync API inside asyncio.to_thread to maintain Windows event-loop compatibility.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from bs4 import BeautifulSoup
import httpx

import sellbird.local_ai.network_guard as network_guard

logger = logging.getLogger("sellbird.local_ai.scraper")

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

UNWANTED_TAGS = [
    "script",
    "style",
    "nav",
    "header",
    "footer",
    "noscript",
    "svg",
    "aside",
    "form",
    "iframe",
]


class OfflineModeError(RuntimeError):
    """Raised when live web scraping is attempted while in strict offline mode."""
    pass


def clean_html(html_content: str, url: str = "") -> dict[str, Any]:
    """Parse and clean raw HTML, returning title, clean readable text, and timestamp."""
    soup = BeautifulSoup(html_content, "html.parser")

    # Extract page title
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    elif soup.find("h1"):
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)

    # Decompose boilerplate, navigations, and script/style tags
    for tag in soup.find_all(UNWANTED_TAGS):
        tag.decompose()

    # Extract text and collapse excessive whitespace
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    clean_text = "\n".join(chunk for chunk in lines if chunk)

    return {
        "url": url,
        "title": title,
        "clean_text": clean_text,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }


def _sync_playwright_scrape(target_url: str, timeout_sec: float) -> str:
    """Synchronous Playwright scraper executed inside worker thread for Windows loop safety."""
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(user_agent=DEFAULT_USER_AGENT)
            page.goto(
                target_url,
                timeout=int(timeout_sec * 1000),
                wait_until="domcontentloaded",
            )
            content = page.content()
            browser.close()
            return content
    except Exception as exc:
        logger.warning(f"Playwright fallback failed for {target_url}: {exc}")
        return ""


async def scrape_public_url(url: str, timeout: float = 15.0) -> dict[str, Any]:
    """Scrape a public URL cleanly using httpx + BeautifulSoup, with Playwright fallback."""
    if network_guard.STRICT_OFFLINE_MODE:
        raise OfflineModeError(
            "Live web scraping requires online mode while local AI workflows remain available offline."
        )

    headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    httpx_html: str = ""
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                httpx_html = resp.text
                result = clean_html(httpx_html, url)
                # If we got meaningful content (>200 chars), return immediately
                if len(result["clean_text"]) > 200:
                    return result
    except Exception as exc:
        logger.info(f"httpx scrape attempt failed for {url} ({exc}); trying Playwright fallback...")

    # Fallback to Playwright (for JS-rendered SPAs or anti-bot landing pages)
    pw_html = await asyncio.to_thread(_sync_playwright_scrape, url, timeout)
    if pw_html:
        pw_result = clean_html(pw_html, url)
        if pw_result["clean_text"]:
            return pw_result

    # Best-effort return if httpx had partial content
    if httpx_html:
        return clean_html(httpx_html, url)

    raise RuntimeError(f"Unable to scrape content from {url} using both httpx and Playwright.")
