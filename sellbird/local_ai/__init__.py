"""Local AI subsystem for Harvey/Sellbird.

Strictly offline and local inference via Ollama and local sentence-transformers.
"""

from sellbird.local_ai.hardware import HardwareProfile, get_hardware_profile
from sellbird.local_ai.network_guard import ExternalNetworkBlockedError, StrictOfflineSandbox
from sellbird.local_ai.ollama_client import OllamaClient, OllamaUnavailableError
from sellbird.local_ai.embeddings import embed_text, cosine_similarity
from sellbird.local_ai.store import LocalSalesStore
from sellbird.local_ai.scraper import scrape_public_url, clean_html, OfflineModeError

__all__ = [
    "HardwareProfile",
    "get_hardware_profile",
    "ExternalNetworkBlockedError",
    "StrictOfflineSandbox",
    "OllamaClient",
    "OllamaUnavailableError",
    "embed_text",
    "cosine_similarity",
    "LocalSalesStore",
    "scrape_public_url",
    "clean_html",
    "OfflineModeError",
]
