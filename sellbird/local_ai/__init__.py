"""Local AI subsystem for Harvey/Sellbird.

Strictly offline and local inference via Ollama and local sentence-transformers.
"""

from sellbird.local_ai.hardware import HardwareProfile, get_hardware_profile
from sellbird.local_ai.network_guard import ExternalNetworkBlockedError, StrictOfflineSandbox
from sellbird.local_ai.ollama_client import OllamaClient, OllamaUnavailableError
from sellbird.local_ai.embeddings import embed_text, cosine_similarity

__all__ = [
    "HardwareProfile",
    "get_hardware_profile",
    "ExternalNetworkBlockedError",
    "StrictOfflineSandbox",
    "OllamaClient",
    "OllamaUnavailableError",
    "embed_text",
    "cosine_similarity",
]
