"""One-command launcher for Harvey / Sellbird Local Multi-Product AI Sales Agent.

Starts the 100% offline local dashboard at http://127.0.0.1:8000/local
"""

import asyncio
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["LOCAL_AGENT_MODE"] = "1"

import uvicorn
from sellbird.local_ai.hardware import get_hardware_profile
from sellbird.local_ai.store import DEFAULT_DB_PATH, LocalSalesStore
from scripts.seed_demo_data import seed_demo_data


async def _preflight():
    """Ensure database exists with demo data and log system status."""
    db_file = PROJECT_ROOT / DEFAULT_DB_PATH
    if not db_file.exists():
        print("[*] Initializing local sales database and seeding demo data...")
        await seed_demo_data(db_file)
    else:
        store = LocalSalesStore(db_file)
        await store.init_db()
        await store.close()

    hw = get_hardware_profile()
    print("=" * 65)
    print(" HARVEY / SELLBIRD LOCAL AI SALES AGENT (100% PRIVACY-FIRST)")
    print("=" * 65)
    print(f" Hardware Tier:     {hw.tier.upper()} ({hw.rationale})")
    print(f" CPU Cores:         {hw.cpu_cores}")
    print(f" System RAM:        {hw.ram_gb:.1f} GB")
    if hw.gpu_name:
        print(f" Dedicated GPU:     {hw.gpu_name} ({hw.vram_gb:.1f} GB VRAM)")
    else:
        print(" Dedicated GPU:     None (CPU-Only Mode)")
    print(f" Recommended Model: {hw.recommended_model}")
    print("-" * 65)
    print(" Native Ollama URL: http://127.0.0.1:11434")
    print(" Local Dashboard:   http://127.0.0.1:8000/local")
    print("=" * 65)


def main():
    asyncio.run(_preflight())
    uvicorn.run(
        "sellbird.dashboard:app",
        host="127.0.0.1",
        port=8000,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
