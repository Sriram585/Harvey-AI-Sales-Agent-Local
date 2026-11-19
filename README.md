# Harvey AI Sales Agent (Local) — Multi-Product Autonomous Sales Engine

> **100% Local. Zero Cloud Inference Cost (\$0.00). Strict Offline Air-Gap Compliance.**

Harvey AI Sales Agent (Local) is an open-source, private sales operations platform extending Sellbird into an air-gapped, multi-product autonomous agent. Built for engineering and operations platforms, it eliminates external AI APIs (OpenAI, Anthropic, Gemini, Serper) and paid SaaS subscriptions (Apollo, Instantly, Lemlist), running entirely on local consumer hardware via **Ollama** and **sentence-transformers**.

---

## Key Highlights & Core Differentiators

1. **Multi-Product Fit Matrix:**
   Instead of a single product pitch, evaluates each prospect against the entire product catalog (`CloudCost Guard`, `PipeGuard`, `ShiftSync`) using local `sentence-transformers/all-MiniLM-L6-v2` dense embeddings, automatically computing primary fit and secondary cross-sell candidates.
2. **Anti-Hallucination Evidence Grounding:**
   Extracts buying signals with verbatim evidence quotes from public text, verified via deterministic Python substring and sliding-window `difflib` matching (`verify_quote_grounding`). Prevents hallucinated claims from entering problem statements and emails.
3. **Resilient Failure Handling & Durable Task Queue:**
   When Ollama is offline or restarting, zero data leaves the machine. Executes deterministic keyword and embedding fallbacks, enqueues the pending AI job into SQLite `task_queue`, returns HTTP 503, and drains automatically upon Ollama recovery.
4. **Adaptive Hardware Profiling:**
   Detects CPU cores, system RAM, and GPU VRAM via `psutil` and `torch.cuda`/`nvidia-smi`, automatically recommending the optimal local model (`standard_laptop` for CPU laptops vs `high_performance` for workstations).
5. **100% Offline-Renderable Monochrome UI:**
   Pure black-and-white minimalist dashboard served at `http://127.0.0.1:8000/local` with vendored Geist fonts and zero external CDN or Google Fonts calls.

---

## System Architecture

```
[ Public Web URL ]
        │ (Optional live scrape via httpx + BeautifulSoup4 + Playwright)
        ▼
[ SQLite Database: data/local_sales_agent.db (12 Tables) ]
        ▲
        │  (StrictOfflineSandbox: Enforces loopback 127.0.0.1 only)
        ▼
[ Local AI Engine ]
  ├── Hardware Profiler (psutil + torch.cuda / nvidia-smi)
  ├── Local Vector Embeddings (all-MiniLM-L6-v2 @ local_models/)
  ├── Native Ollama Client (http://127.0.0.1:11434, keep_alive: 30m)
  ├── 8 Core Workflows + Reply Triage Sandbox
  └── Durable Task Queue (SQLite task_queue)
        ▲
        ▼
[ Local Web Dashboard: http://127.0.0.1:8000/local ]
```

---

## Quickstart & Reproducible Setup

### Prerequisites
- **Python 3.11 or 3.12**
- **Ollama** installed ([ollama.ai](https://ollama.ai))

### Windows (PowerShell)

```powershell
# 1. Clone repository & create virtual environment
git clone https://github.com/malladi-101/Harvey-AI-Sales-Agent-Local.git
cd Harvey-AI-Sales-Agent-Local
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Pull recommended local model in Ollama
ollama pull qwen2.5:1.5b
# or: ollama pull qwen3:4b

# 4. Cache local embedding model (sentence-transformers/all-MiniLM-L6-v2)
python scripts/setup_models.py

# 5. Populate SQLite with rich seed data (Vanguard Ops AI + 3 Products + 6 Prospects)
python scripts/seed_demo_data.py

# 6. Launch the Local Sales Agent Dashboard
python scripts/run_local_agent.py
```

### macOS / Linux (Bash)

```bash
# 1. Clone repository & create virtual environment
git clone https://github.com/malladi-101/Harvey-AI-Sales-Agent-Local.git
cd Harvey-AI-Sales-Agent-Local
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Pull recommended local model in Ollama
ollama pull qwen2.5:1.5b
# or: ollama pull qwen3:4b

# 4. Cache local embedding model
python3 scripts/setup_models.py

# 5. Seed demo data
python3 scripts/seed_demo_data.py

# 6. Launch dashboard
python3 scripts/run_local_agent.py
```

Open your browser to: **`http://127.0.0.1:8000/local`**

---

## Automated Offline Acceptance Verification

You can verify full offline compliance even with your computer's Wi-Fi / Ethernet completely disconnected:

```bash
python scripts/verify_offline.py
```

This automated suite:
1. Runs inside `StrictOfflineSandbox()` (raising `ExternalNetworkBlockedError` if any non-loopback socket connection is attempted).
2. Executes all 8 sales workflows and differentiators end-to-end.
3. Simulates an Ollama outage, verifies deterministic fallbacks and task queueing.
4. Restores Ollama and drains the queue.
5. Writes the audit report to `docs/offline_acceptance_report.json` with `"external_network_calls": 0`.

---

## Running the Automated Test Suite

Run all 24 unit and integration tests across local AI, SQLite storage, workflows, and the UI:

```bash
pytest tests/test_local_ai.py tests/test_local_store.py tests/test_workflows.py tests/test_local_ui.py -v
```

---

## Known Limitations & Simulated/Seeded Components

To maintain strict compliance with privacy, zero-cost, and non-spam standards:
1. **Outreach Emails Are Draft-Only:**
   Generated cold emails are stored in SQLite with `status="draft_only"`. No external SMTP warmup engine or automated bulk delivery is included, eliminating spam and API delivery costs.
2. **Seeded Prospects & Responses:**
   6 realistic enterprise prospects (`ApexFin Systems`, `NovaPay Technologies`, etc.) and 6 sample prospect replies are pre-seeded in SQLite to enable instant, zero-leak offline testing without needing live internet access.
3. **Web Scraper Boundary:**
   Live public URL scraping (`sellbird/local_ai/scraper.py`) requires online access. When `STRICT_OFFLINE_MODE` is enabled, web scraping raises `OfflineModeError` to ensure zero unexpected internet calls occur during offline operation.

---

## Licence

- Upstream Sellbird core: **AGPL-3.0-only** (as of v0.4.0+).
- Local AI subsystem, workflows, and documentation: Distributed under **AGPL-3.0-only**.
- All bundled fonts (`Geist`, `Geist Mono`): **SIL Open Font License 1.1**.
- Local embedding model (`all-MiniLM-L6-v2`): **Apache 2.0**.
