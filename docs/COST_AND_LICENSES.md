# Cost & Licences — Harvey AI Sales Agent (Local)

## 1. Zero-Cost & Zero-External-API Compliance Statement

Harvey AI Sales Agent (Local) operates at **\$0.00 marginal cost per inference**, **\$0.00 SaaS subscription cost**, and **zero external network dependencies**.

### Compliance Proof & Enforcement Mechanisms

1. **Static Analysis & Codebase Scrubbing (`git grep`):**
   - Zero occurrences of OpenAI, Anthropic, Gemini, Serper, Apollo, Instantly, or paid enrichment SDKs/APIs exist in the codebase.
   - All legacy SaaS warmup and paid integrations were completely removed (`sellbird/integrations/instantly.py` deleted).
2. **Runtime Socket Enforcement (`StrictOfflineSandbox`):**
   - Runtime monkey-patching of `socket.create_connection` and `socket.socket.connect` raises `ExternalNetworkBlockedError` whenever any host outside `{"127.0.0.1", "localhost", "::1"}` is contacted.
3. **Audit Trail Verification:**
   - Every inference request is recorded locally in both [logs/local_inference.jsonl](file:///c:/Users/malla/OneDrive/Desktop/projects/Harvey-AI-Sales-Agent-Local/logs/local_inference.jsonl) and SQLite `inference_logs` table with `endpoint_url: "http://127.0.0.1:11434/api/chat"`.
4. **Acceptance Report:**
   - The automated offline test script generates [docs/offline_acceptance_report.json](file:///c:/Users/malla/OneDrive/Desktop/projects/Harvey-AI-Sales-Agent-Local/docs/offline_acceptance_report.json) verifying that `external_network_calls` is strictly `0`.

---

## 2. Software & Model Licences

All models, frameworks, and dependencies used in Harvey AI Sales Agent Local are permissive open-source or copyleft-compliant for local deployment.

| Component / Dependency | Category | Licence | Notes |
| :--- | :--- | :--- | :--- |
| **Sellbird (Upstream Fork)** | Core Platform | **AGPL-3.0-only** | Forked at v0.4.0 (re-licensed to AGPL-3.0-only; v0.3.x was MIT). Code changes are distributed under AGPL-3.0. |
| **Ollama** | Local LLM Server | **MIT** | Runs locally on `http://127.0.0.1:11434`. |
| **Qwen3 (1.7B / 4B / 8B / 14B)** | Local LLM Weights | **Apache 2.0** | Permissive open weights by Alibaba Cloud. |
| **Qwen2.5 (1.5B / 3B / 7B)** | Local LLM Weights | **Apache 2.0** | Permissive open weights by Alibaba Cloud. |
| **sentence-transformers** | Embedding Library | **Apache 2.0** | Local dense vector extraction. |
| **all-MiniLM-L6-v2** | Local Embedding Model | **Apache 2.0** | Cached locally in `./local_models/all-MiniLM-L6-v2`. Run with `HF_HUB_OFFLINE=1`. |
| **FastAPI** | Web Framework | **MIT** | Dashboard API and route handlers. |
| **Uvicorn** | ASGI Server | **BSD-3-Clause** | Fast ASGI web server. |
| **Pydantic** | Data Validation | **MIT** | Strict JSON schema parsing and structured output validation. |
| **aiosqlite** | Async Database Driver | **MIT** | Async SQLite connection pool. |
| **SQLite** | Embedded Database | **Public Domain** | Zero-server local storage engine (`data/local_sales_agent.db`). |
| **httpx** | Async HTTP Client | **BSD-3-Clause** | Local HTTP client for Ollama API and public web scraper. |
| **BeautifulSoup4** | HTML Parser | **MIT** | DOM sanitization and boilerplate removal. |
| **Playwright** | Browser Automation | **Apache 2.0** | Headless Chromium scraper fallback for JS-rendered pages. |
| **psutil** | System Profiler | **BSD-3-Clause** | Cross-platform RAM and CPU core detection. |
| **Geist & Geist Mono** | Typography | **SIL Open Font License 1.1** | Vendored locally in `sellbird/web/fonts/`. |

---

## 3. Financial Summary

| Item | Cloud SaaS Baseline (HubSpot / Apollo / Instantly / OpenAI) | Harvey AI Local |
| :--- | :--- | :--- |
| **LLM Inference** | \$0.02 – \$0.06 per prospect | **\$0.00** (Local Ollama) |
| **Embedding Generation** | \$0.0001 per 1k tokens (OpenAI ada-002) | **\$0.00** (Local all-MiniLM-L6-v2) |
| **Sales Engagement SaaS** | \$97 – \$297 / month | **\$0.00** (Self-hosted SQLite & FastAPI) |
| **Data Enrichment APIs** | \$0.20 – \$0.50 per verified lead | **\$0.00** (Local Web Scraper & Seed Data) |
| **Total Monthly Cost** | **\$300 – \$1,000+ / mo** | **\$0.00 / month forever** |
