"""FastAPI routes for local sales agent workflows and SQLite store.

Mounted under prefix /api/local.
Provides offline AI workflows, deterministic fallback handling, task queue processing,
and full dashboard inspection endpoints.
"""

import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from sellbird.local_ai import hardware, network_guard
from sellbird.local_ai.ollama_client import DEFAULT_RECOMMENDED_MODEL, OllamaClient
from sellbird.local_ai.scraper import OfflineModeError, scrape_public_url
from sellbird.local_ai.store import DEFAULT_DB_PATH, LocalSalesStore
from sellbird.local_ai.workflows import OllamaWorkflowError, WorkflowEngine

logger = logging.getLogger("sellbird.local_ai.routes")

router = APIRouter(prefix="/api/local", tags=["Local AI"])

# Global shared store and engine instances (default to data/local_sales_agent.db)
_store = LocalSalesStore(DEFAULT_DB_PATH)
_engine = WorkflowEngine(store=_store)


def get_engine() -> WorkflowEngine:
    return _engine


def set_engine(engine: WorkflowEngine):
    global _engine, _store
    _engine = engine
    _store = engine.store


# =============================================================================
# Request Schemas
# =============================================================================

class ScrapeProspectRequest(BaseModel):
    url: str
    company_name: str | None = None
    contact_name: str | None = None
    contact_role: str | None = None
    contact_email: str | None = None


class OptionalProductRequest(BaseModel):
    product_id: int | None = None


class ReplyClassifyRequest(BaseModel):
    reply_text: str
    prospect_id: int | None = None


class SystemConfigRequest(BaseModel):
    active_model: str | None = None
    strict_offline_mode: bool | None = None


# =============================================================================
# Exception Helper
# =============================================================================

def handle_workflow_error(exc: Exception):
    if isinstance(exc, OllamaWorkflowError):
        return JSONResponse(
            status_code=503,
            content={
                "ollama_offline": True,
                "message": exc.message,
                "queued_task_id": exc.queued_task_id,
                "deterministic_fallback": exc.deterministic_fallback,
            },
        )
    if isinstance(exc, OfflineModeError):
        return JSONResponse(
            status_code=403,
            content={"error": "offline_mode_error", "message": str(exc)},
        )
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=404, detail=str(exc))
    logger.exception("Unexpected error in workflow route")
    raise HTTPException(status_code=500, detail=str(exc))


# =============================================================================
# 1. AI Workflows
# =============================================================================

@router.post("/companies/{company_id}/profile")
async def generate_company_profile_endpoint(company_id: int):
    """Workflow 1: Generate structured business profile from company notes."""
    try:
        return await _engine.generate_company_profile(company_id)
    except Exception as exc:
        return handle_workflow_error(exc)


@router.post("/products/{product_id}/icp")
async def generate_product_icp_endpoint(product_id: int):
    """Workflow 2: Generate product ICP and store local embedding."""
    try:
        return await _engine.generate_product_icp(product_id)
    except Exception as exc:
        return handle_workflow_error(exc)


@router.post("/prospects/{prospect_id}/research")
async def research_prospect_endpoint(prospect_id: int):
    """Workflow 3: Analyze prospect raw_text strictly offline from SQLite."""
    try:
        return await _engine.research_prospect(prospect_id)
    except Exception as exc:
        return handle_workflow_error(exc)


@router.post("/prospects/scrape")
async def scrape_prospect_endpoint(req: ScrapeProspectRequest):
    """Online web scraper for prospect research (disabled in STRICT_OFFLINE_MODE)."""
    try:
        scraped = await scrape_public_url(req.url)
        company_name = req.company_name or scraped.get("title") or "Unknown Company"
        prospect_id = await _store.create_prospect(
            company_name=company_name,
            contact_name=req.contact_name or "",
            contact_role=req.contact_role or "",
            contact_email=req.contact_email or "",
            source_url=req.url,
            raw_text=scraped["clean_text"],
            research_json={},
            scraped_at=scraped["scraped_at"],
        )
        return {
            "status": "success",
            "prospect_id": prospect_id,
            "company_name": company_name,
            "source_url": req.url,
            "text_length": len(scraped["clean_text"]),
        }
    except Exception as exc:
        return handle_workflow_error(exc)


@router.post("/prospects/{prospect_id}/product-fit")
async def calculate_product_fit_endpoint(prospect_id: int):
    """Workflow 4 (Differentiator 1): Multi-product cosine fit matrix using local embeddings."""
    try:
        return await _engine.calculate_product_fit(prospect_id)
    except Exception as exc:
        return handle_workflow_error(exc)


@router.post("/prospects/{prospect_id}/signals")
async def classify_signals_endpoint(prospect_id: int):
    """Workflow 5 (Differentiator 2): Classify buying signals with deterministic evidence grounding."""
    try:
        return await _engine.classify_buying_signals(prospect_id)
    except Exception as exc:
        return handle_workflow_error(exc)


@router.post("/prospects/{prospect_id}/problem-statement")
async def generate_problem_statement_endpoint(
    prospect_id: int, req: OptionalProductRequest = OptionalProductRequest()
):
    """Workflow 6: Generate tailored problem statement from grounded signals."""
    try:
        return await _engine.generate_problem_statement(prospect_id, req.product_id)
    except Exception as exc:
        return handle_workflow_error(exc)


@router.post("/prospects/{prospect_id}/score")
async def score_prospect_endpoint(
    prospect_id: int, req: OptionalProductRequest = OptionalProductRequest()
):
    """Workflow 7 (Differentiator 3): Hybrid prospect scoring (50% deterministic + 50% LLM)."""
    try:
        return await _engine.score_prospect(prospect_id, req.product_id)
    except Exception as exc:
        return handle_workflow_error(exc)


@router.post("/prospects/{prospect_id}/email")
async def generate_outreach_email_endpoint(
    prospect_id: int, req: OptionalProductRequest = OptionalProductRequest()
):
    """Workflow 8: Generate cold outreach email (draft_only, zero external SMTP)."""
    try:
        return await _engine.generate_outreach_email(prospect_id, req.product_id)
    except Exception as exc:
        return handle_workflow_error(exc)


@router.post("/replies/classify")
async def classify_reply_endpoint(req: ReplyClassifyRequest):
    """Workflow 9: Classify prospect reply email intent and next action."""
    try:
        return await _engine.classify_reply(req.prospect_id, req.reply_text)
    except Exception as exc:
        return handle_workflow_error(exc)


# =============================================================================
# 2. System Status & Config Endpoints
# =============================================================================

@router.get("/system/status")
async def get_system_status():
    """Retrieve hardware profile, Ollama connectivity, model status, and queue depth."""
    hw = hardware.get_hardware_profile().model_dump()

    installed_models = []
    ollama_online = False
    try:
        installed_models = await _engine.ollama.health_check()
        ollama_online = True
        if installed_models and _engine.model not in installed_models:
            _engine.model = installed_models[0]
    except Exception:
        pass

    conn = await _store.connect()
    async with conn.execute(
        "SELECT count(*) as count FROM task_queue WHERE status = 'pending'"
    ) as cur:
        row = await cur.fetchone()
        pending_count = row["count"] if row else 0

    return {
        "hardware_profile": hw,
        "active_model": _engine.model,
        "installed_models": installed_models,
        "ollama_online": ollama_online,
        "strict_offline_mode": network_guard.STRICT_OFFLINE_MODE,
        "pending_queue_count": pending_count,
    }


@router.post("/system/config")
async def update_system_config(cfg: SystemConfigRequest):
    """Update active Ollama model or toggle strict offline mode."""
    if cfg.active_model:
        _engine.model = cfg.active_model
    if cfg.strict_offline_mode is not None:
        network_guard.STRICT_OFFLINE_MODE = cfg.strict_offline_mode

    return {
        "active_model": _engine.model,
        "strict_offline_mode": network_guard.STRICT_OFFLINE_MODE,
    }


# =============================================================================
# 3. Task Queue Endpoints
# =============================================================================

@router.get("/queue")
async def list_tasks_endpoint(limit: int = 50):
    """List pending and recently completed tasks in task_queue."""
    conn = await _store.connect()
    async with conn.execute(
        "SELECT * FROM task_queue ORDER BY id DESC LIMIT ?", (limit,)
    ) as cur:
        rows = await cur.fetchall()
        return [dict(r) for r in rows]


@router.post("/queue/process")
async def process_queue_endpoint():
    """Drain pending tasks in task_queue using Ollama."""
    return await _engine.process_queue()


# =============================================================================
# 4. Inference Logs
# =============================================================================

@router.get("/logs/inference")
async def list_inference_logs_endpoint(limit: int = 100):
    """Retrieve recent inference audit logs from SQLite."""
    return await _store.list_inference_logs(limit=limit)


# =============================================================================
# 5. Dashboard Data Inspection Endpoints
# =============================================================================

@router.get("/companies")
async def list_companies_endpoint():
    return await _store.list_companies()


@router.get("/companies/{company_id}")
async def get_company_endpoint(company_id: int):
    c = await _store.get_company(company_id)
    if not c:
        raise HTTPException(status_code=404, detail="Company not found")
    return c


@router.get("/products")
async def list_products_endpoint(company_id: int | None = None):
    return await _store.list_products(company_id=company_id)


@router.get("/products/{product_id}")
async def get_product_endpoint(product_id: int):
    p = await _store.get_product(product_id)
    if not p:
        raise HTTPException(status_code=404, detail="Product not found")
    return p


@router.get("/icps")
async def list_icps_endpoint():
    return await _store.list_icps()


@router.get("/prospects")
async def list_prospects_endpoint():
    prospects = await _store.list_prospects()
    enriched = []
    for p in prospects:
        fit = await _store.get_product_fit(p["id"])
        scores = await _store.get_prospect_scores(p["id"])
        enriched.append({
            **p,
            "product_fit_matrix": fit,
            "scores": scores,
        })
    return enriched


@router.get("/prospects/{prospect_id}")
async def get_prospect_endpoint(prospect_id: int):
    p = await _store.get_prospect(prospect_id)
    if not p:
        raise HTTPException(status_code=404, detail="Prospect not found")

    # Enrich with related data for the UI
    fit_matrix = await _store.get_product_fit(prospect_id)
    signals = await _store.list_buying_signals(prospect_id)
    problems = await _store.list_problem_statements(prospect_id)
    scores = await _store.get_prospect_scores(prospect_id)
    emails = await _store.list_outreach_emails(prospect_id=prospect_id)

    return {
        **p,
        "product_fit_matrix": fit_matrix,
        "buying_signals": signals,
        "problem_statements": problems,
        "scores": scores,
        "outreach_emails": emails,
    }


@router.get("/replies")
async def list_replies_endpoint():
    return await _store.list_reply_classifications()
