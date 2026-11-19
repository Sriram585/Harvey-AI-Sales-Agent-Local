"""Unit and integration tests for Phase 3: SQLite storage, scraper, and demo seed data."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sellbird.local_ai.network_guard import StrictOfflineSandbox
from sellbird.local_ai.ollama_client import OllamaClient
from sellbird.local_ai.scraper import OfflineModeError, clean_html, scrape_public_url
from sellbird.local_ai.store import LocalSalesStore
from scripts.seed_demo_data import seed_demo_data


EXPECTED_TABLES = {
    "companies",
    "products",
    "icps",
    "prospects",
    "product_fit_matrix",
    "buying_signals",
    "problem_statements",
    "prospect_scores",
    "outreach_emails",
    "reply_classifications",
    "inference_logs",
    "task_queue",
}


@pytest.mark.asyncio
async def test_init_db_creates_all_12_tables(tmp_path):
    """Verify that LocalSalesStore.init_db creates all 12 specified SQLite tables."""
    db_file = tmp_path / "test_store.db"
    store = LocalSalesStore(db_file)
    await store.init_db()

    conn = await store.connect()
    async with conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    ) as cur:
        rows = await cur.fetchall()
        tables = {row["name"] for row in rows}

    await store.close()
    assert EXPECTED_TABLES.issubset(tables), f"Missing tables: {EXPECTED_TABLES - tables}"


@pytest.mark.asyncio
async def test_store_crud_and_inference_logging(tmp_path):
    """Verify clean CRUD helpers and SQLite inference logging for LocalSalesStore."""
    db_file = tmp_path / "test_crud.db"
    store = LocalSalesStore(db_file)
    await store.init_db()

    # 1. Company
    company_id = await store.create_company(
        name="Acme Corp",
        website="https://acme.test",
        raw_notes="Enterprise notes",
        profile_json={"industry": "Security"},
    )
    assert company_id > 0
    company = await store.get_company(company_id)
    assert company is not None
    assert company["name"] == "Acme Corp"

    # 2. Product
    product_id = await store.create_product(
        company_id=company_id,
        name="ShieldAI",
        description="Security product",
        value_prop="Zero-day protection",
        pricing_tier="$1000/mo",
    )
    assert product_id > 0
    products = await store.list_products(company_id=company_id)
    assert len(products) == 1
    assert products[0]["name"] == "ShieldAI"

    # 3. ICP
    icp_id = await store.save_icp(
        product_id=product_id,
        icp_json={"target_size": "50-500"},
        embedding_json=[0.1, 0.2, 0.3],
    )
    assert icp_id > 0
    icp = await store.get_icp(product_id)
    assert icp is not None

    # 4. Prospect
    prospect_id = await store.create_prospect(
        company_name="TargetTech",
        contact_name="Jane Doe",
        contact_role="CTO",
        contact_email="jane@target.test",
        raw_text="TargetTech is expanding cloud infrastructure.",
        research_json={"cloud": "AWS"},
    )
    assert prospect_id > 0
    prospect = await store.get_prospect(prospect_id)
    assert prospect is not None
    assert prospect["company_name"] == "TargetTech"

    # 5. Product Fit Matrix
    fit_id = await store.save_product_fit(
        prospect_id=prospect_id,
        primary_product_id=product_id,
        cross_sell_product_ids_json=[],
        scores_json={"ShieldAI": 0.95},
    )
    assert fit_id > 0
    fit = await store.get_product_fit(prospect_id)
    assert fit is not None

    # 6. Buying Signal
    signal_id = await store.add_buying_signal(
        prospect_id=prospect_id,
        signal_type="cloud_expansion",
        confidence=0.9,
        evidence_quote="expanding cloud infrastructure",
        is_grounded=True,
        reasoning="Explicit mention in raw text",
    )
    assert signal_id > 0
    signals = await store.list_buying_signals(prospect_id)
    assert len(signals) == 1

    # 7. Problem Statement
    prob_id = await store.save_problem_statement(
        prospect_id=prospect_id,
        product_id=product_id,
        statement="Cloud infrastructure security challenges",
        grounding_signals_json=["cloud_expansion"],
    )
    assert prob_id > 0

    # 8. Prospect Score
    score_id = await store.save_prospect_score(
        prospect_id=prospect_id,
        product_id=product_id,
        total_score=88.5,
        priority="high",
        deterministic_score=45.0,
        llm_score=43.5,
        reasoning="Strong intent cues",
    )
    assert score_id > 0

    # 9. Outreach Email
    email_id = await store.save_outreach_email(
        prospect_id=prospect_id,
        product_id=product_id,
        subject="Securing TargetTech's Cloud",
        body="Hi Jane, noticed your cloud expansion...",
        status="draft",
    )
    assert email_id > 0
    await store.update_email_status(email_id, "approved")
    emails = await store.list_outreach_emails(prospect_id=prospect_id, status="approved")
    assert len(emails) == 1

    # 10. Reply Classification
    reply_id = await store.save_reply_classification(
        prospect_id=prospect_id,
        reply_text="Sounds interesting, send more info.",
        intent="interested",
        confidence=0.95,
        next_action="send_case_study",
        reasoning="Expressed explicit interest",
    )
    assert reply_id > 0

    # 11. Inference Logs (Dict callback from OllamaClient)
    log_id = await store.log_inference(
        {
            "timestamp": "2026-11-19T12:00:00Z",
            "workflow_name": "test_workflow",
            "endpoint_url": "http://127.0.0.1:11434/api/chat",
            "model_name": "qwen3:4b",
            "latency_ms": 250.5,
            "eval_count": 55,
            "tokens_per_sec": 38.2,
            "json_repaired": False,
            "status": "success",
        }
    )
    assert log_id > 0
    logs = await store.list_inference_logs(limit=10)
    assert len(logs) == 1
    assert logs[0]["workflow_name"] == "test_workflow"

    # 12. Task Queue
    task_id = await store.enqueue_task(
        task_type="score_prospect",
        payload_json={"prospect_id": prospect_id},
    )
    assert task_id > 0
    next_task = await store.get_next_task()
    assert next_task is not None
    assert next_task["status"] == "running"
    await store.complete_task(next_task["id"], status="completed")

    await store.close()


@pytest.mark.asyncio
async def test_seeded_demo_data_readable(tmp_path):
    """Verify seed_demo_data correctly populates 1 company, 3 products, 6 prospects, and 6 replies."""
    db_file = tmp_path / "seeded_demo.db"
    await seed_demo_data(db_path=db_file)

    store = LocalSalesStore(db_file)
    companies = await store.list_companies()
    assert len(companies) == 1
    assert companies[0]["name"] == "Vanguard Ops AI"

    products = await store.list_products(company_id=companies[0]["id"])
    assert len(products) == 3
    product_names = {p["name"] for p in products}
    assert product_names == {"CloudCost Guard", "PipeGuard", "ShiftSync"}

    prospects = await store.list_prospects()
    assert len(prospects) == 6
    for p in prospects:
        assert len(p["raw_text"]) > 50
        assert p["contact_name"]
        assert "@" in p["contact_email"]

    replies = await store.list_reply_classifications()
    assert len(replies) == 6
    intents = {r["intent"] for r in replies}
    expected_intents = {"interested", "objection", "not_now", "unsubscribe", "out_of_office", "referral"}
    assert intents == expected_intents

    await store.close()


def test_scraper_parses_html_cleanly():
    """Verify clean_html removes boilerplate and parses title and content cleanly."""
    dirty_html = """
    <!DOCTYPE html>
    <html>
      <head>
        <title>ApexFin Cloud News</title>
        <style>body { color: red; }</style>
        <script>console.log("analytics");</script>
      </head>
      <body>
        <nav><a href="/home">Home</a><a href="/pricing">Pricing</a></nav>
        <header><h1>Header Navigation</h1></header>
        <main>
          <h1>ApexFin Expands Kubernetes Footprint</h1>
          <p>ApexFin announced a 400% surge in AWS cloud transaction volume.</p>
          <aside>Related articles</aside>
        </main>
        <footer><p>Copyright 2026 ApexFin Inc.</p></footer>
      </body>
    </html>
    """
    res = clean_html(dirty_html, "https://apexfin.io")
    assert res["title"] == "ApexFin Cloud News"
    assert "analytics" not in res["clean_text"]
    assert "Home" not in res["clean_text"]
    assert "Copyright" not in res["clean_text"]
    assert "ApexFin announced a 400% surge in AWS cloud transaction volume." in res["clean_text"]
    assert "ApexFin Expands Kubernetes Footprint" in res["clean_text"]


@pytest.mark.asyncio
async def test_scraper_raises_offline_mode_error():
    """Verify that scrape_public_url raises OfflineModeError when STRICT_OFFLINE_MODE is active."""
    with StrictOfflineSandbox():
        with pytest.raises(OfflineModeError) as exc_info:
            await scrape_public_url("https://example.com/test")
        assert "Live web scraping requires online mode" in str(exc_info.value)


@pytest.mark.asyncio
async def test_scraper_mock_httpx_success():
    """Verify scrape_public_url handles standard web pages cleanly via httpx."""
    html_content = """
    <html>
      <head><title>NovaPay Security Advisory</title></head>
      <body>
        <h1>CI/CD Security Hardening</h1>
        <p>NovaPay is establishing automated software supply chain provenance for all GitHub workflows.
           We are consolidating DevSecOps tooling to ensure FedRAMP and PCI-DSS compliance before European expansion.</p>
      </body>
    </html>
    """
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = html_content

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        res = await scrape_public_url("https://novapay.com/security")
        assert res["title"] == "NovaPay Security Advisory"
        assert "software supply chain provenance" in res["clean_text"]
        assert res["url"] == "https://novapay.com/security"
