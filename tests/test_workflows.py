"""Tests for Phase 4: Local sales workflows, differentiators, deterministic fallback, and routes."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from scripts.seed_demo_data import seed_demo_data
from sellbird.dashboard import app
from sellbird.local_ai.network_guard import StrictOfflineSandbox
from sellbird.local_ai.ollama_client import OllamaUnavailableError
from sellbird.local_ai.routes import set_engine
from sellbird.local_ai.store import LocalSalesStore
from sellbird.local_ai.workflows import (
    CompanyProfile,
    OutreachEmailResponse,
    ProblemStatementResponse,
    ProductICP,
    ProspectResearch,
    QualificationScoreResponse,
    ReplyClassificationResponse,
    SignalItem,
    SignalsExtraction,
    WorkflowEngine,
    verify_quote_grounding,
)


@pytest.fixture
def seeded_store(tmp_path):
    """Fixture providing an initialized SQLite store seeded with demo data."""
    import asyncio
    db_file = tmp_path / "test_workflows.db"
    asyncio.run(seed_demo_data(db_path=db_file))
    store = LocalSalesStore(db_file)
    yield store
    asyncio.run(store.close())


# =============================================================================
# 1. Evidence Grounding Verifier Tests (Differentiator 2)
# =============================================================================

def test_evidence_grounding_verifier():
    raw_text = (
        "ApexFin Systems announced a $45M Series B funding round to scale its real-time payment rails. "
        "Sarah Jenkins, VP of Infrastructure, noted: 'As our transaction volume expanded 400% over the last year, "
        "our AWS and multi-region Kubernetes cloud bills skyrocketed beyond budget. Our team is actively seeking automated "
        "FinOps tools to reign in cloud infrastructure costs while preparing for SOC2 Type II audits.'"
    )

    # Verbatim match
    exact_quote = "our AWS and multi-region Kubernetes cloud bills skyrocketed beyond budget"
    assert verify_quote_grounding(exact_quote, raw_text) is True

    # Punctuation/case differences
    fuzzy_quote = "our aws and multi region kubernetes cloud bills skyrocketed beyond budget."
    assert verify_quote_grounding(fuzzy_quote, raw_text) is True

    # Hallucinated quote
    hallucinated = "ApexFin is firing 30% of their staff and halting all cloud migrations."
    assert verify_quote_grounding(hallucinated, raw_text) is False

    # Empty inputs
    assert verify_quote_grounding("", raw_text) is False
    assert verify_quote_grounding("sample", "") is False


# =============================================================================
# 2. End-to-End Workflows with Real Embeddings inside Offline Sandbox
# =============================================================================

@pytest.mark.asyncio
async def test_all_workflows_end_to_end_in_offline_sandbox(seeded_store):
    """Run all 8 workflows + Product Fit matrix strictly inside StrictOfflineSandbox."""
    with StrictOfflineSandbox():
        mock_ollama = AsyncMock()

        # Mock responses for Ollama calls
        mock_profile = CompanyProfile(
            industry="Enterprise Cloud Operations",
            value_proposition="Autonomous FinOps and DevSecOps governance platform",
            target_market=["Fintech", "Healthtech", "B2B SaaS"],
            differentiators=["100% Local AI", "Zero telemetry data leaks"],
            summary="Vanguard Ops AI provides sovereign autonomous operations for engineering leaders.",
        )

        mock_icp = ProductICP(
            target_roles=["VP of Infrastructure", "FinOps Lead", "Head of Cloud"],
            company_size=["50-1000 employees"],
            industries=["Fintech", "High-growth SaaS"],
            pain_points=["Skyrocketing AWS/GCP bills", "Unattached resources"],
            buying_triggers=["Funding round", "Kubernetes cluster migration"],
            disqualifiers=["Pre-revenue startups"],
        )

        mock_research = ProspectResearch(
            summary="ApexFin is scaling payment rails after $45M Series B.",
            tech_stack=["AWS", "Kubernetes", "EKS"],
            key_initiatives=["FinOps automation", "SOC2 Type II prep"],
            pain_indicators=["Cloud bills skyrocketed beyond budget"],
        )

        mock_signals = SignalsExtraction(
            signals=[
                SignalItem(
                    signal_type="cost_pressure",
                    confidence=0.95,
                    evidence_quote="our AWS and multi-region Kubernetes cloud bills skyrocketed beyond budget",
                    reasoning="Executive stated cloud bills exceeded budget after transaction surge.",
                ),
                SignalItem(
                    signal_type="funding",
                    confidence=0.98,
                    evidence_quote="ApexFin Systems announced a $45M Series B funding round",
                    reasoning="Recent major funding round.",
                ),
                SignalItem(
                    signal_type="compliance_audit",
                    confidence=0.40,
                    evidence_quote="ApexFin is closing down European headquarters next month",  # Hallucinated
                    reasoning="False signal test.",
                ),
            ]
        )

        mock_problem = ProblemStatementResponse(
            statement=(
                "ApexFin's rapid 400% transaction surge has triggered unsustainable AWS Kubernetes expenditure, "
                "demanding autonomous FinOps controls before SOC2 certification."
            ),
            grounding_signals=["cost_pressure", "compliance_audit"],
        )

        mock_score = QualificationScoreResponse(
            llm_score=85.0,
            reasoning="High buying urgency due to severe cloud budget overrun and active FinOps tool search.",
        )

        mock_email = OutreachEmailResponse(
            subject="Taming ApexFin's AWS spend after your Series B",
            body=(
                "Hi Sarah,\n\n"
                "Saw your note on ApexFin's transaction volume surging 400% and cloud bills exceeding budget.\n\n"
                "CloudCost Guard autonomously eliminates idle compute and rightsizes Kubernetes clusters with zero downtime.\n\n"
                "Would you be open to a 10-minute demo this week?\n\nBest,\nAlex"
            ),
        )

        mock_reply = ReplyClassificationResponse(
            intent="interested",
            confidence=0.96,
            next_action="send_calendar_invite",
            reasoning="Prospect confirmed interest in demo.",
        )

        mock_ollama.generate_json.side_effect = [
            mock_profile,
            mock_icp,
            mock_research,
            mock_signals,
            mock_problem,
            mock_score,
            mock_email,
            mock_reply,
        ]

        engine = WorkflowEngine(store=seeded_store, ollama_client=mock_ollama)

        # 1. Company Profile
        res_profile = await engine.generate_company_profile(company_id=1)
        assert res_profile["status"] == "success"
        assert res_profile["profile"]["industry"] == "Enterprise Cloud Operations"

        # 2. Product ICP
        res_icp = await engine.generate_product_icp(product_id=1)
        assert res_icp["status"] == "success"
        icp_row = await seeded_store.get_icp(product_id=1)
        assert icp_row is not None
        assert icp_row["embedding_json"] is not None

        # 3. Prospect Research
        res_research = await engine.research_prospect(prospect_id=1)
        assert res_research["status"] == "success"
        assert "AWS" in res_research["research"]["tech_stack"]

        # 4. Multi-Product Fit Matrix (Real local embeddings, zero LLM needed)
        res_fit = await engine.calculate_product_fit(prospect_id=1)
        assert res_fit["status"] == "success"
        assert res_fit["primary_product_id"] == 1  # CloudCost Guard is top fit for ApexFin FinOps cues
        assert "CloudCost Guard" in res_fit["scores"]

        # 5. Buying Signals + Grounding
        res_signals = await engine.classify_buying_signals(prospect_id=1)
        assert res_signals["status"] == "success"
        signals = res_signals["signals"]
        assert len(signals) == 3
        # First two were verbatim quotes in raw_text -> is_grounded = True
        assert signals[0]["is_grounded"] is True
        assert signals[1]["is_grounded"] is True
        # Third was a hallucinated quote -> is_grounded = False!
        assert signals[2]["is_grounded"] is False

        # 6. Problem Statement
        res_prob = await engine.generate_problem_statement(prospect_id=1)
        assert res_prob["status"] == "success"
        assert "ApexFin" in res_prob["statement"]

        # 7. Hybrid Prospect Scoring
        res_score = await engine.score_prospect(prospect_id=1)
        assert res_score["status"] == "success"
        assert res_score["total_score"] > 60.0
        assert res_score["priority"] in ["High", "Medium"]
        assert res_score["deterministic_score"] > 0
        assert res_score["llm_score"] == 85.0

        # 8. Personalised Outreach Email
        res_email = await engine.generate_outreach_email(prospect_id=1)
        assert res_email["status"] == "success"
        assert res_email["email_status"] == "draft_only"
        assert "Series B" in res_email["subject"]

        # 9. Reply Classification
        res_rep = await engine.classify_reply(
            prospect_id=1,
            reply_text="Sounds great, let's talk this Thursday at 2 PM.",
        )
        assert res_rep["status"] == "success"
        assert res_rep["intent"] == "interested"
        assert res_rep["next_action"] == "send_calendar_invite"


# =============================================================================
# 3. HTTP Routes, Deterministic Fallback on Ollama Failure & Queue Drain
# =============================================================================

@pytest.mark.asyncio
async def test_routes_fallback_and_queue_draining(seeded_store):
    """Test 503 fallback on Ollama failure, task queueing, and queue drain."""
    # 1. Setup engine with failing Ollama
    failing_ollama = AsyncMock()
    failing_ollama.generate_json.side_effect = OllamaUnavailableError("Ollama daemon is down.")
    failing_ollama.health_check.side_effect = OllamaUnavailableError("Cannot connect to port 11434.")

    engine = WorkflowEngine(store=seeded_store, ollama_client=failing_ollama)
    set_engine(engine)

    client = TestClient(app, base_url="http://127.0.0.1:5555")

    # System Status
    resp_status = client.get("/api/local/system/status")
    assert resp_status.status_code == 200
    st_data = resp_status.json()
    assert st_data["ollama_online"] is False
    assert "hardware_profile" in st_data

    # Call AI workflow endpoint while Ollama is offline -> Expect 503 + fallback + queue
    resp_signals = client.post("/api/local/prospects/1/signals")
    assert resp_signals.status_code == 503
    err_body = resp_signals.json()
    assert err_body["ollama_offline"] is True
    assert "queued_task_id" in err_body
    assert "deterministic_fallback" in err_body
    queued_task_id = err_body["queued_task_id"]

    # Verify task was added to task_queue table
    resp_queue = client.get("/api/local/queue")
    assert resp_queue.status_code == 200
    tasks = resp_queue.json()
    assert any(t["id"] == queued_task_id and t["status"] == "pending" for t in tasks)

    # Another workflow with fallback: scoring
    resp_score = client.post("/api/local/prospects/1/score", json={})
    assert resp_score.status_code == 503
    score_err = resp_score.json()
    assert score_err["ollama_offline"] is True
    assert score_err["deterministic_fallback"]["total_score"] > 0

    # 2. Simulate Ollama recovery: switch client to working mock
    recovered_ollama = AsyncMock()
    mock_signals = SignalsExtraction(
        signals=[
            SignalItem(
                signal_type="cost_pressure",
                confidence=0.92,
                evidence_quote="our AWS and multi-region Kubernetes cloud bills skyrocketed beyond budget",
                reasoning="Cloud bill expansion cue",
            )
        ]
    )
    mock_score = QualificationScoreResponse(
        llm_score=90.0,
        reasoning="Recovered Ollama scored this lead as tier 1 priority.",
    )
    recovered_ollama.generate_json.side_effect = [mock_signals, mock_score]
    recovered_ollama.health_check.return_value = ["qwen3:4b"]

    engine.ollama = recovered_ollama

    # Process queue
    resp_drain = client.post("/api/local/queue/process")
    assert resp_drain.status_code == 200
    drain_result = resp_drain.json()
    assert drain_result["processed"] >= 1

    # Verify task in queue is now completed
    resp_queue_after = client.get("/api/local/queue")
    tasks_after = resp_queue_after.json()
    target_task = next(t for t in tasks_after if t["id"] == queued_task_id)
    assert target_task["status"] == "completed"
