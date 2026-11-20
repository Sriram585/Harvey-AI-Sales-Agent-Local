"""Core local AI workflows for multi-product sales automation.

Strictly offline and local execution:
- Reads from and writes to LocalSalesStore (SQLite).
- Calls native OllamaClient (POST /api/chat).
- Computes local embeddings with sentence-transformers/all-MiniLM-L6-v2.
- Handles Ollama failures gracefully with deterministic fallbacks and task queueing.
"""

import difflib
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from sellbird.local_ai import embeddings
from sellbird.local_ai.ollama_client import (
    DEFAULT_RECOMMENDED_MODEL,
    OllamaClient,
    OllamaUnavailableError,
)
from sellbird.local_ai.store import LocalSalesStore

logger = logging.getLogger("sellbird.local_ai.workflows")


# =============================================================================
# Pydantic Schemas for Ollama Structured Output
# =============================================================================

class CompanyProfile(BaseModel):
    industry: str = Field(..., description="Primary industry and market domain")
    value_proposition: str = Field(..., description="Core value proposition and benefit")
    target_market: list[str] = Field(default_factory=list, description="Target customer segments")
    differentiators: list[str] = Field(default_factory=list, description="Key competitive differentiators")
    summary: str = Field(..., description="Executive summary of the business")


class ProductICP(BaseModel):
    target_roles: list[str] = Field(default_factory=list, description="Target buyer and user titles")
    company_size: list[str] = Field(default_factory=list, description="Target company size ranges")
    industries: list[str] = Field(default_factory=list, description="Target vertical industries")
    pain_points: list[str] = Field(default_factory=list, description="Key operational pain points solved")
    buying_triggers: list[str] = Field(default_factory=list, description="Events that trigger purchase intent")
    disqualifiers: list[str] = Field(default_factory=list, description="Characteristics that disqualify a lead")


class ProspectResearch(BaseModel):
    summary: str = Field(..., description="Concise summary of prospect business and recent updates")
    tech_stack: list[str] = Field(default_factory=list, description="Observed cloud and developer tools")
    key_initiatives: list[str] = Field(default_factory=list, description="Active projects and initiatives")
    pain_indicators: list[str] = Field(default_factory=list, description="Direct cues of friction or waste")


class SignalItem(BaseModel):
    signal_type: str = Field(
        ...,
        description="One of: hiring, expansion, funding, product_launch, tech_migration, compliance_audit, cost_pressure, none",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    evidence_quote: str = Field(
        ..., description="EXACT verbatim quote copied directly from raw_text demonstrating this signal"
    )
    reasoning: str = Field(..., description="Explanation of why this indicates buying intent")


class SignalsExtraction(BaseModel):
    signals: list[SignalItem] = Field(default_factory=list)


class ProblemStatementResponse(BaseModel):
    statement: str = Field(..., description="Compelling 1-2 sentence problem statement tailored to prospect")
    grounding_signals: list[str] = Field(default_factory=list, description="Signals supporting this statement")


class QualificationScoreResponse(BaseModel):
    llm_score: float = Field(..., ge=0.0, le=100.0, description="Subjective qualification score 0-100")
    reasoning: str = Field(..., description="Detailed rationale for the qualification score")


class OutreachEmailResponse(BaseModel):
    subject: str = Field(..., description="Short, captivating email subject line")
    body: str = Field(..., description="Personalized, concise 3-paragraph cold outreach body")


class ReplyClassificationResponse(BaseModel):
    intent: str = Field(
        ...,
        description="One of: interested, objection, not_now, unsubscribe, out_of_office, referral",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")
    next_action: str = Field(..., description="Recommended automated next step")
    reasoning: str = Field(..., description="Reasoning for this intent classification")


# =============================================================================
# Evidence Grounding Verifier (Differentiator 2)
# =============================================================================

def normalize_text_for_matching(text: str) -> str:
    """Normalize text by extracting lowercase alphanumeric tokens joined by single spaces."""
    if not text:
        return ""
    return " ".join(re.findall(r"\w+", text.lower()))


def verify_quote_grounding(quote: str, raw_text: str) -> bool:
    """Deterministically verify that an extracted evidence quote actually exists in raw_text.

    Returns True if exact normalized substring match, or if a sliding-window
    SequenceMatcher similarity >= 0.85 is found.
    """
    if not quote or not raw_text:
        return False

    norm_quote = normalize_text_for_matching(quote)
    norm_raw = normalize_text_for_matching(raw_text)
    if not norm_quote or not norm_raw:
        return False

    # 1. Exact normalized substring match
    if norm_quote in norm_raw:
        return True

    # 2. Sliding-window fuzzy match
    quote_words = norm_quote.split()
    raw_words = norm_raw.split()
    k = len(quote_words)
    if k == 0:
        return False

    if len(raw_words) < k:
        return difflib.SequenceMatcher(None, norm_quote, norm_raw).ratio() >= 0.85

    # Check exact window length k
    for i in range(len(raw_words) - k + 1):
        window = " ".join(raw_words[i : i + k])
        if difflib.SequenceMatcher(None, norm_quote, window).ratio() >= 0.85:
            return True

    # Check slightly wider window (k + 2) for minor dropped words
    if len(raw_words) >= k + 2:
        for i in range(len(raw_words) - (k + 2) + 1):
            window = " ".join(raw_words[i : i + k + 2])
            if difflib.SequenceMatcher(None, norm_quote, window).ratio() >= 0.85:
                return True

    return False


# =============================================================================
# Workflow Engine
# =============================================================================

class WorkflowEngine:
    """Executes offline sales AI workflows backed by SQLite and local Ollama inference."""

    def __init__(
        self,
        store: LocalSalesStore,
        ollama_client: OllamaClient | None = None,
        model: str = DEFAULT_RECOMMENDED_MODEL,
    ):
        self.store = store
        self.model = model
        # Wire OllamaClient to log to store.log_inference
        self.ollama = ollama_client or OllamaClient(
            db_log_callback=self.store.log_inference
        )

    # -------------------------------------------------------------------------
    # 1. Business Profile Generation
    # -------------------------------------------------------------------------
    async def generate_company_profile(self, company_id: int) -> dict[str, Any]:
        company = await self.store.get_company(company_id)
        if not company:
            raise ValueError(f"Company {company_id} not found.")

        raw_notes = company.get("raw_notes") or company.get("name", "")
        prompt = (
            f"Analyze the following company notes and synthesize a structured business profile:\n\n"
            f"Company: {company['name']}\n"
            f"Website: {company.get('website', '')}\n"
            f"Notes: {raw_notes}\n\n"
            "Respond with a flat JSON object with keys: industry, value_proposition, target_market, differentiators, summary.\n"
            "Do NOT wrap in 'properties' or output schema definitions."
        )
        system_prompt = "You are an expert enterprise B2B strategy analyst. Output valid JSON only."

        try:
            profile: CompanyProfile = await self.ollama.generate_json(
                prompt=prompt,
                system_prompt=system_prompt,
                response_model=CompanyProfile,
                workflow_name="company.profile_generation",
                model=self.model,
            )
            profile_dict = profile.model_dump()
            await self.store.connect()
            conn = await self.store.connect()
            await conn.execute(
                "UPDATE companies SET profile_json = ? WHERE id = ?",
                (json.dumps(profile_dict), company_id),
            )
            await conn.commit()
            return {"status": "success", "company_id": company_id, "profile": profile_dict}

        except (OllamaUnavailableError, ValidationError):
            # Deterministic Fallback
            fallback = {
                "industry": "Enterprise Software",
                "value_proposition": "Automated operations and developer platforms",
                "target_market": ["Enterprise", "Mid-market B2B SaaS"],
                "differentiators": ["Local AI", "Privacy-first architecture"],
                "summary": raw_notes[:250],
            }
            conn = await self.store.connect()
            await conn.execute(
                "UPDATE companies SET profile_json = ? WHERE id = ?",
                (json.dumps(fallback), company_id),
            )
            await conn.commit()
            task_id = await self.store.enqueue_task(
                task_type="company_profile",
                payload_json={"company_id": company_id},
            )
            raise OllamaWorkflowError(
                message="Ollama is unavailable. Task queued for local processing.",
                queued_task_id=task_id,
                deterministic_fallback=fallback,
            )

    # -------------------------------------------------------------------------
    # 2. Product-Specific ICP Generation
    # -------------------------------------------------------------------------
    async def generate_product_icp(self, product_id: int) -> dict[str, Any]:
        product = await self.store.get_product(product_id)
        if not product:
            raise ValueError(f"Product {product_id} not found.")

        company = await self.store.get_company(product["company_id"])
        comp_profile = company.get("profile_json", "{}") if company else "{}"

        prompt = (
            f"Generate an Ideal Customer Profile (ICP) for the following product:\n"
            f"Company: {company['name'] if company else 'Unknown'}\n"
            f"Company Profile: {comp_profile}\n"
            f"Product Name: {product['name']}\n"
            f"Description: {product.get('description', '')}\n"
            f"Value Proposition: {product.get('value_prop', '')}\n"
            f"Pricing Tier: {product.get('pricing_tier', '')}\n\n"
            "Respond with a flat JSON object with keys: target_roles, company_size, industries, pain_points, buying_triggers, disqualifiers.\n"
            "Do NOT wrap in 'properties' or output schema definitions."
        )
        system_prompt = "You are a specialized B2B sales development expert. Output valid JSON only."

        try:
            icp: ProductICP = await self.ollama.generate_json(
                prompt=prompt,
                system_prompt=system_prompt,
                response_model=ProductICP,
                workflow_name="product.icp_generation",
                model=self.model,
            )
            icp_dict = icp.model_dump()

            # Compute local embedding of ICP text
            icp_text = (
                f"{product['name']}: {product.get('description', '')} {product.get('value_prop', '')}. "
                f"Pain points: {' '.join(icp.pain_points)}. Buying triggers: {' '.join(icp.buying_triggers)}"
            )
            emb_vector = embeddings.embed_text(icp_text)

            icp_id = await self.store.save_icp(
                product_id=product_id,
                icp_json=icp_dict,
                embedding_json=emb_vector,
            )
            return {"status": "success", "product_id": product_id, "icp_id": icp_id, "icp": icp_dict}

        except (OllamaUnavailableError, ValidationError):
            fallback = {
                "target_roles": ["VP Engineering", "DevOps Lead", "CTO", "Head of Infrastructure"],
                "company_size": ["50-1000"],
                "industries": ["Fintech", "Enterprise Software", "Cloud Infrastructure"],
                "pain_points": [product.get("description", "Operational friction")],
                "buying_triggers": ["Infrastructure scaling", "Budget reviews", "Security audit"],
                "disqualifiers": ["Pre-seed companies without cloud footprint"],
            }
            emb_text = f"{product['name']}: {product.get('description', '')} {product.get('value_prop', '')}"
            emb_vector = embeddings.embed_text(emb_text)
            icp_id = await self.store.save_icp(
                product_id=product_id,
                icp_json=fallback,
                embedding_json=emb_vector,
            )
            task_id = await self.store.enqueue_task(
                task_type="product_icp",
                payload_json={"product_id": product_id},
            )
            raise OllamaWorkflowError(
                message="Ollama is unavailable. Task queued for local processing.",
                queued_task_id=task_id,
                deterministic_fallback=fallback,
            )

    # -------------------------------------------------------------------------
    # 3. Prospect Research (Offline-Safe)
    # -------------------------------------------------------------------------
    async def research_prospect(self, prospect_id: int) -> dict[str, Any]:
        prospect = await self.store.get_prospect(prospect_id)
        if not prospect:
            raise ValueError(f"Prospect {prospect_id} not found.")

        raw_text = prospect.get("raw_text") or ""
        if not raw_text:
            raise ValueError(f"Prospect {prospect_id} has no raw_text to research.")

        prompt = (
            f"Analyze this prospect's scraped public text and extract research intelligence:\n\n"
            f"Company: {prospect['company_name']}\n"
            f"Contact: {prospect.get('contact_name', '')} ({prospect.get('contact_role', '')})\n"
            f"Raw Content:\n{raw_text}\n\n"
            "Return JSON matching the schema."
        )
        system_prompt = "You are an expert sales intelligence researcher. Extract facts strictly present in the text."

        try:
            research: ProspectResearch = await self.ollama.generate_json(
                prompt=prompt,
                system_prompt=system_prompt,
                response_model=ProspectResearch,
                workflow_name="prospect.research",
                model=self.model,
            )
            research_dict = research.model_dump()
            await self.store.update_prospect(prospect_id, research_json=research_dict)
            return {"status": "success", "prospect_id": prospect_id, "research": research_dict}

        except (OllamaUnavailableError, ValidationError):
            # Deterministic Tech-Stack scan
            tech_keywords = ["AWS", "GCP", "Kubernetes", "EKS", "GKE", "GitHub", "GitLab", "npm", "Docker"]
            found_tech = [t for t in tech_keywords if t.lower() in raw_text.lower()]
            fallback = {
                "summary": raw_text[:300],
                "tech_stack": found_tech,
                "key_initiatives": ["Infrastructure scaling and compliance operations"],
                "pain_indicators": ["Spiraling costs or security review requirements"],
            }
            await self.store.update_prospect(prospect_id, research_json=fallback)
            task_id = await self.store.enqueue_task(
                task_type="prospect_research",
                payload_json={"prospect_id": prospect_id},
            )
            raise OllamaWorkflowError(
                message="Ollama is unavailable. Task queued for local processing.",
                queued_task_id=task_id,
                deterministic_fallback=fallback,
            )

    # -------------------------------------------------------------------------
    # 4. Multi-Product Fit Matrix (Differentiator 1)
    # -------------------------------------------------------------------------
    async def calculate_product_fit(
        self, prospect_id: int, threshold: float = 0.35
    ) -> dict[str, Any]:
        """Compute cosine similarity of prospect raw_text against ALL products' ICP embeddings.

        Completely local and deterministic via all-MiniLM-L6-v2 embeddings.
        No LLM calls required.
        """
        prospect = await self.store.get_prospect(prospect_id)
        if not prospect:
            raise ValueError(f"Prospect {prospect_id} not found.")

        raw_text = prospect.get("raw_text") or prospect["company_name"]
        prospect_vector = embeddings.embed_text(raw_text)

        products = await self.store.list_products()
        if not products:
            raise ValueError("No products found in store to calculate fit against.")

        scores: dict[str, float] = {}
        product_scores: list[tuple[int, float]] = []

        for prod in products:
            prod_id = prod["id"]
            icp = await self.store.get_icp(prod_id)
            if icp and icp.get("embedding_json"):
                prod_vector = json.loads(icp["embedding_json"])
            else:
                prod_text = f"{prod['name']}: {prod.get('description', '')}. {prod.get('value_prop', '')}"
                prod_vector = embeddings.embed_text(prod_text)

            sim = float(embeddings.cosine_similarity(prospect_vector, prod_vector))
            sim_rounded = round(max(0.0, min(1.0, sim)), 4)
            scores[prod["name"]] = sim_rounded
            product_scores.append((prod_id, sim_rounded))

        # Sort descending by score
        product_scores.sort(key=lambda x: x[1], reverse=True)
        primary_product_id = product_scores[0][0]

        # Secondary products meeting the threshold
        cross_sell_ids = [
            pid for pid, score in product_scores[1:] if score >= threshold
        ]

        matrix_id = await self.store.save_product_fit(
            prospect_id=prospect_id,
            primary_product_id=primary_product_id,
            cross_sell_product_ids_json=cross_sell_ids,
            scores_json=scores,
        )

        return {
            "status": "success",
            "matrix_id": matrix_id,
            "prospect_id": prospect_id,
            "primary_product_id": primary_product_id,
            "cross_sell_product_ids": cross_sell_ids,
            "scores": scores,
        }

    # -------------------------------------------------------------------------
    # 5. Buying-Signal Classification + Evidence Grounding (Differentiator 2)
    # -------------------------------------------------------------------------
    async def classify_buying_signals(self, prospect_id: int) -> dict[str, Any]:
        prospect = await self.store.get_prospect(prospect_id)
        if not prospect:
            raise ValueError(f"Prospect {prospect_id} not found.")

        raw_text = prospect.get("raw_text") or ""
        if not raw_text:
            raise ValueError(f"Prospect {prospect_id} has no raw_text.")

        prompt = (
            f"Identify all verifiable buying signals in the following text for {prospect['company_name']}.\n\n"
            f"Allowed signal types: hiring, expansion, funding, product_launch, tech_migration, compliance_audit, cost_pressure, none.\n"
            f"CRITICAL REQUIREMENT: For each signal, 'evidence_quote' MUST be an EXACT, VERBATIM substring copied word-for-word from the text.\n\n"
            f"Text:\n{raw_text}\n\n"
            "Return JSON matching the schema."
        )
        system_prompt = (
            "You are an evidence-grounded B2B qualification agent. "
            "Never paraphrase evidence quotes. Copy verbatim substrings from the source text."
        )

        try:
            extraction: SignalsExtraction = await self.ollama.generate_json(
                prompt=prompt,
                system_prompt=system_prompt,
                response_model=SignalsExtraction,
                workflow_name="prospect.signals_classification",
                model=self.model,
            )

            processed_signals = []
            for item in extraction.signals:
                if item.signal_type == "none":
                    continue
                is_grounded = verify_quote_grounding(item.evidence_quote, raw_text)
                sig_id = await self.store.add_buying_signal(
                    prospect_id=prospect_id,
                    signal_type=item.signal_type,
                    confidence=item.confidence,
                    evidence_quote=item.evidence_quote,
                    is_grounded=is_grounded,
                    reasoning=item.reasoning,
                )
                processed_signals.append({
                    "id": sig_id,
                    "signal_type": item.signal_type,
                    "confidence": item.confidence,
                    "evidence_quote": item.evidence_quote,
                    "is_grounded": is_grounded,
                    "reasoning": item.reasoning,
                })

            return {"status": "success", "prospect_id": prospect_id, "signals": processed_signals}

        except (OllamaUnavailableError, ValidationError):
            # Deterministic Keyword Fallback
            signals_found = self._deterministic_extract_signals(raw_text)
            processed_signals = []
            for sig in signals_found:
                is_grounded = verify_quote_grounding(sig["evidence_quote"], raw_text)
                sig_id = await self.store.add_buying_signal(
                    prospect_id=prospect_id,
                    signal_type=sig["signal_type"],
                    confidence=sig["confidence"],
                    evidence_quote=sig["evidence_quote"],
                    is_grounded=is_grounded,
                    reasoning=sig["reasoning"],
                )
                processed_signals.append({
                    "id": sig_id,
                    "signal_type": sig["signal_type"],
                    "confidence": sig["confidence"],
                    "evidence_quote": sig["evidence_quote"],
                    "is_grounded": is_grounded,
                    "reasoning": sig["reasoning"],
                })

            task_id = await self.store.enqueue_task(
                task_type="buying_signals",
                payload_json={"prospect_id": prospect_id},
            )
            raise OllamaWorkflowError(
                message="Ollama is unavailable. Task queued for local processing.",
                queued_task_id=task_id,
                deterministic_fallback={"signals": processed_signals},
            )

    def _deterministic_extract_signals(self, raw_text: str) -> list[dict[str, Any]]:
        """Extract signals deterministically by sentence scanning when Ollama is offline."""
        sentences = re.split(r"(?<=[.!?])\s+", raw_text)
        detected = []

        patterns = [
            (r"\b(cloud bill|spend|cost|expensive|waste|ebs)\b", "cost_pressure", "Cloud cost and spend cues detected"),
            (r"\b(soc2|audit|compliance|fedramp|hipaa|pci)\b", "compliance_audit", "Security compliance cues detected"),
            (r"\b(migrat|kubernetes|gcp|aws|eks|gke)\b", "tech_migration", "Cloud infrastructure migration detected"),
            (r"\b(series|funding|round|\$\d+m)\b", "funding", "Fundraising milestone detected"),
            (r"\b(hiring|recruit|headcount)\b", "hiring", "Active engineering hiring detected"),
        ]

        seen_types = set()
        for sent in sentences:
            sent_clean = sent.strip()
            if not sent_clean:
                continue
            for regex, sig_type, reason in patterns:
                if sig_type not in seen_types and re.search(regex, sent_clean, re.IGNORECASE):
                    seen_types.add(sig_type)
                    detected.append({
                        "signal_type": sig_type,
                        "confidence": 0.85,
                        "evidence_quote": sent_clean,
                        "reasoning": reason,
                    })
                    break
        return detected

    # -------------------------------------------------------------------------
    # 6. Problem-Statement Generation
    # -------------------------------------------------------------------------
    async def generate_problem_statement(
        self, prospect_id: int, product_id: int | None = None
    ) -> dict[str, Any]:
        prospect = await self.store.get_prospect(prospect_id)
        if not prospect:
            raise ValueError(f"Prospect {prospect_id} not found.")

        # If product_id not supplied, retrieve primary from fit matrix or calculate
        if product_id is None:
            fit = await self.store.get_product_fit(prospect_id)
            if not fit:
                fit = await self.calculate_product_fit(prospect_id)
            product_id = fit["primary_product_id"]

        product = await self.store.get_product(product_id)
        if not product:
            raise ValueError(f"Product {product_id} not found.")

        # Collect grounded signals
        all_signals = await self.store.list_buying_signals(prospect_id)
        grounded_signals = [s for s in all_signals if s.get("is_grounded", 0) == 1]
        signals_summary = "\n".join(
            f"- [{s['signal_type']}] \"{s['evidence_quote']}\" (Confidence: {s['confidence']})"
            for s in grounded_signals
        ) or "No specific buying signals recorded."

        prompt = (
            f"Generate a crisp, compelling problem statement for {prospect['company_name']}.\n\n"
            f"Product: {product['name']}\n"
            f"Product Value Prop: {product.get('value_prop', '')}\n"
            f"Verified Grounded Buying Signals:\n{signals_summary}\n\n"
            "Synthesize the prospect's operational friction into a tailored problem statement.\n"
            "Return JSON matching the schema."
        )
        system_prompt = "You are an elite B2B enterprise solution strategist. Output valid JSON only."

        try:
            resp: ProblemStatementResponse = await self.ollama.generate_json(
                prompt=prompt,
                system_prompt=system_prompt,
                response_model=ProblemStatementResponse,
                workflow_name="prospect.problem_statement",
                model=self.model,
            )
            prob_id = await self.store.save_problem_statement(
                prospect_id=prospect_id,
                product_id=product_id,
                statement=resp.statement,
                grounding_signals_json=resp.grounding_signals,
            )
            return {
                "status": "success",
                "problem_statement_id": prob_id,
                "prospect_id": prospect_id,
                "product_id": product_id,
                "statement": resp.statement,
                "grounding_signals": resp.grounding_signals,
            }

        except (OllamaUnavailableError, ValidationError):
            evidence = grounded_signals[0]["evidence_quote"] if grounded_signals else "operational complexity"
            sig_type = grounded_signals[0]["signal_type"] if grounded_signals else "cloud operations"
            fallback_statement = (
                f"{prospect['company_name']} is facing acute challenges with {sig_type}, "
                f"evidenced by: '{evidence}'. {product['name']} provides immediate resolution via {product.get('value_prop', '')}."
            )
            fallback_signals = [sig_type]
            prob_id = await self.store.save_problem_statement(
                prospect_id=prospect_id,
                product_id=product_id,
                statement=fallback_statement,
                grounding_signals_json=fallback_signals,
            )
            task_id = await self.store.enqueue_task(
                task_type="problem_statement",
                payload_json={"prospect_id": prospect_id, "product_id": product_id},
            )
            raise OllamaWorkflowError(
                message="Ollama is unavailable. Task queued for local processing.",
                queued_task_id=task_id,
                deterministic_fallback={
                    "statement": fallback_statement,
                    "grounding_signals": fallback_signals,
                },
            )

    # -------------------------------------------------------------------------
    # 7. Hybrid Prospect Scoring & Prioritisation (Differentiator 3)
    # -------------------------------------------------------------------------
    async def score_prospect(
        self, prospect_id: int, product_id: int | None = None
    ) -> dict[str, Any]:
        """Compute hybrid score: 50% deterministic (embedding fit + verified signals) + 50% LLM."""
        prospect = await self.store.get_prospect(prospect_id)
        if not prospect:
            raise ValueError(f"Prospect {prospect_id} not found.")

        if product_id is None:
            fit = await self.store.get_product_fit(prospect_id)
            if not fit:
                fit = await self.calculate_product_fit(prospect_id)
            product_id = fit["primary_product_id"]

        product = await self.store.get_product(product_id)

        # 1. Deterministic component (0-100)
        fit_matrix = await self.store.get_product_fit(prospect_id)
        if not fit_matrix:
            fit_matrix = await self.calculate_product_fit(prospect_id)

        scores_dict = json.loads(fit_matrix.get("scores_json", "{}")) if fit_matrix else {}
        emb_similarity = float(scores_dict.get(product["name"], 0.5)) if product else 0.5
        emb_component = emb_similarity * 50.0  # up to 50 pts

        signals = await self.store.list_buying_signals(prospect_id)
        grounded_signals = [s for s in signals if s.get("is_grounded", 0) == 1]

        signal_points = 0.0
        signal_weights = {
            "cost_pressure": 20.0,
            "compliance_audit": 18.0,
            "tech_migration": 15.0,
            "funding": 12.0,
            "hiring": 10.0,
            "expansion": 10.0,
        }
        for s in grounded_signals:
            signal_points += signal_weights.get(s["signal_type"], 8.0) * float(s.get("confidence", 0.8))
        signal_component = min(50.0, signal_points)

        deterministic_score = round(min(100.0, emb_component + signal_component), 2)

        prompt = (
            f"Qualify this prospect for {product['name'] if product else 'our solution'}.\n\n"
            f"Prospect: {prospect['company_name']}\n"
            f"Role: {prospect.get('contact_name', '')} - {prospect.get('contact_role', '')}\n"
            f"Content: {prospect.get('raw_text', '')}\n"
            f"Deterministic score: {deterministic_score}/100\n"
            f"Grounded signals: {len(grounded_signals)} verified signals.\n\n"
            "Provide an LLM qualification score (0-100) and reasoning.\n"
            "Return JSON matching the schema."
        )
        system_prompt = "You are an objective enterprise sales qualification director. Output valid JSON only."

        try:
            qual: QualificationScoreResponse = await self.ollama.generate_json(
                prompt=prompt,
                system_prompt=system_prompt,
                response_model=QualificationScoreResponse,
                workflow_name="prospect.qualification_score",
                model=self.model,
            )
            llm_score = round(qual.llm_score, 2)
            total_score = round((0.5 * deterministic_score) + (0.5 * llm_score), 2)
            priority = "High" if total_score >= 75.0 else ("Medium" if total_score >= 45.0 else "Low")

            score_id = await self.store.save_prospect_score(
                prospect_id=prospect_id,
                product_id=product_id,
                total_score=total_score,
                priority=priority,
                deterministic_score=deterministic_score,
                llm_score=llm_score,
                reasoning=qual.reasoning,
            )
            return {
                "status": "success",
                "score_id": score_id,
                "prospect_id": prospect_id,
                "product_id": product_id,
                "total_score": total_score,
                "priority": priority,
                "deterministic_score": deterministic_score,
                "llm_score": llm_score,
                "reasoning": qual.reasoning,
            }

        except (OllamaUnavailableError, ValidationError):
            total_score = deterministic_score
            priority = "High" if total_score >= 75.0 else ("Medium" if total_score >= 45.0 else "Low")
            reasoning = f"Deterministic scoring fallback applied. Embedding fit={emb_similarity:.2f}, verified signals={len(grounded_signals)}."
            score_id = await self.store.save_prospect_score(
                prospect_id=prospect_id,
                product_id=product_id,
                total_score=total_score,
                priority=priority,
                deterministic_score=deterministic_score,
                llm_score=deterministic_score,
                reasoning=reasoning,
            )
            task_id = await self.store.enqueue_task(
                task_type="prospect_score",
                payload_json={"prospect_id": prospect_id, "product_id": product_id},
            )
            raise OllamaWorkflowError(
                message="Ollama is unavailable. Task queued for local processing.",
                queued_task_id=task_id,
                deterministic_fallback={
                    "total_score": total_score,
                    "priority": priority,
                    "deterministic_score": deterministic_score,
                    "llm_score": deterministic_score,
                    "reasoning": reasoning,
                },
            )

    # -------------------------------------------------------------------------
    # 8. Personalised Outreach Email (Draft Only)
    # -------------------------------------------------------------------------
    async def generate_outreach_email(
        self, prospect_id: int, product_id: int | None = None
    ) -> dict[str, Any]:
        prospect = await self.store.get_prospect(prospect_id)
        if not prospect:
            raise ValueError(f"Prospect {prospect_id} not found.")

        if product_id is None:
            fit = await self.store.get_product_fit(prospect_id)
            if not fit:
                fit = await self.calculate_product_fit(prospect_id)
            product_id = fit["primary_product_id"]

        product = await self.store.get_product(product_id)
        signals = await self.store.list_buying_signals(prospect_id)
        grounded = [s for s in signals if s.get("is_grounded", 0) == 1]
        primary_quote = grounded[0]["evidence_quote"] if grounded else "your recent cloud updates"

        prompt = (
            f"Draft a personalized, high-converting cold outreach email for {prospect['contact_name']} at {prospect['company_name']}.\n\n"
            f"Role: {prospect.get('contact_role', 'Engineering Leader')}\n"
            f"Product: {product['name'] if product else 'Platform'}\n"
            f"Value Proposition: {product.get('value_prop', '') if product else ''}\n"
            f"Verified Buying Signal Evidence: \"{primary_quote}\"\n\n"
            "Requirements:\n"
            "- Reference the specific evidence quote or initiative naturally in opening paragraph.\n"
            "- Connect value proposition directly to their problem.\n"
            "- Keep it under 150 words. Finish with a low-friction question or CTA.\n"
            "Return JSON matching the schema."
        )
        system_prompt = "You are a world-class outbound sales copywriter. Output valid JSON only."

        try:
            email_resp: OutreachEmailResponse = await self.ollama.generate_json(
                prompt=prompt,
                system_prompt=system_prompt,
                response_model=OutreachEmailResponse,
                workflow_name="outreach.email_generation",
                model=self.model,
            )
            email_id = await self.store.save_outreach_email(
                prospect_id=prospect_id,
                product_id=product_id,
                subject=email_resp.subject,
                body=email_resp.body,
                status="draft_only",
            )
            return {
                "status": "success",
                "email_id": email_id,
                "prospect_id": prospect_id,
                "product_id": product_id,
                "subject": email_resp.subject,
                "body": email_resp.body,
                "email_status": "draft_only",
            }

        except (OllamaUnavailableError, ValidationError):
            contact = prospect.get("contact_name") or "there"
            p_name = product["name"] if product else "Vanguard Ops"
            v_prop = product.get("value_prop", "") if product else ""
            fallback_subject = f"{p_name} for {prospect['company_name']}"
            fallback_body = (
                f"Hi {contact},\n\n"
                f"I noticed {prospect['company_name']}'s work in scaling infrastructure, particularly around: \"{primary_quote}\".\n\n"
                f"{v_prop}\n\n"
                "Would you be open to a 10-minute chat this week to share how peers solve this?\n\n"
                "Best,\nVanguard Ops Team"
            )
            email_id = await self.store.save_outreach_email(
                prospect_id=prospect_id,
                product_id=product_id,
                subject=fallback_subject,
                body=fallback_body,
                status="draft_only",
            )
            task_id = await self.store.enqueue_task(
                task_type="outreach_email",
                payload_json={"prospect_id": prospect_id, "product_id": product_id},
            )
            raise OllamaWorkflowError(
                message="Ollama is unavailable. Task queued for local processing.",
                queued_task_id=task_id,
                deterministic_fallback={
                    "subject": fallback_subject,
                    "body": fallback_body,
                    "email_status": "draft_only",
                },
            )

    # -------------------------------------------------------------------------
    # 9. Reply Classification
    # -------------------------------------------------------------------------
    async def classify_reply(
        self, prospect_id: int | None, reply_text: str
    ) -> dict[str, Any]:
        if not reply_text:
            raise ValueError("reply_text cannot be empty.")

        prompt = (
            f"Classify the intent of the following prospect reply email:\n\n"
            f"Reply Content:\n{reply_text}\n\n"
            "Allowed intents: interested, objection, not_now, unsubscribe, out_of_office, referral.\n"
            "Recommend appropriate next_action.\n"
            "Return JSON matching the schema."
        )
        system_prompt = "You are an automated inbound email triage specialist. Output valid JSON only."

        try:
            resp: ReplyClassificationResponse = await self.ollama.generate_json(
                prompt=prompt,
                system_prompt=system_prompt,
                response_model=ReplyClassificationResponse,
                workflow_name="reply.classification",
                model=self.model,
            )
            reply_id = await self.store.save_reply_classification(
                prospect_id=prospect_id,
                reply_text=reply_text,
                intent=resp.intent,
                confidence=resp.confidence,
                next_action=resp.next_action,
                reasoning=resp.reasoning,
            )
            return {
                "status": "success",
                "reply_id": reply_id,
                "prospect_id": prospect_id,
                "intent": resp.intent,
                "confidence": resp.confidence,
                "next_action": resp.next_action,
                "reasoning": resp.reasoning,
            }

        except (OllamaUnavailableError, ValidationError):
            # Deterministic keyword triage
            text_lower = reply_text.lower()
            if any(w in text_lower for w in ["unsubscribe", "remove me", "opt out", "stop emailing"]):
                intent = "unsubscribe"
                confidence = 0.99
                action = "mark_do_not_contact"
                reason = "Explicit unsubscribe request detected."
            elif any(w in text_lower for w in ["out of the office", "on leave", "vacation", "auto-reply", "away until"]):
                intent = "out_of_office"
                confidence = 0.95
                action = "snooze_until_return_date"
                reason = "Out of office auto-reply detected."
            elif any(w in text_lower for w in ["talk to", "reach out to", "refer to", "contact"]) and "@" in text_lower:
                intent = "referral"
                confidence = 0.90
                action = "contact_referral"
                reason = "Referral contact information detected."
            elif any(w in text_lower for w in ["demo", "interested", "15 minutes", "thursday", "call", "schedule"]):
                intent = "interested"
                confidence = 0.90
                action = "send_calendar_invite"
                reason = "Positive interest or meeting request detected."
            elif any(w in text_lower for w in ["not now", "next quarter", "code freeze", "crunch", "later"]):
                intent = "not_now"
                confidence = 0.85
                action = "set_follow_up_timer"
                reason = "Timing objection detected."
            else:
                intent = "objection"
                confidence = 0.70
                action = "send_competitive_comparison"
                reason = "General objection or inquiry detected."

            reply_id = await self.store.save_reply_classification(
                prospect_id=prospect_id,
                reply_text=reply_text,
                intent=intent,
                confidence=confidence,
                next_action=action,
                reasoning=reason,
            )
            task_id = await self.store.enqueue_task(
                task_type="reply_classification",
                payload_json={"prospect_id": prospect_id, "reply_text": reply_text},
            )
            raise OllamaWorkflowError(
                message="Ollama is unavailable. Task queued for local processing.",
                queued_task_id=task_id,
                deterministic_fallback={
                    "intent": intent,
                    "confidence": confidence,
                    "next_action": action,
                    "reasoning": reason,
                },
            )

    # -------------------------------------------------------------------------
    # Queue Draining
    # -------------------------------------------------------------------------
    async def process_queue(self) -> dict[str, Any]:
        """Drain pending tasks from task_queue when Ollama recovers."""
        processed = 0
        failed = 0

        while True:
            task = await self.store.get_next_task()
            if not task:
                break

            task_id = task["id"]
            task_type = task["task_type"]
            payload = json.loads(task["payload_json"])

            try:
                if task_type == "company_profile":
                    await self.generate_company_profile(payload["company_id"])
                elif task_type == "product_icp":
                    await self.generate_product_icp(payload["product_id"])
                elif task_type == "prospect_research":
                    await self.research_prospect(payload["prospect_id"])
                elif task_type == "buying_signals":
                    await self.classify_buying_signals(payload["prospect_id"])
                elif task_type == "problem_statement":
                    await self.generate_problem_statement(
                        payload["prospect_id"], payload.get("product_id")
                    )
                elif task_type == "prospect_score":
                    await self.score_prospect(
                        payload["prospect_id"], payload.get("product_id")
                    )
                elif task_type == "outreach_email":
                    await self.generate_outreach_email(
                        payload["prospect_id"], payload.get("product_id")
                    )
                elif task_type == "reply_classification":
                    await self.classify_reply(
                        payload.get("prospect_id"), payload["reply_text"]
                    )
                else:
                    logger.warning(f"Unknown task type: {task_type}")

                await self.store.complete_task(task_id, status="completed")
                processed += 1

            except Exception as e:
                logger.error(f"Failed processing queue task {task_id}: {e}")
                await self.store.complete_task(task_id, status="failed", error_msg=str(e))
                failed += 1

        return {"processed": processed, "failed": failed}


class OllamaWorkflowError(Exception):
    """Raised when an Ollama-dependent workflow fails due to Ollama being unavailable."""

    def __init__(
        self,
        message: str,
        queued_task_id: int,
        deterministic_fallback: Any = None,
    ):
        super().__init__(message)
        self.message = message
        self.queued_task_id = queued_task_id
        self.deterministic_fallback = deterministic_fallback
