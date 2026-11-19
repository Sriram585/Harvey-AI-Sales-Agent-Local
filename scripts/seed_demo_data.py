"""Idempotent seed script for local demo data.

Populates data/local_sales_agent.db with:
- 1 B2B SaaS Company (Vanguard Ops AI)
- 3 Distinct Products (CloudCost Guard, PipeGuard, ShiftSync)
- 6 Pre-scraped prospects with rich raw_text containing buying cues
- 6 Realistic prospect replies covering all standard intents
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sellbird.local_ai.store import DEFAULT_DB_PATH, LocalSalesStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("seed_demo_data")


COMPANY_DATA = {
    "name": "Vanguard Ops AI",
    "website": "https://vanguardops.ai",
    "raw_notes": (
        "Vanguard Ops AI is an enterprise-grade AI infrastructure and developer operations platform founded in 2023. "
        "We provide deep observability, autonomous cloud cost governance, CI/CD supply chain security, and developer "
        "productivity analytics for high-growth tech companies, fintechs, and modern enterprise software organizations. "
        "Our mission is to eliminate cloud waste, prevent CI/CD supply-chain attacks, and unblock engineering bottlenecks "
        "without disrupting developer flow."
    ),
    "profile_json": {
        "industry": "Enterprise DevSecOps & Cloud Infrastructure",
        "target_customers": [
            "Mid-market to enterprise tech companies",
            "Fintech",
            "Healthtech",
            "High-growth SaaS",
        ],
        "headquarters": "San Francisco, CA",
        "key_competitors": ["Datadog", "Snyk", "Kubecost", "Jellyfish"],
        "unique_selling_proposition": (
            "Unified agentic operations that autonomously identify FinOps waste, protect build pipelines, "
            "and uncover delivery friction using local, privacy-first models."
        ),
    },
}

PRODUCTS_DATA = [
    {
        "name": "CloudCost Guard",
        "description": "Autonomous cloud spend and FinOps governance engine for AWS, GCP, and Kubernetes clusters.",
        "value_prop": "Real-time anomaly detection, rightsizing recommendations, and automated idle resource shutdown that slashes cloud bills by 30-45% with zero downtime.",
        "pricing_tier": "$1,500/mo - $5,000/mo (Enterprise FinOps)",
    },
    {
        "name": "PipeGuard",
        "description": "Next-generation CI/CD security and software supply-chain protection platform.",
        "value_prop": "Real-time dependency vulnerability scanning, zero-day secret leakage prevention in GitHub Actions and GitLab CI, and automated SOC2/FedRAMP compliance audit trails.",
        "pricing_tier": "$999/mo - $3,500/mo (Unlimited Repos & Compliance)",
    },
    {
        "name": "ShiftSync",
        "description": "Engineering analytics and velocity intelligence platform for distributed engineering teams.",
        "value_prop": "Tracks cycle time, PR review bottlenecks, deployment frequency, and developer burnout indicators without intrusive surveillance or micromanagement.",
        "pricing_tier": "$18 - $35 / seat / month (Enterprise Velocity)",
    },
]

PROSPECTS_DATA = [
    {
        "company_name": "ApexFin Systems",
        "contact_name": "Sarah Jenkins",
        "contact_role": "VP of Infrastructure & Cloud",
        "contact_email": "sarah.jenkins@apexfin.io",
        "source_url": "https://apexfin.io/news/series-b",
        "raw_text": (
            "ApexFin Systems announced a $45M Series B funding round to scale its real-time payment rails. "
            "Sarah Jenkins, VP of Infrastructure, noted: 'As our transaction volume expanded 400% over the last year, "
            "our AWS and multi-region Kubernetes cloud bills skyrocketed beyond budget. Our team is actively seeking automated "
            "FinOps tools to reign in cloud infrastructure costs while preparing for SOC2 Type II audits. We are aggressively "
            "hiring 25 DevOps engineers across North America.'"
        ),
        "research_json": {
            "funding_stage": "Series B ($45M)",
            "cloud_provider": "AWS / EKS",
            "key_pains": ["Skyrocketing cloud bills", "SOC2 compliance audit", "Rapid DevOps hiring"],
        },
    },
    {
        "company_name": "NovaPay Technologies",
        "contact_name": "David Zhang",
        "contact_role": "Head of Security & DevSecOps",
        "contact_email": "david.zhang@novapay.com",
        "source_url": "https://novapay.com/security/update-q3",
        "raw_text": (
            "NovaPay Technologies recently suffered an attempted dependency confusion attack on its npm build pipelines. "
            "David Zhang, Head of Security, published an advisory stating: 'Securing our CI/CD pipelines, preventing secret leakage "
            "in GitHub workflows, and establishing automated software supply chain provenance are our #1 top priority for Q3. "
            "We are consolidating security vendors to ensure FedRAMP and PCI-DSS Level 1 compliance before our European expansion.'"
        ),
        "research_json": {
            "focus_area": "CI/CD Security",
            "incident": "npm dependency confusion attack",
            "compliance": ["PCI-DSS Level 1", "FedRAMP"],
        },
    },
    {
        "company_name": "OmniTrack Logistics",
        "contact_name": "Marcus Vance",
        "contact_role": "VP of Engineering",
        "contact_email": "marcus.vance@omnitracklogistics.com",
        "source_url": "https://omnitracklogistics.com/engineering/scaling-remote",
        "raw_text": (
            "OmniTrack Logistics transitioned to a 100% remote global engineering team across 8 time zones with 180 engineers. "
            "Marcus Vance, VP of Engineering, shared on his tech blog: 'Our cycle time has stretched from 2 days to 11 days. "
            "Pull requests are sitting idle waiting for reviews, causing engineering leadership to lose visibility into pipeline "
            "blockers and developer burnout. We need an engineering intelligence platform to measure PR velocity and DORA metrics "
            "without micromanaging devs.'"
        ),
        "research_json": {
            "team_size": 180,
            "work_model": "100% Remote",
            "pains": ["Cycle time 2d -> 11d", "PR review latency", "Lack of DORA metrics visibility"],
        },
    },
    {
        "company_name": "Zenith Health",
        "contact_name": "Elena Rostova",
        "contact_role": "Chief Technology Officer",
        "contact_email": "elena.rostova@zenithhealth.org",
        "source_url": "https://zenithhealth.org/press/cloud-migration",
        "raw_text": (
            "Zenith Health is migrating its core electronic health record platform to Google Cloud and Kubernetes. "
            "Elena Rostova, CTO, remarked: 'Our cloud migration has quadrupled our monthly GCP expenditure, and HIPAA compliance "
            "mandates that our deployment pipelines have rigorous vulnerability scanning and zero secret leaks. We need tooling "
            "that solves both our spiraling cloud compute waste and hardens our deployment automation.'"
        ),
        "research_json": {
            "cloud_provider": "GCP / GKE",
            "compliance": "HIPAA",
            "pains": ["Quadrupled GCP bill", "Vulnerability scanning in CI/CD"],
        },
    },
    {
        "company_name": "KiteCommerce",
        "contact_name": "Alex Rivera",
        "contact_role": "Director of Platform Engineering",
        "contact_email": "alex.rivera@kitecommerce.net",
        "source_url": "https://kitecommerce.net/blog/merging-stacks",
        "raw_text": (
            "KiteCommerce acquired two competing storefront engines, resulting in 3 separate engineering stacks and ballooning "
            "AWS infrastructure costs. Alex Rivera, Director of Platform Engineering, reported: 'Integrating 120 engineers across "
            "different repositories has wrecked our deployment frequency. Furthermore, duplicate cloud resources and unattached "
            "EBS volumes are costing us over $80,000 every single month in pure waste. We are looking for immediate remedies.'"
        ),
        "research_json": {
            "team_size": 120,
            "cloud_waste_monthly": "$80,000",
            "pains": ["Duplicate AWS resources", "Deployment frequency collapse"],
        },
    },
    {
        "company_name": "CyberSentinel Labs",
        "contact_name": "Rachel Chen",
        "contact_role": "Lead DevOps Architect",
        "contact_email": "rachel.chen@cybersentinellabs.io",
        "source_url": "https://cybersentinellabs.io/posts/gitlab-ci-security",
        "raw_text": (
            "CyberSentinel Labs expanded its product portfolio to support ISO 27001 and FedRAMP high benchmarks. "
            "Rachel Chen, Lead DevOps Architect, indicated: 'Our CI/CD build scripts in GitLab are vulnerable to supply-chain "
            "tampering and lack automated policy enforcement. We must implement automated build attestations and secret scanning "
            "across all internal microservices before our enterprise audit next month.'"
        ),
        "research_json": {
            "pipeline": "GitLab CI",
            "compliance": ["ISO 27001", "FedRAMP High"],
            "pains": ["Supply-chain tampering", "Secret scanning"],
        },
    },
]

REPLIES_DATA = [
    {
        "intent": "interested",
        "confidence": 0.96,
        "reply_text": (
            "Hi team, this sounds very relevant to what we are facing right now with our AWS spend. "
            "Our cloud bills surged after the recent cluster migration. Do you have 15 minutes this Thursday at 2 PM PT to show a demo?"
        ),
        "next_action": "send_calendar_invite",
        "reasoning": "Prospect explicitly confirmed interest in AWS cost reduction and proposed a specific demo time.",
    },
    {
        "intent": "objection",
        "confidence": 0.88,
        "reply_text": (
            "Thanks for reaching out. We already looked at Kubecost and found it too complex and noisy for our engineers to adopt. "
            "How is CloudCost Guard different in terms of automation and false alerts?"
        ),
        "next_action": "send_competitive_comparison",
        "reasoning": "Prospect expressed skepticism based on past negative experience with Kubecost complexity.",
    },
    {
        "intent": "not_now",
        "confidence": 0.92,
        "reply_text": (
            "We are in the middle of our annual code freeze and SOC2 audit crunch right now. "
            "Please circle back with me at the start of Q1 next year when our budget opens up."
        ),
        "next_action": "set_follow_up_timer",
        "reasoning": "Timing objection due to code freeze / audit crunch; requested follow-up at start of Q1.",
    },
    {
        "intent": "unsubscribe",
        "confidence": 0.99,
        "reply_text": "Please remove me and our domain from your mailing list immediately. Unsubscribe.",
        "next_action": "mark_do_not_contact",
        "reasoning": "Explicit opt-out request.",
    },
    {
        "intent": "out_of_office",
        "confidence": 0.95,
        "reply_text": (
            "Thank you for your message. I am out of the office on annual leave through November 28th with limited access to email. "
            "For urgent production incidents, please contact oncall@apexfin.io."
        ),
        "next_action": "snooze_until_return_date",
        "reasoning": "Automated out of office auto-reply.",
    },
    {
        "intent": "referral",
        "confidence": 0.94,
        "reply_text": (
            "I am no longer handling infrastructure procurement directly. "
            "You should talk to our VP of Cloud Operations, Liam Ward (liam.ward@apexfin.io)—he owns this tooling budget."
        ),
        "next_action": "contact_referral",
        "reasoning": "Referred to colleague Liam Ward with contact information provided.",
    },
]


async def seed_demo_data(db_path: Path | str = DEFAULT_DB_PATH):
    """Seed the database idempotently with rich demo data."""
    store = LocalSalesStore(db_path)
    await store.init_db()

    conn = await store.connect()

    # Check if company already exists to maintain idempotency
    async with conn.execute("SELECT id FROM companies WHERE name = ?", (COMPANY_DATA["name"],)) as cur:
        existing_company = await cur.fetchone()

    if existing_company:
        company_id = existing_company["id"]
        logger.info(f"Existing company found (id={company_id}), refreshing seed data...")
        # Clean up existing demo products and prospects to ensure clean seed state
        await conn.execute("DELETE FROM reply_classifications")
        await conn.execute("DELETE FROM prospect_scores")
        await conn.execute("DELETE FROM problem_statements")
        await conn.execute("DELETE FROM buying_signals")
        await conn.execute("DELETE FROM product_fit_matrix")
        await conn.execute("DELETE FROM outreach_emails")
        await conn.execute("DELETE FROM prospects")
        await conn.execute("DELETE FROM icps")
        await conn.execute("DELETE FROM products")
        await conn.execute("DELETE FROM companies WHERE id = ?", (company_id,))
        await conn.commit()

    # 1. Insert Company
    company_id = await store.create_company(
        name=COMPANY_DATA["name"],
        website=COMPANY_DATA["website"],
        raw_notes=COMPANY_DATA["raw_notes"],
        profile_json=COMPANY_DATA["profile_json"],
    )
    logger.info(f"Seeded company: {COMPANY_DATA['name']} (id={company_id})")

    # 2. Insert Products
    product_ids = []
    for prod in PRODUCTS_DATA:
        p_id = await store.create_product(
            company_id=company_id,
            name=prod["name"],
            description=prod["description"],
            value_prop=prod["value_prop"],
            pricing_tier=prod["pricing_tier"],
        )
        product_ids.append(p_id)
        logger.info(f"  - Seeded product: {prod['name']} (id={p_id})")

    # 3. Insert Prospects
    prospect_ids = []
    for p_data in PROSPECTS_DATA:
        pr_id = await store.create_prospect(
            company_name=p_data["company_name"],
            contact_name=p_data["contact_name"],
            contact_role=p_data["contact_role"],
            contact_email=p_data["contact_email"],
            source_url=p_data["source_url"],
            raw_text=p_data["raw_text"],
            research_json=p_data["research_json"],
        )
        prospect_ids.append(pr_id)
        logger.info(f"  - Seeded prospect: {p_data['company_name']} / {p_data['contact_name']} (id={pr_id})")

    # 4. Insert Sample Replies
    for i, rep in enumerate(REPLIES_DATA):
        assigned_prospect_id = prospect_ids[i % len(prospect_ids)]
        rep_id = await store.save_reply_classification(
            prospect_id=assigned_prospect_id,
            reply_text=rep["reply_text"],
            intent=rep["intent"],
            confidence=rep["confidence"],
            next_action=rep["next_action"],
            reasoning=rep["reasoning"],
        )
        logger.info(f"  - Seeded reply [{rep['intent']}] for prospect id={assigned_prospect_id} (id={rep_id})")

    await store.close()
    logger.info("Demo data seeding completed successfully.")


if __name__ == "__main__":
    asyncio.run(seed_demo_data())
