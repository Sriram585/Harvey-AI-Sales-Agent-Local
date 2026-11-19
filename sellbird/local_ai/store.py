"""Persistent SQLite storage for multi-product local sales workflows via aiosqlite."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger("sellbird.local_ai.store")

DEFAULT_DB_PATH = Path("data/local_sales_agent.db")


def _now_iso() -> str:
    """Current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


class LocalSalesStore:
    """Async SQLite data store for multi-product local sales agent workflows."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> aiosqlite.Connection:
        """Establish or return open database connection."""
        if self._db is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = await aiosqlite.connect(self.db_path)
            self._db.row_factory = aiosqlite.Row
        return self._db

    async def close(self):
        """Close database connection if open."""
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def init_db(self):
        """Initialize all 12 database tables."""
        conn = await self.connect()

        schema_statements = [
            # 1. companies
            """
            CREATE TABLE IF NOT EXISTS companies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                website TEXT,
                raw_notes TEXT,
                profile_json TEXT,
                created_at TEXT NOT NULL
            );
            """,
            # 2. products
            """
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                description TEXT,
                value_prop TEXT,
                pricing_tier TEXT,
                created_at TEXT NOT NULL
            );
            """,
            # 3. icps
            """
            CREATE TABLE IF NOT EXISTS icps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
                icp_json TEXT NOT NULL,
                embedding_json TEXT,
                updated_at TEXT NOT NULL
            );
            """,
            # 4. prospects
            """
            CREATE TABLE IF NOT EXISTS prospects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_name TEXT NOT NULL,
                contact_name TEXT,
                contact_role TEXT,
                contact_email TEXT,
                source_url TEXT,
                raw_text TEXT,
                research_json TEXT,
                scraped_at TEXT,
                created_at TEXT NOT NULL
            );
            """,
            # 5. product_fit_matrix
            """
            CREATE TABLE IF NOT EXISTS product_fit_matrix (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prospect_id INTEGER NOT NULL REFERENCES prospects(id) ON DELETE CASCADE,
                primary_product_id INTEGER NOT NULL REFERENCES products(id),
                cross_sell_product_ids_json TEXT,
                scores_json TEXT,
                computed_at TEXT NOT NULL
            );
            """,
            # 6. buying_signals
            """
            CREATE TABLE IF NOT EXISTS buying_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prospect_id INTEGER NOT NULL REFERENCES prospects(id) ON DELETE CASCADE,
                signal_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                evidence_quote TEXT,
                is_grounded INTEGER NOT NULL DEFAULT 1,
                reasoning TEXT,
                created_at TEXT NOT NULL
            );
            """,
            # 7. problem_statements
            """
            CREATE TABLE IF NOT EXISTS problem_statements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prospect_id INTEGER NOT NULL REFERENCES prospects(id) ON DELETE CASCADE,
                product_id INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
                statement TEXT NOT NULL,
                grounding_signals_json TEXT,
                created_at TEXT NOT NULL
            );
            """,
            # 8. prospect_scores
            """
            CREATE TABLE IF NOT EXISTS prospect_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prospect_id INTEGER NOT NULL REFERENCES prospects(id) ON DELETE CASCADE,
                product_id INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
                total_score REAL NOT NULL,
                priority TEXT NOT NULL,
                deterministic_score REAL,
                llm_score REAL,
                reasoning TEXT,
                created_at TEXT NOT NULL
            );
            """,
            # 9. outreach_emails
            """
            CREATE TABLE IF NOT EXISTS outreach_emails (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prospect_id INTEGER NOT NULL REFERENCES prospects(id) ON DELETE CASCADE,
                product_id INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
                subject TEXT NOT NULL,
                body TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'draft',
                created_at TEXT NOT NULL
            );
            """,
            # 10. reply_classifications
            """
            CREATE TABLE IF NOT EXISTS reply_classifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prospect_id INTEGER REFERENCES prospects(id) ON DELETE SET NULL,
                reply_text TEXT NOT NULL,
                intent TEXT NOT NULL,
                confidence REAL NOT NULL,
                next_action TEXT,
                reasoning TEXT,
                created_at TEXT NOT NULL
            );
            """,
            # 11. inference_logs
            """
            CREATE TABLE IF NOT EXISTS inference_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                workflow_name TEXT NOT NULL,
                endpoint_url TEXT NOT NULL,
                model_name TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                eval_count INTEGER NOT NULL,
                tokens_per_sec REAL NOT NULL,
                json_repaired INTEGER NOT NULL,
                status TEXT NOT NULL
            );
            """,
            # 12. task_queue
            """
            CREATE TABLE IF NOT EXISTS task_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                error_msg TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT
            );
            """,
        ]

        for stmt in schema_statements:
            await conn.execute(stmt)
        await conn.commit()

    # -------------------------------------------------------------------------
    # 1. Companies CRUD
    # -------------------------------------------------------------------------
    async def create_company(
        self,
        name: str,
        website: str = "",
        raw_notes: str = "",
        profile_json: str | dict = "{}",
    ) -> int:
        conn = await self.connect()
        if isinstance(profile_json, dict):
            profile_json = json.dumps(profile_json)
        cursor = await conn.execute(
            """
            INSERT INTO companies (name, website, raw_notes, profile_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (name, website, raw_notes, profile_json, _now_iso()),
        )
        await conn.commit()
        return cursor.lastrowid

    async def get_company(self, company_id: int) -> dict[str, Any] | None:
        conn = await self.connect()
        async with conn.execute("SELECT * FROM companies WHERE id = ?", (company_id,)) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def list_companies(self) -> list[dict[str, Any]]:
        conn = await self.connect()
        async with conn.execute("SELECT * FROM companies ORDER BY id ASC") as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    # -------------------------------------------------------------------------
    # 2. Products CRUD
    # -------------------------------------------------------------------------
    async def create_product(
        self,
        company_id: int,
        name: str,
        description: str = "",
        value_prop: str = "",
        pricing_tier: str = "",
    ) -> int:
        conn = await self.connect()
        cursor = await conn.execute(
            """
            INSERT INTO products (company_id, name, description, value_prop, pricing_tier, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (company_id, name, description, value_prop, pricing_tier, _now_iso()),
        )
        await conn.commit()
        return cursor.lastrowid

    async def get_product(self, product_id: int) -> dict[str, Any] | None:
        conn = await self.connect()
        async with conn.execute("SELECT * FROM products WHERE id = ?", (product_id,)) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def list_products(self, company_id: int | None = None) -> list[dict[str, Any]]:
        conn = await self.connect()
        if company_id is not None:
            query = "SELECT * FROM products WHERE company_id = ? ORDER BY id ASC"
            params = (company_id,)
        else:
            query = "SELECT * FROM products ORDER BY id ASC"
            params = ()
        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    # -------------------------------------------------------------------------
    # 3. ICPs CRUD
    # -------------------------------------------------------------------------
    async def save_icp(
        self,
        product_id: int,
        icp_json: str | dict,
        embedding_json: str | list | None = None,
    ) -> int:
        conn = await self.connect()
        if isinstance(icp_json, dict):
            icp_json = json.dumps(icp_json)
        if isinstance(embedding_json, list):
            embedding_json = json.dumps(embedding_json)

        cursor = await conn.execute(
            """
            INSERT INTO icps (product_id, icp_json, embedding_json, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (product_id, icp_json, embedding_json, _now_iso()),
        )
        await conn.commit()
        return cursor.lastrowid

    async def get_icp(self, product_id: int) -> dict[str, Any] | None:
        conn = await self.connect()
        async with conn.execute(
            "SELECT * FROM icps WHERE product_id = ? ORDER BY id DESC LIMIT 1", (product_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def list_icps(self) -> list[dict[str, Any]]:
        conn = await self.connect()
        async with conn.execute("SELECT * FROM icps ORDER BY id ASC") as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    # -------------------------------------------------------------------------
    # 4. Prospects CRUD
    # -------------------------------------------------------------------------
    async def create_prospect(
        self,
        company_name: str,
        contact_name: str = "",
        contact_role: str = "",
        contact_email: str = "",
        source_url: str = "",
        raw_text: str = "",
        research_json: str | dict = "{}",
        scraped_at: str | None = None,
    ) -> int:
        conn = await self.connect()
        if isinstance(research_json, dict):
            research_json = json.dumps(research_json)
        cursor = await conn.execute(
            """
            INSERT INTO prospects (
                company_name, contact_name, contact_role, contact_email,
                source_url, raw_text, research_json, scraped_at, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                company_name,
                contact_name,
                contact_role,
                contact_email,
                source_url,
                raw_text,
                research_json,
                scraped_at or _now_iso(),
                _now_iso(),
            ),
        )
        await conn.commit()
        return cursor.lastrowid

    async def get_prospect(self, prospect_id: int) -> dict[str, Any] | None:
        conn = await self.connect()
        async with conn.execute("SELECT * FROM prospects WHERE id = ?", (prospect_id,)) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def list_prospects(self) -> list[dict[str, Any]]:
        conn = await self.connect()
        async with conn.execute("SELECT * FROM prospects ORDER BY id ASC") as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def update_prospect(self, prospect_id: int, **kwargs) -> bool:
        if not kwargs:
            return False
        conn = await self.connect()
        fields = []
        values = []
        for k, v in kwargs.items():
            if isinstance(v, (dict, list)):
                v = json.dumps(v)
            fields.append(f"{k} = ?")
            values.append(v)
        values.append(prospect_id)
        stmt = f"UPDATE prospects SET {', '.join(fields)} WHERE id = ?"
        await conn.execute(stmt, values)
        await conn.commit()
        return True

    # -------------------------------------------------------------------------
    # 5. Product Fit Matrix CRUD
    # -------------------------------------------------------------------------
    async def save_product_fit(
        self,
        prospect_id: int,
        primary_product_id: int,
        cross_sell_product_ids_json: str | list = "[]",
        scores_json: str | dict = "{}",
    ) -> int:
        conn = await self.connect()
        if isinstance(cross_sell_product_ids_json, list):
            cross_sell_product_ids_json = json.dumps(cross_sell_product_ids_json)
        if isinstance(scores_json, dict):
            scores_json = json.dumps(scores_json)

        cursor = await conn.execute(
            """
            INSERT INTO product_fit_matrix (
                prospect_id, primary_product_id, cross_sell_product_ids_json, scores_json, computed_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (prospect_id, primary_product_id, cross_sell_product_ids_json, scores_json, _now_iso()),
        )
        await conn.commit()
        return cursor.lastrowid

    async def get_product_fit(self, prospect_id: int) -> dict[str, Any] | None:
        conn = await self.connect()
        async with conn.execute(
            "SELECT * FROM product_fit_matrix WHERE prospect_id = ? ORDER BY id DESC LIMIT 1",
            (prospect_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    # -------------------------------------------------------------------------
    # 6. Buying Signals CRUD
    # -------------------------------------------------------------------------
    async def add_buying_signal(
        self,
        prospect_id: int,
        signal_type: str,
        confidence: float,
        evidence_quote: str = "",
        is_grounded: bool = True,
        reasoning: str = "",
    ) -> int:
        conn = await self.connect()
        cursor = await conn.execute(
            """
            INSERT INTO buying_signals (
                prospect_id, signal_type, confidence, evidence_quote, is_grounded, reasoning, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prospect_id,
                signal_type,
                confidence,
                evidence_quote,
                1 if is_grounded else 0,
                reasoning,
                _now_iso(),
            ),
        )
        await conn.commit()
        return cursor.lastrowid

    async def list_buying_signals(self, prospect_id: int) -> list[dict[str, Any]]:
        conn = await self.connect()
        async with conn.execute(
            "SELECT * FROM buying_signals WHERE prospect_id = ? ORDER BY id ASC", (prospect_id,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    # -------------------------------------------------------------------------
    # 7. Problem Statements CRUD
    # -------------------------------------------------------------------------
    async def save_problem_statement(
        self,
        prospect_id: int,
        product_id: int,
        statement: str,
        grounding_signals_json: str | list = "[]",
    ) -> int:
        conn = await self.connect()
        if isinstance(grounding_signals_json, list):
            grounding_signals_json = json.dumps(grounding_signals_json)

        cursor = await conn.execute(
            """
            INSERT INTO problem_statements (
                prospect_id, product_id, statement, grounding_signals_json, created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (prospect_id, product_id, statement, grounding_signals_json, _now_iso()),
        )
        await conn.commit()
        return cursor.lastrowid

    async def list_problem_statements(
        self, prospect_id: int, product_id: int | None = None
    ) -> list[dict[str, Any]]:
        conn = await self.connect()
        if product_id is not None:
            query = "SELECT * FROM problem_statements WHERE prospect_id = ? AND product_id = ? ORDER BY id ASC"
            params = (prospect_id, product_id)
        else:
            query = "SELECT * FROM problem_statements WHERE prospect_id = ? ORDER BY id ASC"
            params = (prospect_id,)
        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    # -------------------------------------------------------------------------
    # 8. Prospect Scores CRUD
    # -------------------------------------------------------------------------
    async def save_prospect_score(
        self,
        prospect_id: int,
        product_id: int,
        total_score: float,
        priority: str,
        deterministic_score: float = 0.0,
        llm_score: float = 0.0,
        reasoning: str = "",
    ) -> int:
        conn = await self.connect()
        cursor = await conn.execute(
            """
            INSERT INTO prospect_scores (
                prospect_id, product_id, total_score, priority, deterministic_score, llm_score, reasoning, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prospect_id,
                product_id,
                total_score,
                priority,
                deterministic_score,
                llm_score,
                reasoning,
                _now_iso(),
            ),
        )
        await conn.commit()
        return cursor.lastrowid

    async def get_prospect_scores(
        self, prospect_id: int, product_id: int | None = None
    ) -> list[dict[str, Any]]:
        conn = await self.connect()
        if product_id is not None:
            query = "SELECT * FROM prospect_scores WHERE prospect_id = ? AND product_id = ? ORDER BY id DESC"
            params = (prospect_id, product_id)
        else:
            query = "SELECT * FROM prospect_scores WHERE prospect_id = ? ORDER BY id DESC"
            params = (prospect_id,)
        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    # -------------------------------------------------------------------------
    # 9. Outreach Emails CRUD
    # -------------------------------------------------------------------------
    async def save_outreach_email(
        self,
        prospect_id: int,
        product_id: int,
        subject: str,
        body: str,
        status: str = "draft",
    ) -> int:
        conn = await self.connect()
        cursor = await conn.execute(
            """
            INSERT INTO outreach_emails (
                prospect_id, product_id, subject, body, status, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (prospect_id, product_id, subject, body, status, _now_iso()),
        )
        await conn.commit()
        return cursor.lastrowid

    async def list_outreach_emails(
        self, prospect_id: int | None = None, status: str | None = None
    ) -> list[dict[str, Any]]:
        conn = await self.connect()
        clauses = []
        params = []
        if prospect_id is not None:
            clauses.append("prospect_id = ?")
            params.append(prospect_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)

        query = "SELECT * FROM outreach_emails"
        if clauses:
            query += f" WHERE {' AND '.join(clauses)}"
        query += " ORDER BY id DESC"

        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def update_email_status(self, email_id: int, status: str) -> bool:
        conn = await self.connect()
        await conn.execute(
            "UPDATE outreach_emails SET status = ? WHERE id = ?", (status, email_id)
        )
        await conn.commit()
        return True

    # -------------------------------------------------------------------------
    # 10. Reply Classifications CRUD
    # -------------------------------------------------------------------------
    async def save_reply_classification(
        self,
        prospect_id: int | None,
        reply_text: str,
        intent: str,
        confidence: float,
        next_action: str = "",
        reasoning: str = "",
    ) -> int:
        conn = await self.connect()
        cursor = await conn.execute(
            """
            INSERT INTO reply_classifications (
                prospect_id, reply_text, intent, confidence, next_action, reasoning, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (prospect_id, reply_text, intent, confidence, next_action, reasoning, _now_iso()),
        )
        await conn.commit()
        return cursor.lastrowid

    async def list_reply_classifications(
        self, prospect_id: int | None = None
    ) -> list[dict[str, Any]]:
        conn = await self.connect()
        if prospect_id is not None:
            query = "SELECT * FROM reply_classifications WHERE prospect_id = ? ORDER BY id DESC"
            params = (prospect_id,)
        else:
            query = "SELECT * FROM reply_classifications ORDER BY id DESC"
            params = ()
        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    # -------------------------------------------------------------------------
    # 11. Inference Logs CRUD & Callback
    # -------------------------------------------------------------------------
    async def log_inference(
        self,
        record: dict[str, Any] | None = None,
        *,
        timestamp: str | None = None,
        workflow_name: str = "",
        endpoint_url: str = "",
        model_name: str = "",
        latency_ms: float = 0.0,
        eval_count: int = 0,
        tokens_per_sec: float = 0.0,
        json_repaired: bool = False,
        status: str = "success",
    ) -> int:
        """Log inference either from an OllamaClient dict callback or individual parameters."""
        conn = await self.connect()
        if record is not None:
            ts = record.get("timestamp", _now_iso())
            wf = record.get("workflow_name", "")
            ep = record.get("endpoint_url", "")
            m = record.get("model_name", "")
            lat = float(record.get("latency_ms", 0.0))
            ec = int(record.get("eval_count", 0))
            tps = float(record.get("tokens_per_sec", 0.0))
            rep = 1 if record.get("json_repaired", False) else 0
            st = record.get("status", "success")
        else:
            ts = timestamp or _now_iso()
            wf = workflow_name
            ep = endpoint_url
            m = model_name
            lat = latency_ms
            ec = eval_count
            tps = tokens_per_sec
            rep = 1 if json_repaired else 0
            st = status

        cursor = await conn.execute(
            """
            INSERT INTO inference_logs (
                timestamp, workflow_name, endpoint_url, model_name,
                latency_ms, eval_count, tokens_per_sec, json_repaired, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ts, wf, ep, m, lat, ec, tps, rep, st),
        )
        await conn.commit()
        return cursor.lastrowid

    async def list_inference_logs(self, limit: int = 100) -> list[dict[str, Any]]:
        conn = await self.connect()
        async with conn.execute(
            "SELECT * FROM inference_logs ORDER BY id DESC LIMIT ?", (limit,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    # -------------------------------------------------------------------------
    # 12. Task Queue CRUD
    # -------------------------------------------------------------------------
    async def enqueue_task(self, task_type: str, payload_json: str | dict) -> int:
        conn = await self.connect()
        if isinstance(payload_json, dict):
            payload_json = json.dumps(payload_json)
        cursor = await conn.execute(
            """
            INSERT INTO task_queue (task_type, payload_json, status, created_at)
            VALUES (?, ?, 'pending', ?)
            """,
            (task_type, payload_json, _now_iso()),
        )
        await conn.commit()
        return cursor.lastrowid

    async def get_next_task(self) -> dict[str, Any] | None:
        conn = await self.connect()
        async with conn.execute(
            "SELECT * FROM task_queue WHERE status = 'pending' ORDER BY id ASC LIMIT 1"
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            task = dict(row)
            await conn.execute(
                "UPDATE task_queue SET status = 'running' WHERE id = ?", (task["id"],)
            )
            await conn.commit()
            task["status"] = "running"
            return task

    async def complete_task(
        self, task_id: int, status: str = "completed", error_msg: str | None = None
    ) -> bool:
        conn = await self.connect()
        await conn.execute(
            """
            UPDATE task_queue
            SET status = ?, error_msg = ?, completed_at = ?
            WHERE id = ?
            """,
            (status, error_msg, _now_iso(), task_id),
        )
        await conn.commit()
        return True
