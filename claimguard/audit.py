"""Append-only, hash-chained decision log in SQLite.

Two guarantees:
  * Append-only: SQLite triggers reject every UPDATE and DELETE. Investigator
    outcomes go into a separate `reviews` table instead of editing a decision.
  * Tamper-evident: each row stores the SHA-256 of its own contents plus the
    previous row's hash. Changing or removing a row breaks the chain, which
    `python -m claimguard.audit verify` reports.

Limit: someone with write access to the file could rebuild the whole chain after
an edit. Publishing the latest hash somewhere else (a ticket, a daily email)
closes that gap; `head` prints it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

GENESIS = "0" * 64
OUTCOMES = ("CONFIRMED_FRAUD", "CLEARED")

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    seq                INTEGER PRIMARY KEY,
    decision_id        TEXT NOT NULL UNIQUE,
    created_at         TEXT NOT NULL,
    claim_json         TEXT NOT NULL,
    score              REAL NOT NULL,
    band               TEXT NOT NULL,
    rule_hits_json     TEXT NOT NULL,
    model_reasons_json TEXT NOT NULL,
    model_version      TEXT NOT NULL,
    config_hash        TEXT NOT NULL,
    prev_hash          TEXT NOT NULL,
    row_hash           TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reviews (
    seq          INTEGER PRIMARY KEY,
    review_id    TEXT NOT NULL UNIQUE,
    decision_id  TEXT NOT NULL REFERENCES decisions(decision_id),
    created_at   TEXT NOT NULL,
    outcome      TEXT NOT NULL CHECK (outcome IN ('CONFIRMED_FRAUD', 'CLEARED')),
    reviewer     TEXT NOT NULL,
    notes        TEXT NOT NULL,
    prev_hash    TEXT NOT NULL,
    row_hash     TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS decisions_no_update BEFORE UPDATE ON decisions
BEGIN SELECT RAISE(ABORT, 'decisions is append-only'); END;
CREATE TRIGGER IF NOT EXISTS decisions_no_delete BEFORE DELETE ON decisions
BEGIN SELECT RAISE(ABORT, 'decisions is append-only'); END;
CREATE TRIGGER IF NOT EXISTS reviews_no_update BEFORE UPDATE ON reviews
BEGIN SELECT RAISE(ABORT, 'reviews is append-only'); END;
CREATE TRIGGER IF NOT EXISTS reviews_no_delete BEFORE DELETE ON reviews
BEGIN SELECT RAISE(ABORT, 'reviews is append-only'); END;
"""

DECISION_FIELDS = ["seq", "decision_id", "created_at", "claim_json", "score", "band", "rule_hits_json",
                   "model_reasons_json", "model_version", "config_hash", "prev_hash"]
REVIEW_FIELDS = ["seq", "review_id", "decision_id", "created_at", "outcome", "reviewer", "notes", "prev_hash"]
TABLES = {"decisions": DECISION_FIELDS, "reviews": REVIEW_FIELDS}


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def row_hash(table: str, row: dict[str, Any]) -> str:
    payload = {"table": table, **{k: row[k] for k in TABLES[table]}}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class AuditLog:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, isolation_level=None, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.close()

    def _append(self, table: str, row: dict[str, Any], make_id) -> dict[str, Any]:
        return self._append_many(table, [row], make_id)[0]

    def _append_many(self, table: str, rows: list[dict[str, Any]], make_id) -> list[dict[str, Any]]:
        """Insert rows in one transaction under a write lock, so the chain cannot fork.

        All rows are written or none are.
        """
        cols = TABLES[table] + ["row_hash"]
        sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                last = conn.execute(f"SELECT seq, row_hash FROM {table} ORDER BY seq DESC LIMIT 1").fetchone()
                seq, prev = (last["seq"], last["row_hash"]) if last else (0, GENESIS)
                for row in rows:
                    seq += 1
                    row["seq"], row["prev_hash"] = seq, prev
                    row.update(make_id(seq))
                    row["row_hash"] = prev = row_hash(table, row)
                    conn.execute(sql, [row[c] for c in cols])
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return rows

    # ---------- writes ----------

    def record_decision(self, claim: dict, score: float, band: str, rule_hits: list[dict],
                        model_reasons: list[dict], model_version: str, config_hash: str) -> dict[str, Any]:
        return self.record_decisions([{"claim": claim, "score": score, "band": band, "rule_hits": rule_hits,
                                       "model_reasons": model_reasons}], model_version, config_hash)[0]

    def record_decisions(self, items: list[dict], model_version: str, config_hash: str) -> list[dict[str, Any]]:
        """Log many decisions in one transaction. Each item has claim, score, band, rule_hits, model_reasons."""
        created = _now()
        rows = [{
            "created_at": created,
            "claim_json": canonical_json(it["claim"]),
            "score": round(float(it["score"]), 6),
            "band": it["band"],
            "rule_hits_json": canonical_json(it["rule_hits"]),
            "model_reasons_json": canonical_json(it["model_reasons"]),
            "model_version": model_version,
            "config_hash": config_hash,
        } for it in items]
        return self._append_many("decisions", rows,
                                 lambda seq: {"decision_id": f"d-{created[:4]}-{seq:06d}"})

    def add_review(self, decision_id: str, outcome: str, reviewer: str, notes: str = "") -> dict[str, Any]:
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}")
        if self.get_decision(decision_id) is None:
            raise KeyError(decision_id)
        row = {"decision_id": decision_id, "created_at": _now(), "outcome": outcome,
               "reviewer": reviewer, "notes": notes}
        return self._append("reviews", row, lambda seq: {"review_id": f"r-{seq:06d}"})

    # ---------- reads ----------

    def get_decision(self, decision_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM decisions WHERE decision_id = ?", (decision_id,)).fetchone()
            if row is None:
                return None
            reviews = conn.execute("SELECT review_id, created_at, outcome, reviewer, notes FROM reviews "
                                   "WHERE decision_id = ? ORDER BY seq", (decision_id,)).fetchall()
        d = dict(row)
        return {
            "decision_id": d["decision_id"],
            "created_at": d["created_at"],
            "claim": json.loads(d["claim_json"]),
            "score": d["score"],
            "band": d["band"],
            "rule_hits": json.loads(d["rule_hits_json"]),
            "model_reasons": json.loads(d["model_reasons_json"]),
            "model_version": d["model_version"],
            "config_hash": d["config_hash"],
            "row_hash": d["row_hash"],
            "prev_hash": d["prev_hash"],
            "reviews": [dict(r) for r in reviews],
        }

    def list_decisions(self, bands: list[str] | None = None, reviewed: bool | None = None,
                       sort: str = "score", limit: int = 100) -> list[dict[str, Any]]:
        """Review queue: decision summaries with their latest review, if any."""
        where, params = [], []
        if bands:
            where.append(f"d.band IN ({', '.join('?' * len(bands))})")
            params += bands
        if reviewed is not None:
            where.append("l.outcome IS NOT NULL" if reviewed else "l.outcome IS NULL")
        order = {"score": "d.score DESC, d.seq DESC", "recent": "d.seq DESC"}[sort]
        query = f"""
        WITH latest AS (
            SELECT r.decision_id, r.outcome, r.reviewer FROM reviews r
            JOIN (SELECT decision_id, MAX(seq) AS seq FROM reviews GROUP BY decision_id) m ON r.seq = m.seq
        )
        SELECT d.decision_id, d.created_at, d.score, d.band, d.claim_json, d.rule_hits_json,
               l.outcome, l.reviewer
        FROM decisions d LEFT JOIN latest l ON l.decision_id = d.decision_id
        {"WHERE " + " AND ".join(where) if where else ""}
        ORDER BY {order} LIMIT ?
        """
        with self._connect() as conn:
            rows = conn.execute(query, [*params, limit]).fetchall()
        return [{"decision_id": r["decision_id"], "created_at": r["created_at"], "score": r["score"],
                 "band": r["band"], "claim_ref": json.loads(r["claim_json"]).get("claim_ref"),
                 "rules_hit": len(json.loads(r["rule_hits_json"])),
                 "latest_outcome": r["outcome"], "reviewer": r["reviewer"]} for r in rows]

    def metrics(self) -> dict[str, Any]:
        """Counts per band, and confirmed-fraud rate among reviewed claims (latest review wins)."""
        query = """
        WITH latest AS (
            SELECT r.decision_id, r.outcome FROM reviews r
            JOIN (SELECT decision_id, MAX(seq) AS seq FROM reviews GROUP BY decision_id) m
              ON r.seq = m.seq
        )
        SELECT d.band,
               COUNT(*) AS decisions,
               SUM(l.outcome IS NOT NULL) AS reviewed,
               SUM(l.outcome = 'CONFIRMED_FRAUD') AS confirmed_fraud
        FROM decisions d LEFT JOIN latest l ON l.decision_id = d.decision_id
        GROUP BY d.band
        """
        with self._connect() as conn:
            rows = {r["band"]: dict(r) for r in conn.execute(query)}
        out = {}
        for band in ("LOW", "MEDIUM", "HIGH"):
            r = rows.get(band, {"decisions": 0, "reviewed": 0, "confirmed_fraud": 0})
            reviewed = r["reviewed"] or 0
            confirmed = r["confirmed_fraud"] or 0
            out[band] = {"decisions": r["decisions"], "reviewed": reviewed, "confirmed_fraud": confirmed,
                         "confirmed_fraud_rate": (confirmed / reviewed) if reviewed else None}
        return {"bands": out, "total_decisions": sum(v["decisions"] for v in out.values())}

    def head(self) -> dict[str, str]:
        with self._connect() as conn:
            return {t: (conn.execute(f"SELECT row_hash FROM {t} ORDER BY seq DESC LIMIT 1").fetchone()
                        or {"row_hash": GENESIS})["row_hash"] for t in TABLES}

    # ---------- verification ----------

    def verify(self) -> dict[str, Any]:
        problems = []
        counts = {}
        with self._connect() as conn:
            for table in TABLES:
                prev, expected_seq, n = GENESIS, 1, 0
                for r in conn.execute(f"SELECT * FROM {table} ORDER BY seq"):
                    row = dict(r)
                    n += 1
                    ident = row.get("decision_id") if table == "decisions" else row.get("review_id")
                    if row["seq"] != expected_seq:
                        problems.append({"table": table, "seq": row["seq"], "id": ident,
                                         "problem": f"missing row(s) before seq {row['seq']}"})
                    if row["prev_hash"] != prev:
                        problems.append({"table": table, "seq": row["seq"], "id": ident,
                                         "problem": "prev_hash does not match the previous row"})
                    if row_hash(table, row) != row["row_hash"]:
                        problems.append({"table": table, "seq": row["seq"], "id": ident,
                                         "problem": "row contents changed after it was written"})
                    prev, expected_seq = row["row_hash"], row["seq"] + 1
                counts[table] = n
        return {"ok": not problems, "rows": counts, "problems": problems}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ClaimGuard audit log tools")
    ap.add_argument("command", choices=["verify", "head"])
    ap.add_argument("--db", default="data/claimguard.db")
    args = ap.parse_args(argv)
    if not Path(args.db).exists():
        print(f"no database at {args.db}", file=sys.stderr)
        return 2
    log = AuditLog(args.db)
    if args.command == "head":
        print(json.dumps(log.head(), indent=2))
        return 0
    report = log.verify()
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
