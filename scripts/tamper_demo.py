"""Demo: show that the audit log blocks edits and detects tampering.

    python scripts/tamper_demo.py [--db data/demo.db]

Works on a COPY of the database, so the real audit log is never touched.
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from claimguard.audit import AuditLog  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="data/demo.db")
    db = Path(ap.parse_args().db)
    if not db.exists():
        print(f"No audit database at {db}. Score a few claims first.")
        return 2

    copy = Path(tempfile.mkdtemp()) / "tamper_copy.db"
    shutil.copy(db, copy)
    conn = sqlite3.connect(copy)
    row = conn.execute("SELECT decision_id, band, score FROM decisions "
                       "ORDER BY score DESC LIMIT 1").fetchone()
    if row is None:
        print("The audit log has no decisions yet. Score a few claims first.")
        return 2
    did, band, score = row
    print(f"Working on a copy: {copy}")
    print(f"Target: {did}, band {band}, score {score}\n")

    print("1) Before: the chain verifies")
    print(f"   ok = {AuditLog(copy).verify()['ok']}\n")

    print(f"2) Try to quietly downgrade {did} to LOW with a normal UPDATE")
    try:
        conn.execute("UPDATE decisions SET band = 'LOW', score = 0.01 WHERE decision_id = ?", (did,))
        conn.commit()
        print("   the update went through (unexpected)")
    except sqlite3.DatabaseError as e:
        print(f"   BLOCKED by the database: {e}\n")

    print("3) A determined insider removes the protection and edits the row anyway")
    conn.execute("DROP TRIGGER decisions_no_update")
    conn.execute("UPDATE decisions SET band = 'LOW', score = 0.01 WHERE decision_id = ?", (did,))
    conn.commit()
    conn.close()
    print(f"   {did} now says LOW, score 0.01\n")

    print("4) Run verification again")
    report = AuditLog(copy).verify()
    print(f"   ok = {report['ok']}")
    for p in report["problems"]:
        print(f"   DETECTED: {p['id']} (seq {p['seq']}): {p['problem']}")
    print(f"\nThe real log at {db} was not modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
