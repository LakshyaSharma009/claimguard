import sqlite3

import pytest

from claimguard.audit import AuditLog, main


@pytest.fixture
def log(tmp_path):
    log = AuditLog(tmp_path / "audit.db")
    for i in range(5):
        log.record_decision({"claim": i}, 0.1 * i, "LOW", [], [], "test-v1", "cfg")
    return log


def _tamper(path, sql):
    """Someone with file access drops the triggers and edits the table directly."""
    conn = sqlite3.connect(path)
    conn.execute("DROP TRIGGER decisions_no_update")
    conn.execute("DROP TRIGGER decisions_no_delete")
    conn.execute(sql)
    conn.commit()
    conn.close()


def test_chain_verifies(log):
    report = log.verify()
    assert report["ok"], report
    assert report["rows"]["decisions"] == 5


def test_decision_ids_are_sequential(log):
    d = log.record_decision({"claim": 99}, 0.5, "MEDIUM", [], [], "test-v1", "cfg")
    assert d["decision_id"].endswith("-000006")
    assert log.get_decision(d["decision_id"])["claim"] == {"claim": 99}


def test_updates_and_deletes_are_blocked(log):
    conn = sqlite3.connect(log.path)
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("UPDATE decisions SET score = 0.99 WHERE seq = 2")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("DELETE FROM decisions WHERE seq = 2")
    conn.close()


def test_changing_a_stored_row_breaks_verify(log):
    _tamper(log.path, "UPDATE decisions SET band = 'HIGH' WHERE seq = 3")
    report = AuditLog(log.path).verify()
    assert not report["ok"]
    assert any(p["seq"] == 3 and "changed" in p["problem"] for p in report["problems"])


def test_deleting_a_row_breaks_verify(log):
    _tamper(log.path, "DELETE FROM decisions WHERE seq = 2")
    report = AuditLog(log.path).verify()
    assert not report["ok"]
    assert any(p["seq"] == 3 for p in report["problems"])


def test_reviews_append_and_latest_wins(log):
    did = log.record_decision({"x": 1}, 0.9, "HIGH", [], [], "test-v1", "cfg")["decision_id"]
    log.add_review(did, "CLEARED", "investigator-1")
    log.add_review(did, "CONFIRMED_FRAUD", "investigator-2", "second look")
    high = log.metrics()["bands"]["HIGH"]
    assert high == {"decisions": 1, "reviewed": 1, "confirmed_fraud": 1, "confirmed_fraud_rate": 1.0}
    assert len(log.get_decision(did)["reviews"]) == 2
    assert log.verify()["ok"]


def test_review_of_unknown_decision(log):
    with pytest.raises(KeyError):
        log.add_review("d-0000-999999", "CLEARED", "investigator-1")


def test_cli_exit_codes(log):
    assert main(["verify", "--db", log.path]) == 0
    _tamper(log.path, "UPDATE decisions SET score = 0.99 WHERE seq = 2")
    assert main(["verify", "--db", log.path]) == 1
