import io

import pandas as pd


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_version"]
    assert body["config_matches_training"] is True


def test_score_returns_full_decision(client, quiet_claim):
    r = client.post("/score", json=quiet_claim)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["decision_id"].startswith("d-")
    assert 0 <= body["score"] <= 1
    assert body["band"] in {"LOW", "MEDIUM", "HIGH"}
    assert body["rule_hits"] == []


def test_same_claim_same_score(client, quiet_claim):
    a = client.post("/score", json=quiet_claim).json()
    b = client.post("/score", json=quiet_claim).json()
    assert a["score"] == b["score"] and a["band"] == b["band"]
    assert a["decision_id"] != b["decision_id"]  # each call is its own logged decision


def test_bad_category_is_422(client, quiet_claim):
    r = client.post("/score", json={**quiet_claim, "VehiclePrice": "cheap"})
    assert r.status_code == 422
    assert "VehiclePrice" in r.text


def test_missing_field_out_of_range_and_extra_field_are_422(client, quiet_claim):
    missing = {k: v for k, v in quiet_claim.items() if k != "Fault"}
    assert client.post("/score", json=missing).status_code == 422
    assert client.post("/score", json={**quiet_claim, "DriverRating": 9}).status_code == 422
    assert client.post("/score", json={**quiet_claim, "Surprise": 1}).status_code == 422


def test_decision_lookup_and_review(client, quiet_claim):
    flagged = {**quiet_claim, "Days_Policy_Claim": "8 to 15"}
    did = client.post("/score", json=flagged).json()["decision_id"]

    stored = client.get(f"/decisions/{did}").json()
    assert stored["claim"]["Days_Policy_Claim"] == "8 to 15"
    assert stored["rule_hits"][0]["id"] == "R07_EARLY_CLAIM"

    r = client.post(f"/decisions/{did}/review", json={"outcome": "CONFIRMED_FRAUD", "reviewer": "inv-7"})
    assert r.status_code == 201
    assert client.get(f"/decisions/{did}").json()["reviews"][0]["outcome"] == "CONFIRMED_FRAUD"

    metrics = client.get("/metrics").json()
    band = stored["band"]
    assert metrics["bands"][band]["confirmed_fraud"] == 1


def test_unknown_decision_is_404(client):
    assert client.get("/decisions/d-1999-000001").status_code == 404
    r = client.post("/decisions/d-1999-000001/review", json={"outcome": "CLEARED", "reviewer": "x"})
    assert r.status_code == 404


def test_bad_review_outcome_is_422(client, quiet_claim):
    did = client.post("/score", json=quiet_claim).json()["decision_id"]
    r = client.post(f"/decisions/{did}/review", json={"outcome": "MAYBE", "reviewer": "x"})
    assert r.status_code == 422


def test_batch_scores_good_rows_and_reports_bad_ones(client, quiet_claim):
    rows = [quiet_claim, {**quiet_claim, "Fault": "Nobody"}, {**quiet_claim, "Days_Policy_Accident": "1 to 7"}]
    buf = io.StringIO()
    pd.DataFrame(rows).assign(FraudFound_P=0).to_csv(buf, index=False)  # extra columns are ignored
    r = client.post("/score/batch", files={"file": ("claims.csv", buf.getvalue(), "text/csv")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["scored"] == 2 and body["rejected"] == 1
    assert body["errors"][0]["row"] == 2
    assert body["results"][1]["band"] in {"MEDIUM", "HIGH"}  # strong rule R01


def test_batch_missing_columns_is_422(client):
    r = client.post("/score/batch", files={"file": ("c.csv", "Fault\nThird Party\n", "text/csv")})
    assert r.status_code == 422


def test_audit_chain_is_valid_after_api_use(client, quiet_claim):
    from claimguard.api import state
    for _ in range(3):
        client.post("/score", json=quiet_claim)
    assert state.audit.verify()["ok"]


def test_age_blank_or_zero_means_not_recorded(client, quiet_claim):
    no_age = {k: v for k, v in quiet_claim.items() if k != "Age"}
    a = client.post("/score", json=no_age)
    b = client.post("/score", json={**quiet_claim, "Age": 0})
    assert a.status_code == 200 and b.status_code == 200, (a.text, b.text)
    assert a.json()["score"] == b.json()["score"]


def test_holdout_fixture_scores_through_batch(client):
    from pathlib import Path
    fixture = Path(__file__).parent / "fixtures" / "holdout_sample.csv"
    r = client.post("/score/batch", files={"file": ("h.csv", fixture.read_text(), "text/csv")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rejected"] == 0 and body["scored"] == 1074


def test_claim_ref_is_echoed(client, quiet_claim):
    body = client.post("/score", json={**quiet_claim, "claim_ref": "CLM-42"}).json()
    assert body["claim_ref"] == "CLM-42"
    assert client.get(f"/decisions/{body['decision_id']}").json()["claim"]["claim_ref"] == "CLM-42"


def test_stored_reasons_match_the_response(client, quiet_claim):
    # DriverRating is numeric; its reason value must be stored as a number, not text.
    for rating in (1, 2, 3, 4):
        body = client.post("/score", json={**quiet_claim, "DriverRating": rating, "Deductible": 500}).json()
        stored = client.get(f"/decisions/{body['decision_id']}").json()
        assert stored["model_reasons"] == body["model_reasons"]


def test_schema_lists_every_feature_with_valid_options(client):
    from claimguard import schema as S
    body = client.get("/schema").json()
    fields = {f["name"]: f for g in body["groups"] for f in g["fields"]}
    assert sorted(fields) == sorted(S.FEATURES)
    assert fields["Fault"]["options"] == S.CATEGORIES["Fault"]
    assert fields["DriverRating"]["min"] == 1 and fields["DriverRating"]["max"] == 4
    assert fields["Age"]["optional"] is True and fields["Age"]["used_by_model"] is False
    assert fields["Fault"]["used_by_model"] is True
    assert client.post("/score", json=body["example"]).status_code == 200


def test_review_queue_filters_and_sorts(client, quiet_claim):
    flagged = client.post("/score", json={**quiet_claim, "Days_Policy_Accident": "1 to 7",
                                          "claim_ref": "Q-1"}).json()
    assert flagged["band"] in {"MEDIUM", "HIGH"}
    queue = client.get("/decisions", params={"band": ["MEDIUM", "HIGH"], "reviewed": "false"}).json()
    ids = [d["decision_id"] for d in queue]
    assert flagged["decision_id"] in ids
    assert all(d["band"] in {"MEDIUM", "HIGH"} and d["latest_outcome"] is None for d in queue)
    assert [d["score"] for d in queue] == sorted((d["score"] for d in queue), reverse=True)
    assert next(d for d in queue if d["decision_id"] == flagged["decision_id"])["claim_ref"] == "Q-1"

    client.post(f"/decisions/{flagged['decision_id']}/review", json={"outcome": "CLEARED", "reviewer": "x"})
    pending = client.get("/decisions", params={"reviewed": "false", "limit": 500}).json()
    assert flagged["decision_id"] not in [d["decision_id"] for d in pending]
    done = client.get("/decisions", params={"reviewed": "true", "sort": "recent"}).json()
    assert done[0]["decision_id"] == flagged["decision_id"] and done[0]["latest_outcome"] == "CLEARED"
    assert client.get("/decisions", params={"band": "NOPE"}).status_code == 422
    assert client.get("/decisions", params={"limit": 0}).status_code == 422


def test_web_ui_is_served(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/ui/"
    page = client.get("/ui/")
    assert page.status_code == 200 and "ClaimGuard" in page.text
    assert client.get("/ui/app.js").status_code == 200
    assert client.get("/ui/styles.css").status_code == 200
