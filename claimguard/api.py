"""ClaimGuard REST API.

    uvicorn claimguard.api:app --reload

Environment variables:
    CLAIMGUARD_DB     SQLite audit log path (default data/claimguard.db)
    CLAIMGUARD_MODEL  model bundle path (default models/claimguard_model.joblib)
"""
from __future__ import annotations

import csv
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import schema as S
from .audit import AuditLog
from .rules import config_hash
from .scoring import ClaimScorer

ROOT = Path(__file__).resolve().parent.parent
MAX_BATCH_ROWS = 5000

# Literal types built from the shared vocabulary: an unknown value is a 422, never a wrong score.
Month = Literal[*S.MONTHS]
Day = Literal[*S.DAYS]


def _lit(col: str):
    return Literal[*S.CATEGORIES[col]]


class Claim(BaseModel):
    """One claim, in the same shape as a row of fraud_oracle.csv."""
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "Month": "Jan", "WeekOfMonth": 1, "DayOfWeek": "Monday", "Make": "Honda", "AccidentArea": "Urban",
        "DayOfWeekClaimed": "Tuesday", "MonthClaimed": "Jan", "WeekOfMonthClaimed": 1, "Sex": "Male",
        "MaritalStatus": "Married", "Age": 30, "Fault": "Policy Holder", "PolicyType": "Sedan - All Perils",
        "VehicleCategory": "Sedan", "VehiclePrice": "less than 20000", "Deductible": 400, "DriverRating": 1,
        "Days_Policy_Accident": "1 to 7", "Days_Policy_Claim": "15 to 30", "PastNumberOfClaims": "none",
        "AgeOfVehicle": "7 years", "AgeOfPolicyHolder": "26 to 30", "PoliceReportFiled": "No",
        "WitnessPresent": "No", "AgentType": "External", "NumberOfSuppliments": "none",
        "AddressChange_Claim": "1 year", "NumberOfCars": "1 vehicle", "BasePolicy": "All Perils"}})

    Month: Month
    WeekOfMonth: int = Field(ge=1, le=5)
    DayOfWeek: Day
    Make: _lit("Make")
    AccidentArea: _lit("AccidentArea")
    DayOfWeekClaimed: Day
    MonthClaimed: Month
    WeekOfMonthClaimed: int = Field(ge=1, le=5)
    Sex: _lit("Sex")
    MaritalStatus: _lit("MaritalStatus")
    Age: int | None = Field(default=None, ge=0, le=120, description="0 or blank means not recorded")
    Fault: _lit("Fault")
    PolicyType: _lit("PolicyType")
    VehicleCategory: _lit("VehicleCategory")
    VehiclePrice: _lit("VehiclePrice")
    Deductible: int = Field(ge=0, le=10_000)
    DriverRating: int = Field(ge=1, le=4)
    Days_Policy_Accident: _lit("Days_Policy_Accident")
    Days_Policy_Claim: _lit("Days_Policy_Claim")
    PastNumberOfClaims: _lit("PastNumberOfClaims")
    AgeOfVehicle: _lit("AgeOfVehicle")
    AgeOfPolicyHolder: _lit("AgeOfPolicyHolder")
    PoliceReportFiled: _lit("PoliceReportFiled")
    WitnessPresent: _lit("WitnessPresent")
    AgentType: _lit("AgentType")
    NumberOfSuppliments: _lit("NumberOfSuppliments")
    AddressChange_Claim: _lit("AddressChange_Claim")
    NumberOfCars: _lit("NumberOfCars")
    BasePolicy: _lit("BasePolicy")
    claim_ref: str | None = Field(default=None, max_length=64, description="Caller's own claim number")


class RuleHit(BaseModel):
    id: str
    reason: str
    weight: float
    strong: bool


class ModelReason(BaseModel):
    feature: str
    value: str | float | int | None
    text: str
    impact: float


class Decision(BaseModel):
    decision_id: str
    claim_ref: str | None = None
    score: float
    band: Literal["LOW", "MEDIUM", "HIGH"]
    model_probability: float
    rule_score: float
    rule_hits: list[RuleHit]
    model_reasons: list[ModelReason]
    model_version: str


class ReviewIn(BaseModel):
    outcome: Literal["CONFIRMED_FRAUD", "CLEARED"]
    reviewer: str = Field(min_length=1, max_length=100)
    notes: str = Field(default="", max_length=2000)


class AppState:
    scorer: ClaimScorer
    audit: AuditLog
    config_hash: str


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.scorer = ClaimScorer.from_paths(os.environ.get("CLAIMGUARD_MODEL"))
    state.audit = AuditLog(os.environ.get("CLAIMGUARD_DB", str(ROOT / "data" / "claimguard.db")))
    state.config_hash = config_hash()
    yield


app = FastAPI(title="ClaimGuard", version="1.0",
              description="Insurance claims fraud triage: score, band, reasons and an audit trail.",
              lifespan=lifespan)


def _score_and_log(claim: Claim) -> Decision:
    payload = claim.model_dump()
    result = state.scorer.score(payload)
    row = state.audit.record_decision(payload, result["score"], result["band"], result["rule_hits"],
                                      result["model_reasons"], result["model_version"], state.config_hash)
    return Decision(decision_id=row["decision_id"], claim_ref=claim.claim_ref, **result)


@app.post("/score", response_model=Decision)
def score(claim: Claim) -> Decision:
    return _score_and_log(claim)


@app.post("/score/batch")
async def score_batch(file: UploadFile = File(...)) -> dict:
    """Score a CSV of claims (same columns as fraud_oracle.csv; extra columns are ignored)."""
    text = (await file.read()).decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise HTTPException(422, "empty CSV")
    missing = [c for c in S.FEATURES if c not in reader.fieldnames]
    if missing:
        raise HTTPException(422, f"CSV is missing columns: {missing}")
    wanted = set(S.FEATURES) | {"claim_ref"}
    results, errors = [], []
    for i, raw in enumerate(reader, start=1):
        if i > MAX_BATCH_ROWS:
            raise HTTPException(413, f"at most {MAX_BATCH_ROWS} rows per batch")
        try:
            claim = Claim(**{k: v for k, v in raw.items() if k in wanted and v not in (None, "")})
        except ValidationError as e:
            errors.append({"row": i, "errors": [{"field": ".".join(map(str, err["loc"])), "msg": err["msg"]}
                                                for err in e.errors()]})
            continue
        results.append({"row": i, **_score_and_log(claim).model_dump()})
    bands = {b: sum(r["band"] == b for r in results) for b in ("LOW", "MEDIUM", "HIGH")}
    return {"scored": len(results), "rejected": len(errors), "bands": bands,
            "results": results, "errors": errors}


@app.get("/decisions/{decision_id}")
def get_decision(decision_id: str) -> dict:
    found = state.audit.get_decision(decision_id)
    if found is None:
        raise HTTPException(404, "decision not found")
    return found


@app.post("/decisions/{decision_id}/review", status_code=201)
def review(decision_id: str, body: ReviewIn) -> dict:
    try:
        return state.audit.add_review(decision_id, body.outcome, body.reviewer, body.notes)
    except KeyError:
        raise HTTPException(404, "decision not found")


@app.get("/metrics")
def metrics() -> dict:
    return state.audit.metrics()


@app.get("/health")
def health() -> dict:
    b = state.scorer.bundle
    return {
        "status": "ok",
        "model_version": state.scorer.model_version,
        "trained_at": b.get("trained_at"),
        "thresholds": b["thresholds"],
        "combine_weight": b["combine_weight"],
        "rules_loaded": [r.id for r in state.scorer.rules],
        "config_hash": state.config_hash,
        "config_matches_training": state.config_hash == b.get("config_hash"),
    }
