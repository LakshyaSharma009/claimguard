from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "claimguard_model.joblib"

# A claim built to trip none of the rules.
QUIET_CLAIM = {
    "Month": "Mar", "WeekOfMonth": 2, "DayOfWeek": "Tuesday", "Make": "Toyota", "AccidentArea": "Urban",
    "DayOfWeekClaimed": "Wednesday", "MonthClaimed": "Mar", "WeekOfMonthClaimed": 2, "Sex": "Male",
    "MaritalStatus": "Married", "Age": 45, "Fault": "Third Party", "PolicyType": "Sedan - Liability",
    "VehicleCategory": "Sedan", "VehiclePrice": "20000 to 29000", "Deductible": 400, "DriverRating": 2,
    "Days_Policy_Accident": "more than 30", "Days_Policy_Claim": "more than 30",
    "PastNumberOfClaims": "1", "AgeOfVehicle": "5 years", "AgeOfPolicyHolder": "41 to 50",
    "PoliceReportFiled": "Yes", "WitnessPresent": "Yes", "AgentType": "External",
    "NumberOfSuppliments": "none", "AddressChange_Claim": "no change", "NumberOfCars": "1 vehicle",
    "BasePolicy": "Liability",
}


@pytest.fixture
def quiet_claim():
    return dict(QUIET_CLAIM)


@pytest.fixture(scope="session")
def scorer():
    if not MODEL.exists():
        pytest.skip("no trained model; run python -m claimguard.train")
    from claimguard.scoring import ClaimScorer
    return ClaimScorer.from_paths(MODEL)


@pytest.fixture
def client(tmp_path, monkeypatch):
    if not MODEL.exists():
        pytest.skip("no trained model; run python -m claimguard.train")
    monkeypatch.setenv("CLAIMGUARD_DB", str(tmp_path / "audit.db"))
    from fastapi.testclient import TestClient
    from claimguard.api import app
    with TestClient(app) as c:
        yield c
