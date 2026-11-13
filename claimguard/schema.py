"""Column vocabulary for fraud_oracle.csv.

One place lists every allowed category, so the encoder, the rules engine and the
API validation all agree on what a valid claim looks like. Ordered columns list
their categories from low to high.
"""

TARGET = "FraudFound_P"
TIME_COLUMN = "Year"

# ID-like columns: no real signal, and they let a model memorise rows.
DROP_COLUMNS = ["PolicyNumber", "RepNumber"]

# Calendar position of the accident and claim. Tested as a group in training: inside one
# year they look predictive, but they memorise that year's patterns and fail on the next.
CALENDAR_COLUMNS = ["Month", "DayOfWeek", "WeekOfMonth", "MonthClaimed", "DayOfWeekClaimed",
                    "WeekOfMonthClaimed"]

# Columns used only for the fairness experiment.
PROTECTED_COLUMNS = ["Sex", "Age", "AgeOfPolicyHolder"]

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Ranges where order carries meaning. Listed low -> high.
ORDINAL = {
    "VehiclePrice": [
        "less than 20000", "20000 to 29000", "30000 to 39000",
        "40000 to 59000", "60000 to 69000", "more than 69000",
    ],
    # "none" is read as 0 days (accident on the day the policy started).
    "Days_Policy_Accident": ["none", "1 to 7", "8 to 15", "15 to 30", "more than 30"],
    "Days_Policy_Claim": ["none", "8 to 15", "15 to 30", "more than 30"],
    "PastNumberOfClaims": ["none", "1", "2 to 4", "more than 4"],
    "AgeOfVehicle": [
        "new", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "more than 7",
    ],
    "AgeOfPolicyHolder": [
        "16 to 17", "18 to 20", "21 to 25", "26 to 30", "31 to 35",
        "36 to 40", "41 to 50", "51 to 65", "over 65",
    ],
    "NumberOfSuppliments": ["none", "1 to 2", "3 to 5", "more than 5"],
    "NumberOfCars": ["1 vehicle", "2 vehicles", "3 to 4", "5 to 8", "more than 8"],
}

# Categories with no natural order: one-hot encoded.
NOMINAL = {
    "Month": MONTHS,
    "DayOfWeek": DAYS,
    "Make": [
        "Accura", "BMW", "Chevrolet", "Dodge", "Ferrari", "Ford", "Honda", "Jaguar",
        "Lexus", "Mazda", "Mecedes", "Mercury", "Nisson", "Pontiac", "Porche",
        "Saab", "Saturn", "Toyota", "VW",
    ],
    "AccidentArea": ["Urban", "Rural"],
    "DayOfWeekClaimed": DAYS,
    "MonthClaimed": MONTHS,
    "Sex": ["Male", "Female"],
    "MaritalStatus": ["Single", "Married", "Divorced", "Widow"],
    "Fault": ["Policy Holder", "Third Party"],
    "PolicyType": [
        "Sedan - All Perils", "Sedan - Collision", "Sedan - Liability",
        "Sport - All Perils", "Sport - Collision", "Sport - Liability",
        "Utility - All Perils", "Utility - Collision", "Utility - Liability",
    ],
    "VehicleCategory": ["Sedan", "Sport", "Utility"],
    "PoliceReportFiled": ["Yes", "No"],
    "WitnessPresent": ["Yes", "No"],
    "AgentType": ["External", "Internal"],
    # A recency range, but "no change" does not sit on the same scale, so one-hot.
    "AddressChange_Claim": ["no change", "under 6 months", "1 year", "2 to 3 years", "4 to 8 years"],
    "BasePolicy": ["All Perils", "Collision", "Liability"],
}

# Plain numbers. Age 0 in the raw data means "unknown" and becomes NaN.
NUMERIC = {
    "WeekOfMonth": (1, 5),
    "WeekOfMonthClaimed": (1, 5),
    "Age": (0, 120),
    "Deductible": (0, 10_000),
    "DriverRating": (1, 4),
}

CATEGORIES = {**ORDINAL, **NOMINAL}
FEATURES = list(NOMINAL) + list(ORDINAL) + list(NUMERIC)


def feature_columns(include_protected: bool = True, include_calendar: bool = True) -> list[str]:
    drop = set()
    if not include_protected:
        drop |= set(PROTECTED_COLUMNS)
    if not include_calendar:
        drop |= set(CALENDAR_COLUMNS)
    return [c for c in FEATURES if c not in drop]
