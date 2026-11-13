"""Encoding pipeline shared by every model."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .schema import NOMINAL, NUMERIC, ORDINAL


def build_preprocessor(columns: list[str]) -> ColumnTransformer:
    """Ordered ranges -> integers, other categories -> one-hot, numbers -> imputed.

    Scaling does not change tree models but keeps Logistic Regression well behaved,
    so every model shares the same pipeline.
    """
    ordinal = [c for c in ORDINAL if c in columns]
    nominal = [c for c in NOMINAL if c in columns]
    numeric = [c for c in NUMERIC if c in columns]

    return ColumnTransformer(
        [
            ("ord", Pipeline([
                ("encode", OrdinalEncoder(categories=[ORDINAL[c] for c in ordinal])),
                ("scale", StandardScaler()),
            ]), ordinal),
            ("cat", OneHotEncoder(categories=[NOMINAL[c] for c in nominal],
                                  handle_unknown="ignore", sparse_output=False), nominal),
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric),
        ],
        verbose_feature_names_out=False,
    )


def source_feature(encoded_name: str, columns: list[str]) -> str:
    """Map an encoded column (e.g. 'Fault_Policy Holder') back to its raw column."""
    if encoded_name in columns:
        return encoded_name
    for col in sorted(columns, key=len, reverse=True):
        if encoded_name.startswith(col + "_"):
            return col
    return encoded_name
