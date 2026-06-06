from __future__ import annotations

import pandas as pd


def to_pandas(df) -> pd.DataFrame:
    """Convert a Polars DataFrame (awpy 2.x output) to Pandas.

    Also accepts existing Pandas DataFrames or dict/list fallbacks.
    """
    if df is None:
        return pd.DataFrame()
    if hasattr(df, "to_pandas"):
        return df.to_pandas()
    if isinstance(df, pd.DataFrame):
        return df
    return pd.DataFrame(df)


def coerce_schema(df: pd.DataFrame, schema: dict[str, str]) -> pd.DataFrame:
    """Retain only schema columns present in the DataFrame and apply types.

    Missing columns are ignored. Conversion failures are silently skipped
    to prevent pipeline crashes.
    """
    present = [c for c in schema if c in df.columns]
    out = df[present].copy()
    for col in present:
        try:
            out[col] = out[col].astype(schema[col])
        except (ValueError, TypeError):
            pass
    return out


def missing_columns(df: pd.DataFrame, schema: dict[str, str]) -> list[str]:
    """Return schema columns missing from the DataFrame.

    Useful for logging warnings when expected columns are absent.
    """
    return [c for c in schema if c not in df.columns]