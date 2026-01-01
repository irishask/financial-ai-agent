"""
LangChain StructuredTool for querying transactions.csv for LLM-2 (Executor).

This module exposes:
- query_transactions_tool(spec: TransactionQuerySpec) -> TransactionQueryResult
- query_transactions_lc_tool  -> LangChain StructuredTool bound to that function

All reasoning about WHICH filters to use is done by LLM-2.
This module only:
- applies deterministic filters (user, account, date, categories, amount, direction),
- computes basic aggregates,
- returns TransactionQueryResult.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

import pandas as pd

from langchain_core.tools import StructuredTool
from schemas.executor_models_llm2 import (
    TransactionQuerySpec,
    TransactionQueryResult,
    TransactionRecord,
)


# --------------------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_transactions(csv_path: str = "data/transactions.csv") -> pd.DataFrame:
    
    """
    Load transactions.csv once and cache it.

    - Parses 'date' column as datetime.
    - Leaves other columns as-is.
    """
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', dayfirst=True)
    return df


def _nan_to_none(value):
    """Convert pandas NaN to Python None for optional fields."""
    if pd.isna(value):
        return None
    return value


# --------------------------------------------------------------------------------------
# Core tool function
# --------------------------------------------------------------------------------------


def query_transactions_tool(spec: TransactionQuerySpec) -> TransactionQueryResult:
    """
    Apply a TransactionQuerySpec to transactions.csv and return TransactionQueryResult.

    This function is intentionally dumb:
    - No understanding of "recent", "large", "coffee", etc.
    - It only respects the explicit filters in `spec`.
    All semantics (how to fill spec) come from LLM-2.
    """
    df = _load_transactions()

    # Filter by user_id (mandatory)
    df_f = df[df["user_id"] == spec.user_id].copy()

    # Optional account_ids filter
    if spec.account_ids:
        df_f = df_f[df_f["account_id"].isin(spec.account_ids)]

    # Date range filters (inclusive)
    if spec.start_date is not None:
        df_f = df_f[df_f["date"].dt.date >= spec.start_date]
    if spec.end_date is not None:
        df_f = df_f[df_f["date"].dt.date <= spec.end_date]

    # Category group / subcategory filters
    if spec.category_group_ids:
        df_f = df_f[df_f["categoryGroupId"].isin(spec.category_group_ids)]
    if spec.sub_category_ids:
        df_f = df_f[df_f["subCategoryId"].isin(spec.sub_category_ids)]

    # Amount filters (absolute value)
    if spec.min_amount is not None:
        df_f = df_f[df_f["amount"].abs() >= spec.min_amount]
    if spec.max_amount is not None:
        df_f = df_f[df_f["amount"].abs() <= spec.max_amount]

    # Direction filter
    if spec.direction == "D":
        df_f = df_f[df_f["direction"] == "D"]
    elif spec.direction == "C":
        df_f = df_f[df_f["direction"] == "C"]
    # "BOTH" or None -> no direction filter

    # Sorting
    if spec.sort_by == "date_asc":
        df_f = df_f.sort_values("date", ascending=True)
    elif spec.sort_by == "date_desc":
        df_f = df_f.sort_values("date", ascending=False)

    # Limit
    if spec.limit is not None and spec.limit > 0:
        df_f = df_f.head(spec.limit)

    # Compute aggregates
    total_count = int(len(df_f))

    if total_count > 0:
        # Debit and credit sums
        debits = df_f[df_f["direction"] == "D"]["amount"].sum()
        credits = df_f[df_f["direction"] == "C"]["amount"].sum()

        total_debit_amount = float(debits)
        total_credit_amount = float(credits)
        net_amount = float(total_credit_amount - total_debit_amount)

        abs_amounts = df_f["amount"].abs()
        avg_amount = float(abs_amounts.mean())
        max_amount = float(abs_amounts.max())
        min_amount = float(abs_amounts.min())
    else:
        total_debit_amount = 0.0
        total_credit_amount = 0.0
        net_amount = 0.0
        avg_amount = None
        max_amount = None
        min_amount = None

    # Build TransactionRecord list
    records: List[TransactionRecord] = []
    for row in df_f.itertuples(index=False):
        record = TransactionRecord(
            transaction_id=row.transaction_id,
            user_id=row.user_id,
            account_id=row.account_id,
            account_type=_nan_to_none(row.account_type),
            amount=float(row.amount),            
            direction=row.direction,
            date=row.date.date(),  # pandas.Timestamp -> date
            month=_nan_to_none(row.month),        
            year=_nan_to_none(row.year),
            dayOfWeek=_nan_to_none(row.dayOfWeek),
            categoryGroupId=_nan_to_none(row.categoryGroupId),
            categoryName=_nan_to_none(row.categoryName),
            subCategoryId=_nan_to_none(row.subCategoryId),
            subCategoryName=_nan_to_none(row.subCategoryName),
        )
        records.append(record)

    # Wrap in TransactionQueryResult
    result = TransactionQueryResult(
        transactions=records,
        total_count=total_count,
        total_debit_amount=total_debit_amount,
        total_credit_amount=total_credit_amount,
        net_amount=net_amount,
        avg_amount=avg_amount,
        max_amount=max_amount,
        min_amount=min_amount,
    )

    return result


# --------------------------------------------------------------------------------------
# LangChain StructuredTool wrapper
# --------------------------------------------------------------------------------------


query_transactions_lc_tool = StructuredTool.from_function(
    name="query_transactions",
    func=query_transactions_tool,
    description=(
        "Filter financial transactions for a single user by date range, "
        "categories, direction (spend vs income), accounts, and amount thresholds. "
        "Returns both matching transactions and basic aggregates."
    ),
)
