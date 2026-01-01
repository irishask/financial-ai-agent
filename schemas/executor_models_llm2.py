"""
Pydantic models for LLM-2 (Executor) data access over transactions.csv.

These models are used to:
- Describe HOW LLM-2 wants to filter transactions (TransactionQuerySpec).
- Represent individual transactions (TransactionRecord).
- Return both raw transactions and basic aggregates (TransactionQueryResult).

All "intelligence" about WHICH filters to use is in LLM-2 prompts.
Python code will only apply these filters deterministically.
"""

from __future__ import annotations

from datetime import date
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


###########################################################################################
# TRANSACTION MODELS
###########################################################################################

class TransactionRecord(BaseModel):
    """
    One transaction row as seen by LLM-2.

    Mirrors the core columns from transactions.csv so the executor can:
    - list transactions (e.g. "show dining transactions"),
    - inspect dates, amounts, categories,
    - use them to build explanations and back-office logs.
    """

    transaction_id: str
    user_id: str
    account_id: str
    account_type: Optional[str] = None

    amount: float
    direction: Literal["D", "C"] = Field(
        description='"D" = debit (spending), "C" = credit (income)'
    )

    date: date
    month: Optional[int] = None
    year: Optional[int] = None
    dayOfWeek: Optional[str] = None

    categoryGroupId: Optional[str] = None
    categoryName: Optional[str] = None
    subCategoryId: Optional[str] = None
    subCategoryName: Optional[str] = None


class TransactionQuerySpec(BaseModel):
    """
    Generic, typed filter specification for querying transactions.

    LLM-2 decides which filters to set based on:
    - router_output (uc_type, subtype, complexity_axes),
    - conversation_summary (preferences),
    - query text and clarifications.

    Python code will:
    - load transactions.csv,
    - apply these filters deterministically (no extra "smart" logic),
    - return a TransactionQueryResult.
    """

    # Which end user we are querying transactions for (mandatory)
    user_id: str = Field(
        description="Logical end-user identifier (e.g. USER_001)."
    )

    # Account scope (optional)
    account_ids: Optional[List[str]] = Field(
        default=None,
        description=(
            "Restrict to specific accounts. If None, use all accounts for this user."
        ),
    )

    # Explicit date range (inclusive). If None, no bound on that side.
    start_date: Optional[date] = Field(
        default=None,
        description="Start date (inclusive) for the date filter.",
    )
    end_date: Optional[date] = Field(
        default=None,
        description="End date (inclusive) for the date filter.",
    )

    # Category filters (resolved via Category KB before calling this function).
    category_group_ids: Optional[List[str]] = Field(
        default=None,
        description="List of categoryGroupId values to include (e.g. ['CG10000', 'CG800']).",
    )
    sub_category_ids: Optional[List[str]] = Field(
        default=None,
        description="List of subCategoryId values to include (e.g. ['C801', 'C802']).",
    )

    # Amount filters (for things like 'large purchases').
    min_amount: Optional[float] = Field(
        default=None,
        description="Minimum absolute transaction amount (inclusive).",
    )
    max_amount: Optional[float] = Field(
        default=None,
        description="Maximum absolute transaction amount (inclusive).",
    )

    # Direction: spending vs income vs both.
    direction: Optional[Literal["D", "C", "BOTH"]] = Field(
        default=None,
        description='"D" = debit (spending), "C" = credit (income), "BOTH" = no direction filter.',
    )

    # Optional result shaping: for last transaction, top N, etc.
    limit: Optional[int] = Field(
        default=None,
        description="If set, limit the number of returned rows (after sorting, if sort_by is set).",
    )
    sort_by: Optional[Literal["date_asc", "date_desc"]] = Field(
        default=None,
        description='Optional sort order for date (e.g. "date_desc" for last transaction).',
    )


class TransactionQueryResult(BaseModel):
    """
    Result of applying a TransactionQuerySpec to transactions.csv.

    Contains:
    - The matching transactions (as records) for detailed reasoning and logging.
    - Basic aggregates that are cheap to compute in Python and common across many UCs.
    """

    # Raw rows after filtering
    transactions: List[TransactionRecord] = Field(
        default_factory=list,
        description="List of transactions matching the filter spec.",
    )

    # Basic aggregates over the matched transactions
    total_count: int = Field(
        description="Number of matched transactions."
    )
    total_debit_amount: float = Field(
        description="Sum of amounts for debit (direction='D') transactions."
    )
    total_credit_amount: float = Field(
        description="Sum of amounts for credit (direction='C') transactions."
    )
    net_amount: float = Field(
        description="total_credit_amount - total_debit_amount (useful for net balance-like reasoning)."
    )

    avg_amount: Optional[float] = Field(
        default=None,
        description="Average absolute transaction amount over all matched transactions (optional).",
    )
    max_amount: Optional[float] = Field(
        default=None,
        description="Maximum absolute transaction amount over all matched transactions (optional).",
    )
    min_amount: Optional[float] = Field(
        default=None,
        description="Minimum absolute transaction amount over all matched transactions (optional).",
    )


###########################################################################################
# DATE RANGE MODELS
###########################################################################################

# class DateRangeRequest(BaseModel):
#     """
#     Input schema for the get_date_range tool.

#     LLM-2 chooses a normalized period_type based on the user query,
#     router_output, and conversation_summary (preferences). This model
#     does NOT interpret vague phrases itself – it only carries the
#     decision that LLM-2 already made.

#     The tool implementation will:
#     - Use period_type (+ optional month/year) together with an anchor_date
#       (or today) to compute start_date and end_date.
#     """

#     period_type: Literal[
#         "last_month",
#         "this_week",
#         "last_30_days",
#         "last_7_days",    
#         "this_year",
#         "named_month",
#     ] = Field(
#         description=(
#             "Normalized period type chosen by LLM-2. "
#             "Examples: 'last_month', 'this_week', 'last_30_days', "
#             "'this_year', 'named_month'."
#         )
#     )

#     month: Optional[int] = Field(
#         default=None,
#         description=(
#             "Month number (1-12) used only when period_type == 'named_month'. "
#             "For example, 3 for March."
#         ),
#     )

#     year: Optional[int] = Field(
#         default=None,
#         description=(
#             "Year used for 'named_month' (e.g. 2024). "
#             "If None, the tool may default to anchor_date.year or the current year."
#         ),
#     )

#     anchor_date: Optional[date] = Field(
#         default=None,
#         description=(
#             "Reference date for relative periods such as 'last_month' or "
#             "'this_week'. If None, the tool will use today's date."
#         ),
#     )


# class DateRangeResult(BaseModel):
#     """
#     Output schema for the get_date_range tool.

#     Represents a concrete, inclusive date range that LLM-2 can plug into
#     TransactionQuerySpec.start_date / end_date, plus a small label and
#     optional notes for back-office logging.
#     """

#     start_date: date = Field(
#         description="Start date (inclusive) of the resolved period."
#     )
#     end_date: date = Field(
#         description="End date (inclusive) of the resolved period."
#     )

#     label: str = Field(
#         description=(
#             "Short identifier for the resolved period, e.g. "
#             "'last_month_from_2025-11-19' or 'march_2025'."
#         )
#     )

#     notes: Optional[str] = Field(
#         default=None,
#         description=(
#             "Optional human-readable explanation of how the range was computed, "
#             "useful for back-office logs (e.g. 'Calendar month before anchor_date')."
#         ),
#     )