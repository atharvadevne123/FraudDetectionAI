"""Input validation helpers for transaction payloads."""

from __future__ import annotations

from typing import Any

REQUIRED_FIELDS = {"user_id", "amount", "merchant_category", "payment_method"}
AMOUNT_MIN = 0.0
AMOUNT_MAX = 1_000_000.0
CREDIT_UTILIZATION_RANGE = (0.0, 1.0)


def validate_transaction_payload(payload: dict[str, Any]) -> list[str]:
    """Validate a raw transaction payload dict against business rules.

    Checks for required fields, numeric type coercibility, and value range
    constraints for amount, credit_utilization, account_age_days, and
    prior_fraud_count.

    Args:
        payload: Raw transaction dict from API request or batch file.

    Returns:
        List of human-readable error strings; an empty list means the payload is valid.
    """
    errors: list[str] = []

    missing = REQUIRED_FIELDS - set(payload.keys())
    if missing:
        errors.append(f"Missing required fields: {sorted(missing)}")

    amount = payload.get("amount")
    if amount is not None:
        try:
            amount = float(amount)
        except (TypeError, ValueError):
            errors.append("'amount' must be a number")
            amount = None

    if amount is not None and not (AMOUNT_MIN <= amount <= AMOUNT_MAX):
        errors.append(
            f"'amount' must be between {AMOUNT_MIN} and {AMOUNT_MAX}, got {amount}"
        )

    util = payload.get("credit_utilization")
    if util is not None:
        try:
            util = float(util)
        except (TypeError, ValueError):
            errors.append("'credit_utilization' must be a number")
            util = None

    if util is not None and not (
        CREDIT_UTILIZATION_RANGE[0] <= util <= CREDIT_UTILIZATION_RANGE[1]
    ):
        errors.append(
            f"'credit_utilization' must be between 0 and 1, got {util}"
        )

    age = payload.get("account_age_days")
    if age is not None:
        try:
            age = int(age)
        except (TypeError, ValueError):
            errors.append("'account_age_days' must be an integer")
            age = None

    if age is not None and age < 0:
        errors.append(f"'account_age_days' must be non-negative, got {age}")

    prior = payload.get("prior_fraud_count")
    if prior is not None:
        try:
            prior = int(prior)
        except (TypeError, ValueError):
            errors.append("'prior_fraud_count' must be an integer")
            prior = None

    if prior is not None and prior < 0:
        errors.append(f"'prior_fraud_count' must be non-negative, got {prior}")

    return errors
