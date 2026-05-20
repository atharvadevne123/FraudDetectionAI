"""Tests for utils.validation."""
from __future__ import annotations

import pytest
from utils.validation import validate_transaction_payload


VALID_PAYLOAD = {
    "user_id": 1,
    "amount": 150.0,
    "merchant_category": "retail",
    "payment_method": "credit",
    "account_age_days": 365,
    "credit_utilization": 0.4,
    "prior_fraud_count": 0,
}


class TestRequiredFields:
    def test_valid_payload_no_errors(self):
        assert validate_transaction_payload(VALID_PAYLOAD) == []

    def test_missing_amount(self):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "amount"}
        errors = validate_transaction_payload(payload)
        assert any("amount" in e for e in errors)

    def test_missing_user_id(self):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "user_id"}
        errors = validate_transaction_payload(payload)
        assert any("user_id" in e for e in errors)

    def test_missing_multiple_fields(self):
        errors = validate_transaction_payload({})
        assert len(errors) >= 1


class TestAmountValidation:
    def test_negative_amount(self):
        p = {**VALID_PAYLOAD, "amount": -1.0}
        assert validate_transaction_payload(p)

    def test_zero_amount_is_valid(self):
        p = {**VALID_PAYLOAD, "amount": 0.0}
        assert validate_transaction_payload(p) == []

    def test_max_amount_is_valid(self):
        p = {**VALID_PAYLOAD, "amount": 1_000_000.0}
        assert validate_transaction_payload(p) == []

    def test_over_max_amount(self):
        p = {**VALID_PAYLOAD, "amount": 1_000_001.0}
        assert validate_transaction_payload(p)

    def test_non_numeric_amount(self):
        p = {**VALID_PAYLOAD, "amount": "lots"}
        assert validate_transaction_payload(p)


class TestCreditUtilization:
    def test_utilization_above_one(self):
        p = {**VALID_PAYLOAD, "credit_utilization": 1.5}
        assert validate_transaction_payload(p)

    def test_utilization_below_zero(self):
        p = {**VALID_PAYLOAD, "credit_utilization": -0.1}
        assert validate_transaction_payload(p)

    def test_utilization_exactly_one(self):
        p = {**VALID_PAYLOAD, "credit_utilization": 1.0}
        assert validate_transaction_payload(p) == []


class TestOptionalFields:
    def test_negative_account_age(self):
        p = {**VALID_PAYLOAD, "account_age_days": -10}
        assert validate_transaction_payload(p)

    def test_negative_prior_fraud_count(self):
        p = {**VALID_PAYLOAD, "prior_fraud_count": -1}
        assert validate_transaction_payload(p)

    def test_non_integer_account_age(self):
        p = {**VALID_PAYLOAD, "account_age_days": "old"}
        assert validate_transaction_payload(p)
