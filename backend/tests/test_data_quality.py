"""Tests for DataQualityService — ingestion-time payload validation."""

from datetime import UTC, datetime, timedelta

from app.services.data_quality_service import DataQualityService


def _make_payload(**kwargs):
    """Build a minimal valid WebhookPayload for testing."""
    from app.models.transaction import TransactionType
    from app.schemas.transaction import WebhookPayload

    defaults = dict(
        transaction_id="TX-TEST-001",
        account_id="ACC-001",
        client_id="CLI-001",
        transaction_type=TransactionType.DEPOSIT,
        amount=500.0,
        currency="XAF",
        transaction_date=datetime.now(UTC) - timedelta(minutes=5),
    )
    defaults.update(kwargs)
    return WebhookPayload(**defaults)


class TestDataQualityService:
    def setup_method(self):
        self.svc = DataQualityService()

    # ── Valid payload ──────────────────────────────────────────────────────────

    def test_valid_payload_passes(self):
        payload = _make_payload()
        result = self.svc.validate(payload)
        assert result.is_valid is True
        assert result.errors == []

    # ── Timestamp checks ───────────────────────────────────────────────────────

    def test_future_timestamp_is_error(self):
        payload = _make_payload(
            transaction_date=datetime.now(UTC) + timedelta(minutes=10)
        )
        result = self.svc.validate(payload)
        assert result.is_valid is False
        assert any("future" in e for e in result.errors)

    def test_future_within_clock_skew_is_ok(self):
        """Up to 5 minutes ahead is tolerated."""
        payload = _make_payload(
            transaction_date=datetime.now(UTC) + timedelta(seconds=60)
        )
        result = self.svc.validate(payload)
        assert result.is_valid is True

    def test_ancient_timestamp_is_error(self):
        payload = _make_payload(
            transaction_date=datetime.now(UTC) - timedelta(days=365 * 6)
        )
        result = self.svc.validate(payload)
        assert result.is_valid is False
        assert any("past" in e for e in result.errors)

    # ── Amount cap ─────────────────────────────────────────────────────────────

    def test_amount_exceeding_hard_cap_is_error(self):
        payload = _make_payload(amount=2_000_000_000.0)
        result = self.svc.validate(payload)
        assert result.is_valid is False
        assert any("cap" in e for e in result.errors)

    def test_amount_at_cap_boundary_passes(self):
        payload = _make_payload(amount=999_999_999.0)
        result = self.svc.validate(payload)
        # No amount error (other warnings may exist)
        assert not any("cap" in e for e in result.errors)

    # ── Currency code ──────────────────────────────────────────────────────────

    def test_invalid_currency_is_warning_not_error(self):
        payload = _make_payload(currency="ZZZ")
        result = self.svc.validate(payload)
        assert result.is_valid is True  # warning, not error
        assert any("ISO 4217" in w for w in result.warnings)

    def test_valid_xaf_currency_passes(self):
        payload = _make_payload(currency="XAF")
        result = self.svc.validate(payload)
        assert not any("ISO 4217" in w for w in result.warnings)

    # ── Country code ───────────────────────────────────────────────────────────

    def test_invalid_country_code_is_warning(self):
        payload = _make_payload(country_code="XX")
        result = self.svc.validate(payload)
        assert result.is_valid is True
        assert any("ISO 3166" in w for w in result.warnings)

    def test_valid_country_code_cm_passes(self):
        payload = _make_payload(country_code="CM")
        result = self.svc.validate(payload)
        assert not any("ISO 3166" in w for w in result.warnings)

    # ── IP address ─────────────────────────────────────────────────────────────

    def test_invalid_ip_address_is_warning(self):
        payload = _make_payload(ip_address="999.999.999.999")
        result = self.svc.validate(payload)
        assert result.is_valid is True
        assert any("IPv4" in w or "ip_address" in w for w in result.warnings)

    def test_valid_ipv4_passes(self):
        payload = _make_payload(ip_address="192.168.1.1")
        result = self.svc.validate(payload)
        assert not any("ip_address" in w for w in result.warnings)

    def test_valid_ipv6_passes(self):
        payload = _make_payload(ip_address="2001:db8::1")
        result = self.svc.validate(payload)
        assert not any("ip_address" in w for w in result.warnings)

    # ── Counterparty consistency ───────────────────────────────────────────────

    def test_counterparty_on_deposit_is_warning(self):
        payload = _make_payload(
            counterparty_account_id="ACC-OTHER",
        )
        result = self.svc.validate(payload)
        # DEPOSIT with counterparty_account_id should produce a warning
        assert any("counterparty" in w for w in result.warnings)

    def test_counterparty_on_transfer_is_ok(self):
        from app.models.transaction import TransactionType

        payload = _make_payload(
            transaction_type=TransactionType.TRANSFER,
            counterparty_account_id="ACC-OTHER",
        )
        result = self.svc.validate(payload)
        assert not any("counterparty" in w for w in result.warnings)

    # ── Actor type ─────────────────────────────────────────────────────────────

    def test_invalid_actor_type_is_warning(self):
        payload = _make_payload(actor_type="robot")
        result = self.svc.validate(payload)
        assert result.is_valid is True
        assert any("actor_type" in w for w in result.warnings)

    def test_valid_actor_types_pass(self):
        for actor_type in ("customer", "agent", "merchant"):
            payload = _make_payload(actor_type=actor_type)
            result = self.svc.validate(payload)
            assert not any("actor_type" in w for w in result.warnings), actor_type
