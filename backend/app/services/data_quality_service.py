"""Data quality validation for incoming webhook payloads."""

import ipaddress
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from app.schemas.transaction import WebhookPayload

logger = logging.getLogger(__name__)

# ISO 4217 currency codes (common subset; extend as needed)
_VALID_CURRENCIES = {
    "XAF", "XOF", "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD",
    "CNY", "HKD", "SGD", "NOK", "SEK", "DKK", "PLN", "CZK", "HUF", "RON",
    "BGN", "HRK", "RUB", "TRY", "BRL", "MXN", "ZAR", "INR", "IDR", "MYR",
    "PHP", "THB", "VND", "KRW", "AED", "SAR", "QAR", "KWD", "BHD", "OMR",
    "EGP", "NGN", "KES", "GHS", "TZS", "UGX", "ETB", "MAD", "DZD", "TND",
    "XCD", "XPF", "CFA", "GNF", "BIF", "RWF", "MZN", "ZMW", "BWP", "MWK",
}

# ISO 3166-1 alpha-2 country codes (common subset)
_VALID_COUNTRY_CODES = {
    "AF", "AX", "AL", "DZ", "AS", "AD", "AO", "AI", "AQ", "AG", "AR", "AM",
    "AW", "AU", "AT", "AZ", "BS", "BH", "BD", "BB", "BY", "BE", "BZ", "BJ",
    "BM", "BT", "BO", "BQ", "BA", "BW", "BV", "BR", "IO", "BN", "BG", "BF",
    "BI", "CV", "KH", "CM", "CA", "KY", "CF", "TD", "CL", "CN", "CX", "CC",
    "CO", "KM", "CG", "CD", "CK", "CR", "CI", "HR", "CU", "CW", "CY", "CZ",
    "DK", "DJ", "DM", "DO", "EC", "EG", "SV", "GQ", "ER", "EE", "SZ", "ET",
    "FK", "FO", "FJ", "FI", "FR", "GF", "PF", "TF", "GA", "GM", "GE", "DE",
    "GH", "GI", "GR", "GL", "GD", "GP", "GU", "GT", "GG", "GN", "GW", "GY",
    "HT", "HM", "VA", "HN", "HK", "HU", "IS", "IN", "ID", "IR", "IQ", "IE",
    "IM", "IL", "IT", "JM", "JP", "JE", "JO", "KZ", "KE", "KI", "KP", "KR",
    "KW", "KG", "LA", "LV", "LB", "LS", "LR", "LY", "LI", "LT", "LU", "MO",
    "MG", "MW", "MY", "MV", "ML", "MT", "MH", "MQ", "MR", "MU", "YT", "MX",
    "FM", "MD", "MC", "MN", "ME", "MS", "MA", "MZ", "MM", "NA", "NR", "NP",
    "NL", "NC", "NZ", "NI", "NE", "NG", "NU", "NF", "MK", "MP", "NO", "OM",
    "PK", "PW", "PS", "PA", "PG", "PY", "PE", "PH", "PN", "PL", "PT", "PR",
    "QA", "RE", "RO", "RU", "RW", "BL", "SH", "KN", "LC", "MF", "PM", "VC",
    "WS", "SM", "ST", "SA", "SN", "RS", "SC", "SL", "SG", "SX", "SK", "SI",
    "SB", "SO", "ZA", "GS", "SS", "ES", "LK", "SD", "SR", "SJ", "SE", "CH",
    "SY", "TW", "TJ", "TZ", "TH", "TL", "TG", "TK", "TO", "TT", "TN", "TR",
    "TM", "TC", "TV", "UG", "UA", "AE", "GB", "UM", "US", "UY", "UZ", "VU",
    "VE", "VN", "VG", "VI", "WF", "EH", "YE", "ZM", "ZW",
}

# Max amount hard cap (separate from AML threshold — catches data corruption)
_AMOUNT_HARD_CAP = 1_000_000_000.0  # 1 billion XAF

# Timestamp tolerance
_MAX_FUTURE_SECONDS = 300       # 5 minutes ahead allowed (clock skew)
_MAX_PAST_YEARS = 5             # Transactions older than 5 years are suspicious data


@dataclass
class DataQualityResult:
    is_valid: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class DataQualityService:
    """Validates incoming webhook payloads before they are persisted."""

    def validate(self, payload: WebhookPayload) -> DataQualityResult:
        errors: list[str] = []
        warnings: list[str] = []

        now = datetime.now(UTC)
        tx_date = payload.transaction_date
        if tx_date.tzinfo is None:
            tx_date = tx_date.replace(tzinfo=UTC)

        # 1. Timestamp sanity
        if tx_date > now + timedelta(seconds=_MAX_FUTURE_SECONDS):
            errors.append(
                f"transaction_date is {(tx_date - now).seconds}s in the future"
            )
        elif tx_date < now - timedelta(days=_MAX_PAST_YEARS * 365):
            errors.append(
                f"transaction_date is more than {_MAX_PAST_YEARS} years in the past"
            )

        # 2. Amount hard cap (catches data corruption, not AML threshold)
        if payload.amount > _AMOUNT_HARD_CAP:
            errors.append(
                f"amount {payload.amount} exceeds hard cap of {_AMOUNT_HARD_CAP}"
            )

        # 3. Currency code validity
        if payload.currency and payload.currency.upper() not in _VALID_CURRENCIES:
            warnings.append(
                f"currency '{payload.currency}' is not a recognised ISO 4217 code"
            )

        # 4. Country code validity
        if payload.country_code and payload.country_code.upper() not in _VALID_COUNTRY_CODES:
            warnings.append(
                f"country_code '{payload.country_code}' is not a valid ISO 3166-1 alpha-2 code"
            )

        # 5. IP address format
        if payload.ip_address:
            try:
                ipaddress.ip_address(payload.ip_address)
            except ValueError:
                warnings.append(
                    f"ip_address '{payload.ip_address}' is not a valid IPv4/IPv6 address"
                )

        # 6. Counterparty consistency
        if payload.counterparty_account_id and payload.transaction_type.value not in (
            "transfer", "loan_disbursement", "loan_repayment"
        ):
            warnings.append(
                f"counterparty_account_id present on non-transfer transaction "
                f"type '{payload.transaction_type.value}'"
            )

        # 7. KYC level range (belt-and-suspenders; Pydantic already validates ge=1 le=4)
        if payload.kyc_level is not None and payload.kyc_level not in (1, 2, 3, 4):
            warnings.append(f"kyc_level {payload.kyc_level} is outside expected range 1–4")

        # 8. Actor type validity
        if payload.actor_type and payload.actor_type not in ("customer", "agent", "merchant"):
            warnings.append(
                f"actor_type '{payload.actor_type}' is not one of: customer, agent, merchant"
            )

        return DataQualityResult(
            is_valid=len(errors) == 0,
            warnings=warnings,
            errors=errors,
        )
