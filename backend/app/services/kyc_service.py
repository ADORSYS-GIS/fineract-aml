"""KYC/KYB service — syncs customer data from Fineract and manages due diligence.

Pulls client details from Fineract's REST API, caches them locally,
and determines Enhanced Due Diligence (EDD) requirements based on
risk factors (PEP status, high-risk country, sanctions matches).
"""

import logging
from datetime import UTC, datetime

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.customer import Customer, CustomerRiskLevel, CustomerType

logger = logging.getLogger(__name__)

# Tiered country risk — sourced from Marble's approach to jurisdiction risk scoring
# Updated from FATF Public Statement (2024) and EU Non-Cooperative Jurisdictions list
COUNTRY_RISK_TIERS: dict[str, set[str]] = {
    # Tier 1 — Critical: FATF Black List (Call for Action / High-Risk)
    # Highest AML/CFT deficiencies; mandatory Enhanced Due Diligence
    "critical": {
        "KP",  # North Korea
        "IR",  # Iran
        "MM",  # Myanmar (Burma)
    },
    # Tier 2 — High: FATF Grey List (Increased Monitoring)
    # Jurisdictions under FATF scrutiny with action plans
    # Updated per FATF Public Statement (2023–2024):
    #   - SN (Senegal) removed from grey list (Oct 2023)
    #   - CM (Cameroon) added to grey list (Oct 2023)
    #   - NG (Nigeria) added to grey list (Feb 2023); under review 2025
    #   - BJ (Benin) on grey list since 2021
    "high": {
        "AF",  # Afghanistan
        "YE",  # Yemen
        "SY",  # Syria
        "SS",  # South Sudan
        "LY",  # Libya
        "SO",  # Somalia
        "HT",  # Haiti
        "PA",  # Panama (periodic re-listing)
        "PH",  # Philippines
        "VN",  # Vietnam
        "ML",  # Mali
        "CM",  # Cameroon (FATF Grey List, added Oct 2023)
        "CF",  # Central African Republic
        "CD",  # Democratic Republic of Congo
        "MZ",  # Mozambique
        "TZ",  # Tanzania
        "NG",  # Nigeria (FATF Grey List, added Feb 2023)
        "BJ",  # Benin (FATF Grey List, added 2021)
        "SD",  # Sudan
    },
    # Tier 3 — Elevated: EU Non-Cooperative Jurisdictions + OFAC sanctioned
    "elevated": {
        "RU",  # Russia (OFAC sanctions + EU listing)
        "BY",  # Belarus
        "CU",  # Cuba
        "VE",  # Venezuela
        "ZW",  # Zimbabwe
    },
}

# Kept for backward-compatibility — combines critical + high
HIGH_RISK_COUNTRIES: set[str] = (
    COUNTRY_RISK_TIERS["critical"] | COUNTRY_RISK_TIERS["high"]
)


class KYCService:
    """Manages customer KYC/KYB data and due diligence workflows."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def sync_customer_from_fineract(self, client_id: str) -> Customer:
        """Fetch client data from Fineract API and sync to local DB.

        Args:
            client_id: Fineract client ID.

        Returns:
            The synced Customer record.
        """
        fineract_data = await self._fetch_fineract_client(client_id)

        # Upsert customer
        result = await self.db.execute(
            select(Customer).where(Customer.fineract_client_id == client_id)
        )
        customer = result.scalar_one_or_none()

        now = datetime.now(UTC)

        if fineract_data:
            fields = self._map_fineract_to_customer(fineract_data)
            if customer:
                for key, value in fields.items():
                    setattr(customer, key, value)
                customer.last_synced_at = now
            else:
                customer = Customer(
                    fineract_client_id=client_id,
                    **fields,
                    last_synced_at=now,
                )
                self.db.add(customer)
        elif not customer:
            # Fineract unreachable — create stub record
            customer = Customer(
                fineract_client_id=client_id,
                full_name=f"Client {client_id}",
                customer_type=CustomerType.INDIVIDUAL,
                risk_level=CustomerRiskLevel.MEDIUM,  # Unknown = medium risk
            )
            self.db.add(customer)

        # Assess risk level and EDD requirements
        self._assess_risk(customer)

        await self.db.flush()
        logger.info("Customer %s synced: risk=%s, edd=%s", client_id, customer.risk_level.value, customer.edd_required)
        return customer

    async def get_customer(self, client_id: str) -> Customer | None:
        """Get a customer by Fineract client ID."""
        result = await self.db.execute(
            select(Customer).where(Customer.fineract_client_id == client_id)
        )
        return result.scalar_one_or_none()

    async def get_or_sync_customer(self, client_id: str) -> Customer:
        """Get a customer, syncing from Fineract if not cached."""
        customer = await self.get_customer(client_id)
        if not customer:
            customer = await self.sync_customer_from_fineract(client_id)
        return customer

    def _assess_risk(self, customer: Customer) -> None:
        """Determine customer risk level and EDD requirements using tiered country risk.

        Risk escalation (Marble-inspired tiered approach):
        - PEP → MEDIUM + EDD
        - Sanctioned → HIGH + EDD (immediate)
        - Critical country (FATF Black List) → HIGH + EDD
        - High country (FATF Grey List) → MEDIUM + EDD
        - Elevated country (EU non-cooperative) → MEDIUM (score penalty)
        - Entity without beneficial ownership → MEDIUM + EDD
        """
        edd_reasons: list[str] = []
        risk_level = CustomerRiskLevel.LOW

        def _country_tier(code: str | None) -> str | None:
            if not code:
                return None
            for tier, codes in COUNTRY_RISK_TIERS.items():
                if code.upper() in codes:
                    return tier
            return None

        if customer.is_sanctioned:
            edd_reasons.append("Sanctions match")
            risk_level = CustomerRiskLevel.HIGH

        if customer.is_pep:
            edd_reasons.append("PEP status")
            if risk_level != CustomerRiskLevel.HIGH:
                risk_level = CustomerRiskLevel.MEDIUM

        for country_field, label in [
            (customer.nationality, "nationality"),
            (customer.country_of_residence, "residence"),
        ]:
            tier = _country_tier(country_field)
            if tier == "critical":
                edd_reasons.append(f"Critical-risk {label}: {country_field} (FATF Black List)")
                risk_level = CustomerRiskLevel.HIGH
            elif tier == "high":
                edd_reasons.append(f"High-risk {label}: {country_field} (FATF Grey List)")
                if risk_level != CustomerRiskLevel.HIGH:
                    risk_level = CustomerRiskLevel.MEDIUM
            elif tier == "elevated":
                edd_reasons.append(f"Elevated-risk {label}: {country_field} (EU non-cooperative)")
                if risk_level == CustomerRiskLevel.LOW:
                    risk_level = CustomerRiskLevel.MEDIUM

        if (
            customer.customer_type == CustomerType.ENTITY
            and not customer.beneficial_owners
        ):
            edd_reasons.append("Entity without beneficial ownership data")
            if risk_level == CustomerRiskLevel.LOW:
                risk_level = CustomerRiskLevel.MEDIUM

        customer.risk_level = risk_level
        customer.edd_required = len(edd_reasons) > 0
        customer.edd_reason = "; ".join(edd_reasons) if edd_reasons else None

    async def _fetch_fineract_client(self, client_id: str) -> dict | None:
        """Fetch client data from Fineract REST API."""
        import os
        # Use proper TLS verification; set FINERACT_TLS_VERIFY=false only for local dev
        tls_verify = os.getenv("FINERACT_TLS_VERIFY", "true").lower() != "false"
        try:
            async with httpx.AsyncClient(verify=tls_verify, timeout=10) as client:
                response = await client.get(
                    f"{settings.fineract_base_url}/clients/{client_id}",
                    headers={"Fineract-Platform-TenantId": "default"},
                )
                if response.status_code == 200:
                    return response.json()
                logger.warning(
                    "Fineract client fetch returned %d for %s",
                    response.status_code, client_id,
                )
        except httpx.RequestError as e:
            logger.debug("Cannot reach Fineract API: %s", e)
        return None

    @staticmethod
    def _map_fineract_to_customer(data: dict) -> dict:
        """Map Fineract client JSON to Customer model fields."""
        # Build full name from Fineract fields
        first = data.get("firstname", "")
        middle = data.get("middlename", "")
        last = data.get("lastname", "")
        display = data.get("displayName", "")
        full_name = display or " ".join(filter(None, [first, middle, last])) or "Unknown"

        # Parse activation date as DOB proxy if available
        dob = None
        if "dateOfBirth" in data and data["dateOfBirth"]:
            parts = data["dateOfBirth"]
            if isinstance(parts, list) and len(parts) == 3:
                dob = datetime(parts[0], parts[1], parts[2])

        # Determine customer type
        is_entity = data.get("legalForm", {})
        customer_type = CustomerType.ENTITY if is_entity and is_entity.get("id") == 2 else CustomerType.INDIVIDUAL

        # Extract nationality and country from Fineract client data
        # Fineract stores these in activationDate, address, or custom fields
        client_data = data
        nationality = (
            client_data.get("nationality")
            or client_data.get("countryCode")
            or (client_data.get("address", [{}])[0].get("countryCode") if client_data.get("address") else None)
            or "CM"  # Default to Cameroon if not specified
        )
        country_of_residence = (
            client_data.get("countryOfResidence")
            or nationality
            or "CM"
        )

        return {
            "full_name": full_name,
            "customer_type": customer_type,
            "date_of_birth": dob,
            "email": data.get("emailAddress"),
            "phone": data.get("mobileNo"),
            "nationality": nationality,
            "country_of_residence": country_of_residence,
        }
