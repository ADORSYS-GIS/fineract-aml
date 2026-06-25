"""LLM-powered alert investigation agent using the Claude API with tool use.

When a HIGH or CRITICAL alert is created, this agent autonomously:
1. Fetches the transaction history for the flagged account
2. Reviews the customer's KYC and credit profile
3. Checks prior alerts for the same client
4. Identifies the closest FATF AML typology
5. Generates a structured investigation report including a French SAR narrative

All tool calls are resolved against the live database — the LLM never
has direct DB access; it requests data through declared tools.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID

logger = logging.getLogger(__name__)


@dataclass
class InvestigationReport:
    """Structured output from the LLM alert investigation agent."""

    alert_id: str
    summary: str                           # 2-3 sentence plain-language description
    typology_match: str                    # Closest FATF typology
    risk_factors: list[str]               # Specific findings that increase risk
    mitigating_factors: list[str]         # Factors suggesting legitimate activity
    recommendation: str                   # "dismiss" | "monitor" | "escalate" | "file_sar"
    recommended_actions: list[str]        # Next steps for the analyst
    narrative_fr: str                     # SAR-ready narrative in French (COBAC)
    model_used: str = "claude-opus-4-6"
    generated_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )


_SYSTEM_PROMPT = """Tu es un expert en conformité AML (Anti-Money Laundering) spécialisé dans
le système bancaire WeBank au Cameroun, réglementé par la COBAC et le CEMAC.

WeBank est une plateforme bancaire numérique multi-acteurs:
- CUSTOMERS: portefeuilles numériques pour particuliers
- AGENTS: opérateurs cash-in/cash-out avec float
- MERCHANTS: marchands acceptant les paiements QR

Ta mission est d'analyser les alertes de suspicion de blanchiment d'argent.
Pour chaque alerte, tu dois:
1. Collecter les informations pertinentes via les outils disponibles
2. Identifier le type d'activité suspecte selon les typologies FATF
3. Rédiger un rapport d'investigation structuré
4. Rédiger un projet de déclaration de soupçon (DS) en français pour la COBAC

Les typologies FATF pertinentes pour WeBank:
- Structuration (smurf): dépôts multiples sous le seuil de déclaration
- Layering (stratification): transferts circulaires A→B→C→A
- Scatter-gather: collecte de petits montants puis virement unique
- Bipartite layering: fan-out puis fan-in
- Stacking: transferts séquentiels rapides proportionnels
- Agent structuring: agent traitant de nombreux dépôts sous-seuil
- Loan-and-run: remboursement immédiat du prêt après décaissement
- Account farming: création de faux comptes via agent

Réponds TOUJOURS en format JSON structuré selon le schéma demandé.
La narrative_fr doit être en français professionnel, adapté à la COBAC."""


_TOOLS = [
    {
        "name": "get_transaction_history",
        "description": "Get recent transaction history for an account",
        "input_schema": {
            "type": "object",
            "properties": {
                "account_id": {"type": "string", "description": "Fineract account ID"},
                "days": {"type": "integer", "description": "Number of days of history", "default": 30},
            },
            "required": ["account_id"],
        },
    },
    {
        "name": "get_customer_profile",
        "description": "Get KYC profile and risk level for a customer",
        "input_schema": {
            "type": "object",
            "properties": {
                "client_id": {"type": "string", "description": "Fineract client ID"},
            },
            "required": ["client_id"],
        },
    },
    {
        "name": "get_credit_profile",
        "description": "Get credit score, segment, and disbursement history for a customer",
        "input_schema": {
            "type": "object",
            "properties": {
                "client_id": {"type": "string", "description": "Fineract client ID"},
            },
            "required": ["client_id"],
        },
    },
    {
        "name": "get_related_alerts",
        "description": "Get prior AML alerts for a customer",
        "input_schema": {
            "type": "object",
            "properties": {
                "client_id": {"type": "string", "description": "Fineract client ID"},
                "limit": {"type": "integer", "description": "Max alerts to return", "default": 5},
            },
            "required": ["client_id"],
        },
    },
    {
        "name": "get_agent_profile",
        "description": "Get behavioral baseline for an agent (if actor_type is agent)",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Agent ID"},
            },
            "required": ["agent_id"],
        },
    },
]


class AlertInvestigationAgent:
    """Uses the Claude API with tool use to investigate an AML alert."""

    def __init__(self, db):
        self.db = db

    async def investigate(self, alert_id: UUID) -> InvestigationReport | None:
        """Run the full investigation pipeline for an alert.

        Returns None if the LLM API is not configured or if the alert is not found.
        """
        from app.core.config import settings

        if not settings.anthropic_api_key:
            logger.warning("Anthropic API key not configured; skipping LLM investigation")
            return None

        # Load the alert and transaction
        alert_data = await self._load_alert_context(alert_id)
        if not alert_data:
            return None

        try:
            import anthropic
        except ImportError:
            logger.error("anthropic package not installed; run: pip install anthropic")
            return None

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

        user_message = self._build_user_message(alert_data)
        messages = [{"role": "user", "content": user_message}]

        # Agentic tool-use loop
        for _iteration in range(10):  # max 10 tool call rounds
            response = client.messages.create(
                model=settings.llm_model,
                max_tokens=4096,
                system=_SYSTEM_PROMPT,
                tools=_TOOLS,
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                # Extract final JSON report from last text block
                for block in response.content:
                    if hasattr(block, "text"):
                        return self._parse_report(alert_id, block.text)
                break

            if response.stop_reason == "tool_use":
                # Process tool calls and add results to conversation
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = await self._handle_tool_call(
                            block.name, block.input
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result, default=str),
                        })

                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
            else:
                break

        logger.warning("LLM investigation for alert %s did not complete cleanly", alert_id)
        return None

    async def _load_alert_context(self, alert_id: UUID) -> dict | None:
        """Load basic alert + transaction context to seed the conversation."""
        from sqlalchemy import select

        from app.models.alert import Alert
        from app.models.transaction import Transaction

        result = await self.db.execute(
            select(Alert).where(Alert.id == alert_id)
        )
        alert = result.scalar_one_or_none()
        if not alert:
            logger.warning("Alert %s not found for LLM investigation", alert_id)
            return None

        result = await self.db.execute(
            select(Transaction).where(Transaction.id == alert.transaction_id)
        )
        tx = result.scalar_one_or_none()
        if not tx:
            return None

        return {
            "alert_id": str(alert_id),
            "alert_title": alert.title,
            "alert_source": alert.source.value,
            "alert_risk_score": alert.risk_score,
            "triggered_rules": alert.triggered_rules,
            "transaction": {
                "id": str(tx.id),
                "type": tx.transaction_type.value,
                "amount": tx.amount,
                "currency": tx.currency,
                "date": tx.transaction_date.isoformat(),
                "account_id": tx.fineract_account_id,
                "client_id": tx.fineract_client_id,
                "actor_type": tx.actor_type,
                "agent_id": tx.agent_id,
                "merchant_id": tx.merchant_id,
                "counterparty_account_id": tx.counterparty_account_id,
                "counterparty_name": tx.counterparty_name,
                "country_code": tx.country_code,
            },
        }

    def _build_user_message(self, alert_data: dict) -> str:
        return (
            f"Analyse cette alerte de suspicion WeBank et génère un rapport d'investigation:\n\n"
            f"```json\n{json.dumps(alert_data, indent=2, default=str)}\n```\n\n"
            "Utilise les outils disponibles pour collecter le contexte nécessaire, "
            "puis génère ton rapport au format JSON avec ces champs:\n"
            "- summary (str): résumé en 2-3 phrases\n"
            "- typology_match (str): typologie FATF la plus proche\n"
            "- risk_factors (list[str]): facteurs de risque identifiés\n"
            "- mitigating_factors (list[str]): facteurs atténuants\n"
            "- recommendation (str): 'dismiss' | 'monitor' | 'escalate' | 'file_sar'\n"
            "- recommended_actions (list[str]): prochaines étapes pour l'analyste\n"
            "- narrative_fr (str): projet de déclaration de soupçon COBAC en français\n"
        )

    async def _handle_tool_call(self, tool_name: str, tool_input: dict) -> dict:
        """Resolve a tool call against the live database."""
        try:
            if tool_name == "get_transaction_history":
                return await self._tool_get_transaction_history(
                    tool_input["account_id"],
                    tool_input.get("days", 30),
                )
            elif tool_name == "get_customer_profile":
                return await self._tool_get_customer_profile(tool_input["client_id"])
            elif tool_name == "get_credit_profile":
                return await self._tool_get_credit_profile(tool_input["client_id"])
            elif tool_name == "get_related_alerts":
                return await self._tool_get_related_alerts(
                    tool_input["client_id"],
                    tool_input.get("limit", 5),
                )
            elif tool_name == "get_agent_profile":
                return await self._tool_get_agent_profile(tool_input["agent_id"])
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as exc:
            logger.warning("Tool call '%s' failed: %s", tool_name, exc)
            return {"error": str(exc)}

    async def _tool_get_transaction_history(self, account_id: str, days: int) -> dict:
        from app.services.transaction_service import TransactionService
        svc = TransactionService(self.db)
        txns = await svc.get_account_history(account_id, window_minutes=days * 1440)
        return {
            "account_id": account_id,
            "transaction_count": len(txns),
            "transactions": [
                {
                    "date": t.transaction_date.isoformat(),
                    "type": t.transaction_type.value,
                    "amount": t.amount,
                    "currency": t.currency,
                    "counterparty": t.counterparty_account_id,
                    "risk_score": t.risk_score,
                }
                for t in txns[:30]  # Cap at 30 to keep context manageable
            ],
        }

    async def _tool_get_customer_profile(self, client_id: str) -> dict:
        from app.services.kyc_service import KYCService
        kyc = KYCService(self.db)
        customer = await kyc.get_customer(client_id)
        if not customer:
            return {"error": f"Customer {client_id} not found"}
        return {
            "client_id": client_id,
            "full_name": customer.full_name,
            "customer_type": customer.customer_type.value,
            "nationality": customer.nationality,
            "country_of_residence": customer.country_of_residence,
            "kyc_verified": customer.kyc_verified,
            "risk_level": customer.risk_level.value,
            "is_pep": customer.is_pep,
            "is_sanctioned": customer.is_sanctioned,
            "edd_required": customer.edd_required,
            "edd_reason": customer.edd_reason,
        }

    async def _tool_get_credit_profile(self, client_id: str) -> dict:
        from sqlalchemy import select

        from app.models.credit_profile import CustomerCreditProfile
        from app.models.credit_request import CreditRequest

        result = await self.db.execute(
            select(CustomerCreditProfile).where(
                CustomerCreditProfile.fineract_client_id == client_id
            )
        )
        profile = result.scalar_one_or_none()
        if not profile:
            return {"error": f"No credit profile for client {client_id}"}

        result = await self.db.execute(
            select(CreditRequest)
            .where(CreditRequest.fineract_client_id == client_id)
            .order_by(CreditRequest.created_at.desc())
            .limit(3)
        )
        recent_requests = list(result.scalars().all())

        return {
            "credit_score": profile.credit_score,
            "segment": profile.segment.value,
            "max_credit_amount": profile.max_credit_amount,
            "scoring_method": profile.scoring_method.value,
            "last_computed": profile.last_computed_at.isoformat() if profile.last_computed_at else None,
            "recent_credit_requests": [
                {
                    "amount": r.requested_amount,
                    "recommendation": r.recommendation.value,
                    "status": r.status.value,
                    "inflation_flag": r.score_inflation_flag,
                    "date": r.created_at.isoformat(),
                }
                for r in recent_requests
            ],
        }

    async def _tool_get_related_alerts(self, client_id: str, limit: int) -> dict:
        from sqlalchemy import select

        from app.models.alert import Alert
        from app.models.transaction import Transaction

        result = await self.db.execute(
            select(Alert)
            .join(Transaction, Alert.transaction_id == Transaction.id)
            .where(Transaction.fineract_client_id == client_id)
            .order_by(Alert.created_at.desc())
            .limit(limit)
        )
        alerts = list(result.scalars().all())
        return {
            "client_id": client_id,
            "alert_count": len(alerts),
            "alerts": [
                {
                    "id": str(a.id),
                    "status": a.status.value,
                    "source": a.source.value,
                    "risk_score": a.risk_score,
                    "title": a.title,
                    "date": a.created_at.isoformat(),
                }
                for a in alerts
            ],
        }

    async def _tool_get_agent_profile(self, agent_id: str) -> dict:
        from sqlalchemy import select

        from app.models.agent_profile import AgentProfile

        result = await self.db.execute(
            select(AgentProfile).where(AgentProfile.agent_id == agent_id)
        )
        profile = result.scalar_one_or_none()
        if not profile:
            return {"error": f"No agent profile for agent {agent_id}"}
        return {
            "agent_id": agent_id,
            "branch_id": profile.branch_id,
            "avg_daily_tx_count": profile.avg_daily_tx_count_30d,
            "avg_daily_volume": profile.avg_daily_volume_30d,
            "typical_float_ratio": profile.typical_float_ratio,
            "unique_customers_30d": profile.unique_customers_30d,
            "avg_new_customers_per_day": profile.avg_new_customers_per_day_30d,
        }

    def _parse_report(self, alert_id: UUID, text: str) -> InvestigationReport:
        """Parse the LLM's JSON output into an InvestigationReport."""
        # Extract JSON from the response (may be wrapped in markdown code blocks)
        import re
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try parsing the entire text as JSON
            json_str = text.strip()

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("Could not parse LLM JSON response for alert %s", alert_id)
            return InvestigationReport(
                alert_id=str(alert_id),
                summary="Investigation could not be completed — JSON parse error.",
                typology_match="unknown",
                risk_factors=[],
                mitigating_factors=[],
                recommendation="monitor",
                recommended_actions=["Manual review required"],
                narrative_fr="Erreur lors de la génération automatique du rapport.",
            )

        return InvestigationReport(
            alert_id=str(alert_id),
            summary=data.get("summary", ""),
            typology_match=data.get("typology_match", "unknown"),
            risk_factors=data.get("risk_factors", []),
            mitigating_factors=data.get("mitigating_factors", []),
            recommendation=data.get("recommendation", "monitor"),
            recommended_actions=data.get("recommended_actions", []),
            narrative_fr=data.get("narrative_fr", ""),
        )
