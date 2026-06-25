"""Service layer for credit scoring, profiles, and request management.

Orchestrates credit feature extraction, scoring, profile management,
and the compliance review workflow for credit requests.
"""

import json
import logging
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.features.credit_extractor import CreditFeatureExtractor
from app.ml.credit_scorer import CreditClusterModel, CreditScorer
from app.models.alert import Alert, AlertStatus
from app.models.credit_profile import (
    CreditSegment,
    CustomerCreditProfile,
    ScoringMethod,
)
from app.models.credit_request import (
    CreditRecommendation,
    CreditRequest,
    CreditRequestStatus,
)
from app.models.transaction import Transaction

logger = logging.getLogger(__name__)


class CreditService:
    """Handles credit scoring, profile management, and request workflow."""

    def __init__(self, db: AsyncSession):
        self.db = db

    # ── Transaction & Alert Queries ───────────────────────────

    async def get_client_transactions(
        self, client_id: str, days: int = 180
    ) -> list[Transaction]:
        """Get all transactions for a client within a time window.

        Args:
            client_id: Fineract client ID.
            days: Number of days of history to retrieve.

        Returns:
            List of transactions ordered by date descending.
        """
        cutoff = datetime.now(UTC) - timedelta(days=days)
        result = await self.db.execute(
            select(Transaction)
            .where(Transaction.fineract_client_id == client_id)
            .where(Transaction.transaction_date >= cutoff)
            .order_by(Transaction.transaction_date.desc())
        )
        return list(result.scalars().all())

    async def get_fraud_alert_info(self, client_id: str) -> tuple[int, int | None]:
        """Get fraud alert statistics for a client.

        Returns:
            (total_confirmed_fraud_alerts, days_since_most_recent_fraud_alert).
            days_since is None if no fraud alerts exist.
        """
        # Count confirmed fraud alerts for this client's transactions
        result = await self.db.execute(
            select(func.count(Alert.id))
            .join(Transaction, Alert.transaction_id == Transaction.id)
            .where(Transaction.fineract_client_id == client_id)
            .where(Alert.status == AlertStatus.CONFIRMED_FRAUD)
        )
        total = result.scalar_one()

        if total == 0:
            return 0, None

        # Most recent fraud alert date
        result = await self.db.execute(
            select(func.max(Alert.created_at))
            .join(Transaction, Alert.transaction_id == Transaction.id)
            .where(Transaction.fineract_client_id == client_id)
            .where(Alert.status == AlertStatus.CONFIRMED_FRAUD)
        )
        latest_date = result.scalar_one()
        if latest_date:
            days_since = (datetime.now(UTC) - latest_date).days
        else:
            days_since = None

        return total, days_since

    async def get_account_age_days(self, client_id: str) -> int:
        """Get the age of a client's account in days (since first transaction)."""
        result = await self.db.execute(
            select(func.min(Transaction.transaction_date))
            .where(Transaction.fineract_client_id == client_id)
        )
        first_date = result.scalar_one()
        if first_date:
            return (datetime.now(UTC) - first_date).days
        return 0

    # ── Credit Profile Management ─────────────────────────────

    async def compute_credit_profile(
        self, client_id: str
    ) -> CustomerCreditProfile:
        """Compute or update a customer's credit profile.

        Orchestrates the full scoring pipeline:
        1. Fetch transaction history (180 days)
        2. Get fraud alert info
        3. Extract credit features
        4. Run rule-based scoring
        5. Run ML clustering (if model available)
        6. Upsert the profile

        Args:
            client_id: Fineract client ID.

        Returns:
            The created or updated CustomerCreditProfile.
        """
        transactions = await self.get_client_transactions(client_id, days=180)

        if len(transactions) < settings.credit_min_transactions:
            logger.info(
                "Client %s has only %d transactions (need %d), assigning Tier E",
                client_id, len(transactions), settings.credit_min_transactions,
            )
            return await self._upsert_profile(
                client_id,
                credit_score=0.0,
                segment=CreditSegment.TIER_E,
                max_credit_amount=0.0,
                score_components="{}",
                scoring_method=ScoringMethod.RULE_BASED,
            )

        # Gather metadata
        fraud_count, days_since_fraud = await self.get_fraud_alert_info(client_id)
        account_age = await self.get_account_age_days(client_id)

        # Extract features
        features = CreditFeatureExtractor.extract(
            transactions, fraud_count, days_since_fraud, account_age
        )

        # Rule-based scoring
        scorer = CreditScorer()
        score, components = scorer.score(features)
        segment = scorer.classify_segment(score)
        max_amount = scorer.compute_max_amount(segment)

        # ML clustering (if available)
        ml_cluster_id = None
        ml_segment = None
        scoring_method = ScoringMethod.RULE_BASED

        cluster_model = CreditClusterModel()
        if cluster_model.is_ready:
            ml_cluster_id, ml_segment = cluster_model.predict(features)
            scoring_method = ScoringMethod.HYBRID

        return await self._upsert_profile(
            client_id,
            credit_score=score,
            segment=segment,
            max_credit_amount=max_amount,
            score_components=json.dumps(components),
            scoring_method=scoring_method,
            ml_cluster_id=ml_cluster_id,
            ml_segment_suggestion=ml_segment,
        )

    async def _upsert_profile(
        self,
        client_id: str,
        credit_score: float,
        segment: CreditSegment,
        max_credit_amount: float,
        score_components: str,
        scoring_method: ScoringMethod,
        ml_cluster_id: int | None = None,
        ml_segment_suggestion: CreditSegment | None = None,
    ) -> CustomerCreditProfile:
        """Create or update a credit profile."""
        result = await self.db.execute(
            select(CustomerCreditProfile).where(
                CustomerCreditProfile.fineract_client_id == client_id
            )
        )
        profile = result.scalar_one_or_none()

        now = datetime.now(UTC)
        if profile:
            profile.credit_score = credit_score
            profile.segment = segment
            profile.max_credit_amount = max_credit_amount
            profile.score_components = score_components
            profile.scoring_method = scoring_method
            profile.ml_cluster_id = ml_cluster_id
            profile.ml_segment_suggestion = ml_segment_suggestion
            profile.last_computed_at = now
        else:
            profile = CustomerCreditProfile(
                fineract_client_id=client_id,
                credit_score=credit_score,
                segment=segment,
                max_credit_amount=max_credit_amount,
                score_components=score_components,
                scoring_method=scoring_method,
                ml_cluster_id=ml_cluster_id,
                ml_segment_suggestion=ml_segment_suggestion,
                last_computed_at=now,
                is_active=True,
            )
            self.db.add(profile)

        await self.db.flush()
        logger.info(
            "Credit profile updated for client %s: score=%.2f, segment=%s, max=%,.0f",
            client_id, credit_score, segment.value, max_credit_amount,
        )
        return profile

    async def get_profile(self, client_id: str) -> CustomerCreditProfile | None:
        """Get a customer's credit profile by client ID."""
        result = await self.db.execute(
            select(CustomerCreditProfile).where(
                CustomerCreditProfile.fineract_client_id == client_id
            )
        )
        return result.scalar_one_or_none()

    async def list_profiles(
        self,
        page: int = 1,
        page_size: int = 50,
        segment: CreditSegment | None = None,
    ) -> tuple[list[CustomerCreditProfile], int]:
        """List credit profiles with optional filtering and pagination."""
        query = select(CustomerCreditProfile).where(
            CustomerCreditProfile.is_active == True  # noqa: E712
        )
        count_query = select(func.count(CustomerCreditProfile.id)).where(
            CustomerCreditProfile.is_active == True  # noqa: E712
        )

        if segment:
            query = query.where(CustomerCreditProfile.segment == segment)
            count_query = count_query.where(CustomerCreditProfile.segment == segment)

        query = query.order_by(CustomerCreditProfile.credit_score.desc())
        query = query.offset((page - 1) * page_size).limit(page_size)

        result = await self.db.execute(query)
        profiles = list(result.scalars().all())

        count_result = await self.db.execute(count_query)
        total = count_result.scalar_one()

        return profiles, total

    # ── Credit Score Gaming Detection ────────────────────────────

    async def _detect_score_inflation(self, client_id: str) -> bool:
        """Detect whether recent deposits are abnormally high vs. the 30-day average.

        Pattern: a user makes many deposits in the 7-14 days before applying for credit
        to inflate deposit_consistency and net_monthly_flow components, then plans to
        withdraw immediately after the loan is approved.

        Returns True if score inflation is suspected.
        """
        transactions = await self.get_client_transactions(client_id, days=30)
        if len(transactions) < settings.credit_min_transactions:
            return False

        # Sum deposits in last 7 days vs average weekly deposit over 30 days
        cutoff_7d = datetime.now(UTC) - timedelta(days=7)
        recent_inflow = sum(
            t.amount for t in transactions
            if t.transaction_date >= cutoff_7d
            and t.transaction_type.value == "deposit"
        )
        total_inflow_30d = sum(
            t.amount for t in transactions
            if t.transaction_type.value == "deposit"
        )
        # Average weekly inflow over 30 days (30/7 ≈ 4.3 weeks)
        avg_weekly_inflow = total_inflow_30d / 4.3 if total_inflow_30d > 0 else 0

        if avg_weekly_inflow == 0:
            return False

        multiplier = settings.credit_gaming_inflow_multiplier
        return recent_inflow > avg_weekly_inflow * multiplier

    # ── Credit Request Workflow ────────────────────────────────

    async def create_credit_request(
        self, client_id: str, requested_amount: float
    ) -> CreditRequest:
        """Submit a credit request with auto-generated recommendation.

        Always creates with PENDING_REVIEW status — compliance must review.

        Args:
            client_id: Fineract client ID.
            requested_amount: Amount the customer wants to borrow.

        Returns:
            The created CreditRequest with recommendation.
        """
        # Compute/refresh the credit profile first
        profile = await self.compute_credit_profile(client_id)

        # Detect credit score gaming before generating recommendation
        score_inflation_flag = await self._detect_score_inflation(client_id)

        # Apply inflation penalty and force REVIEW_CAREFULLY if gaming detected
        effective_score = profile.credit_score
        if score_inflation_flag:
            effective_score = max(0.0, profile.credit_score - settings.credit_gaming_score_penalty)
            logger.warning(
                "Score inflation detected for client %s: raw=%.2f, adjusted=%.2f",
                client_id, profile.credit_score, effective_score,
            )

        # Generate recommendation based on effective (potentially penalized) score
        scorer = CreditScorer()
        if score_inflation_flag:
            recommendation = CreditRecommendation.REVIEW_CAREFULLY
        else:
            recommendation = scorer.recommend(
                score=effective_score,
                segment=profile.segment,
                requested_amount=requested_amount,
                max_amount=profile.max_credit_amount,
            )

        inflation_note = (
            "POSSIBLE SCORE INFLATION DETECTED: Recent inflow significantly exceeds "
            "30-day average. Manual verification of deposit source required. "
            f"Raw score: {profile.credit_score:.2f}, Adjusted: {effective_score:.2f}. "
            if score_inflation_flag else ""
        )

        request = CreditRequest(
            fineract_client_id=client_id,
            requested_amount=requested_amount,
            credit_score_at_request=effective_score,
            segment_at_request=profile.segment,
            max_credit_at_request=profile.max_credit_amount,
            recommendation=recommendation,
            status=CreditRequestStatus.PENDING_REVIEW,
            score_inflation_flag=score_inflation_flag,
            reviewer_notes=inflation_note if score_inflation_flag else None,
        )
        self.db.add(request)
        await self.db.flush()

        # Generate LLM explanation (customer + compliance versions)
        if settings.llm_investigation_enabled and settings.anthropic_api_key:
            try:
                explanation = await self._generate_credit_explanation(
                    client_id=client_id,
                    score=effective_score,
                    segment=profile.segment,
                    recommendation=recommendation,
                    requested_amount=requested_amount,
                    max_amount=profile.max_credit_amount,
                    score_inflation_flag=score_inflation_flag,
                    score_components=profile.score_components,
                )
                request.explanation_text = explanation
            except Exception as exc:
                logger.warning("LLM credit explanation failed for %s: %s", client_id, exc)

        logger.info(
            "Credit request created for client %s: amount=%.0f, score=%.2f, "
            "recommendation=%s",
            client_id, requested_amount, profile.credit_score,
            recommendation.value,
        )
        return request

    async def _generate_credit_explanation(
        self,
        client_id: str,
        score: float,
        segment,
        recommendation,
        requested_amount: float,
        max_amount: float,
        score_inflation_flag: bool,
        score_components: str,
    ) -> str:
        """Call Claude to generate dual-audience credit decision explanations."""
        import anthropic

        components = {}
        try:
            components = json.loads(score_components) if score_components else {}
        except (json.JSONDecodeError, TypeError):
            pass

        components_text = "\n".join(
            f"- {k}: {v:.2f}" for k, v in components.items() if isinstance(v, (int, float))
        ) or "Non disponible"

        prompt = f"""Tu es un assistant conformité pour WeBank Cameroun.

Génère une explication de décision de crédit en deux parties:

**DONNÉES:**
- Client ID: {client_id}
- Score de crédit: {score:.2f}/1.0
- Segment: {segment.value}
- Montant demandé: {requested_amount:,.0f} XAF
- Montant maximum autorisé: {max_amount:,.0f} XAF
- Recommandation système: {recommendation.value}
- Alerte gonflement score: {'Oui' if score_inflation_flag else 'Non'}

**COMPOSANTES DU SCORE:**
{components_text}

---

**PARTIE 1 — EXPLICATION CLIENT (en français, simple et direct, 100-150 mots):**
Explique la décision de façon compréhensible pour un particulier.
Si refus ou révision: donne 2-3 conseils concrets pour améliorer son score.
Ne mentionne pas les détails techniques de scoring.

**PARTIE 2 — EXPLICATION CONFORMITÉ (en français, technique, 80-100 mots):**
Résume les facteurs clés du score, l'indicateur d'inflation si présent,
et la justification de la recommandation pour le dossier de conformité."""

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        response = client.messages.create(
            model=settings.llm_model,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text if response.content else ""

    async def list_credit_requests(
        self,
        page: int = 1,
        page_size: int = 50,
        status: CreditRequestStatus | None = None,
    ) -> tuple[list[CreditRequest], int]:
        """List credit requests with optional status filtering."""
        query = select(CreditRequest)
        count_query = select(func.count(CreditRequest.id))

        if status:
            query = query.where(CreditRequest.status == status)
            count_query = count_query.where(CreditRequest.status == status)

        # Pending first, then by creation date
        query = query.order_by(CreditRequest.status.asc(), CreditRequest.created_at.desc())
        query = query.offset((page - 1) * page_size).limit(page_size)

        result = await self.db.execute(query)
        requests = list(result.scalars().all())

        count_result = await self.db.execute(count_query)
        total = count_result.scalar_one()

        return requests, total

    async def get_credit_request(self, request_id: UUID) -> CreditRequest | None:
        """Get a specific credit request by ID."""
        result = await self.db.execute(
            select(CreditRequest).where(CreditRequest.id == request_id)
        )
        return result.scalar_one_or_none()

    async def review_credit_request(
        self,
        request_id: UUID,
        status: CreditRequestStatus,
        reviewer_id: UUID,
        notes: str | None = None,
    ) -> CreditRequest:
        """Approve or reject a credit request.

        Args:
            request_id: The credit request UUID.
            status: Must be APPROVED or REJECTED.
            reviewer_id: UUID of the reviewing user.
            notes: Optional reviewer comments.

        Returns:
            The updated CreditRequest.

        Raises:
            ValueError: If the request is not in PENDING_REVIEW status.
        """
        result = await self.db.execute(
            select(CreditRequest).where(CreditRequest.id == request_id)
        )
        request = result.scalar_one()

        if request.status != CreditRequestStatus.PENDING_REVIEW:
            raise ValueError(
                f"Cannot review request in {request.status.value} status"
            )

        request.status = status
        request.reviewed_by = reviewer_id
        request.reviewed_at = datetime.now(UTC)
        request.reviewer_notes = notes

        await self.db.flush()
        logger.info(
            "Credit request %s reviewed: %s by %s",
            request_id, status.value, reviewer_id,
        )
        return request

    # ── Analytics ─────────────────────────────────────────────

    async def get_segment_stats(self) -> list[dict]:
        """Get aggregate statistics per credit segment.

        Returns:
            List of dicts with segment, count, avg_score, avg_max_amount.
        """
        result = await self.db.execute(
            select(
                CustomerCreditProfile.segment,
                func.count(CustomerCreditProfile.id).label("count"),
                func.avg(CustomerCreditProfile.credit_score).label("avg_score"),
                func.avg(CustomerCreditProfile.max_credit_amount).label("avg_max_amount"),
            )
            .where(CustomerCreditProfile.is_active == True)  # noqa: E712
            .group_by(CustomerCreditProfile.segment)
            .order_by(CustomerCreditProfile.segment)
        )
        return [
            {
                "segment": row.segment,
                "count": row.count,
                "avg_score": float(row.avg_score) if row.avg_score else 0.0,
                "avg_max_amount": float(row.avg_max_amount) if row.avg_max_amount else 0.0,
            }
            for row in result.all()
        ]

    async def get_analytics(self) -> dict:
        """Get comprehensive credit analytics for the dashboard."""
        segment_stats = await self.get_segment_stats()

        total_profiles = sum(s["count"] for s in segment_stats)
        avg_score = (
            sum(s["avg_score"] * s["count"] for s in segment_stats) / total_profiles
            if total_profiles > 0
            else 0.0
        )

        pending = (
            await self.db.execute(
                select(func.count(CreditRequest.id)).where(
                    CreditRequest.status == CreditRequestStatus.PENDING_REVIEW
                )
            )
        ).scalar_one()

        approved = (
            await self.db.execute(
                select(func.count(CreditRequest.id)).where(
                    CreditRequest.status == CreditRequestStatus.APPROVED
                )
            )
        ).scalar_one()

        rejected = (
            await self.db.execute(
                select(func.count(CreditRequest.id)).where(
                    CreditRequest.status == CreditRequestStatus.REJECTED
                )
            )
        ).scalar_one()

        return {
            "segment_distribution": segment_stats,
            "total_profiles": total_profiles,
            "avg_credit_score": avg_score,
            "total_pending_requests": pending,
            "total_approved": approved,
            "total_rejected": rejected,
        }
