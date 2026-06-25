"""Pydantic schemas for credit scoring endpoints."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from app.models.credit_profile import CreditSegment, ScoringMethod
from app.models.credit_request import CreditRecommendation, CreditRequestStatus

# ── Credit Profile ────────────────────────────────────────


class CreditProfileResponse(BaseModel):
    """Credit profile data returned to the client."""

    id: UUID
    fineract_client_id: str
    credit_score: float
    segment: CreditSegment
    max_credit_amount: float
    score_components: str | None
    ml_cluster_id: int | None
    ml_segment_suggestion: CreditSegment | None
    scoring_method: ScoringMethod
    last_computed_at: datetime
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class CreditProfileListResponse(BaseModel):
    """Paginated list of credit profiles."""

    items: list[CreditProfileResponse]
    total: int
    page: int
    page_size: int


# ── Credit Request ────────────────────────────────────────


class CreditRequestCreate(BaseModel):
    """Submit a new credit request for compliance review."""

    fineract_client_id: str = Field(..., description="Fineract client ID")
    requested_amount: float = Field(..., gt=0, description="Loan amount requested")


class CreditRequestResponse(BaseModel):
    """Credit request data returned to the client."""

    id: UUID
    fineract_client_id: str
    requested_amount: float
    credit_score_at_request: float
    segment_at_request: CreditSegment
    max_credit_at_request: float
    recommendation: CreditRecommendation
    status: CreditRequestStatus
    reviewer_notes: str | None
    assigned_to: UUID | None
    reviewed_at: datetime | None
    reviewed_by: UUID | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class CreditRequestListResponse(BaseModel):
    """Paginated list of credit requests."""

    items: list[CreditRequestResponse]
    total: int
    page: int
    page_size: int


class CreditReviewAction(BaseModel):
    """Compliance reviewer approves or rejects a credit request."""

    status: CreditRequestStatus = Field(
        ..., description="Must be 'approved' or 'rejected'"
    )
    reviewer_notes: str | None = Field(
        default=None, description="Reviewer comments"
    )


# ── Analytics ─────────────────────────────────────────────


class CreditSegmentStats(BaseModel):
    """Aggregate statistics for a credit segment tier."""

    segment: CreditSegment
    count: int
    avg_score: float
    avg_max_amount: float


class CreditAnalytics(BaseModel):
    """Dashboard analytics for the credit scoring system."""

    segment_distribution: list[CreditSegmentStats]
    total_profiles: int
    avg_credit_score: float
    total_pending_requests: int
    total_approved: int
    total_rejected: int
