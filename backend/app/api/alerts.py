"""Alert endpoints for the compliance dashboard."""

from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import UserRole, require_role, verify_token
from app.models.alert import AlertStatus
from app.schemas.alert import AlertAssign, AlertListResponse, AlertResponse, AlertStatusUpdate
from app.schemas.review import ReviewCreate, ReviewResponse
from app.services.alert_service import AlertService
from app.services.audit_service import AuditService

router = APIRouter(
    prefix="/alerts", tags=["Alerts"], dependencies=[Depends(verify_token)]
)

_TERMINAL_STATUSES = {"confirmed_fraud", "false_positive", "dismissed", "closed"}


@router.get("", response_model=AlertListResponse, dependencies=[Depends(require_role(UserRole.ANALYST, UserRole.MLRO, UserRole.ADMIN))])
async def list_alerts(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    status_filter: AlertStatus | None = Query(None, alias="status"),
    assigned_to: UUID | None = None,
    db: AsyncSession = Depends(get_db),
):
    """List alerts, sorted by risk score (highest first)."""
    service = AlertService(db)
    alerts, total = await service.list_alerts(page, page_size, status_filter, assigned_to)
    return AlertListResponse(
        items=[AlertResponse.model_validate(a) for a in alerts],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(alert_id: UUID, db: AsyncSession = Depends(get_db)):
    """Get a single alert with its transaction and reviews."""
    service = AlertService(db)
    alert = await service.get_alert(alert_id)
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found"
        )
    return AlertResponse.model_validate(alert)


@router.patch("/{alert_id}/assign", response_model=AlertResponse)
async def assign_alert(
    alert_id: UUID,
    data: AlertAssign,
    db: AsyncSession = Depends(get_db),
):
    """Assign an alert to a compliance analyst."""
    service = AlertService(db)
    alert = await service.assign_alert(alert_id, data.assigned_to)
    return AlertResponse.model_validate(alert)


@router.patch("/{alert_id}/status", response_model=AlertResponse, dependencies=[Depends(require_role(UserRole.ANALYST, UserRole.MLRO, UserRole.ADMIN))])
async def update_alert_status(
    alert_id: UUID,
    data: AlertStatusUpdate,
    current_user: dict = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """Update the status of an alert."""
    service = AlertService(db)
    alert = await service.update_status(alert_id, data.status)

    # COBAC audit trail — record who closed the alert and when
    new_status = data.status.value if hasattr(data.status, "value") else str(data.status)
    if new_status.lower() in _TERMINAL_STATUSES:
        alert.closed_at = datetime.now(UTC)
        alert.closed_by = current_user.get("username") or current_user.get("sub")
        await db.flush()

    # Write audit log entry
    audit = AuditService(db)
    await audit.log(
        action="alert_status_changed",
        resource_type="alert",
        resource_id=str(alert_id),
        user_id=current_user.get("sub"),
        username=current_user.get("username"),
        details={"new_status": new_status},
    )

    return AlertResponse.model_validate(alert)


@router.post("/{alert_id}/review", response_model=ReviewResponse)
async def submit_review(
    alert_id: UUID,
    review_data: ReviewCreate,
    request: Request,
    token_data: dict = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """Submit a review decision for an alert.

    This is the human-in-the-loop step that builds labeled training data.
    """
    reviewer_id = UUID(token_data["sub"])
    service = AlertService(db)
    review = await service.submit_review(alert_id, reviewer_id, review_data)

    # Audit log
    audit = AuditService(db)
    await audit.log(
        action="alert_reviewed",
        resource_type="alert",
        resource_id=str(alert_id),
        user_id=token_data.get("sub"),
        username=token_data.get("username"),
        details={"decision": review_data.decision, "sar_filed": review_data.sar_filed},
        ip_address=request.client.host if request.client else None,
    )

    return ReviewResponse.model_validate(review)
