"""Case management endpoints for the compliance dashboard."""

from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import UserRole, require_role, verify_token
from app.models.case import CaseStatus
from app.schemas.case import (
    CaseAssign,
    CaseCreate,
    CaseListResponse,
    CaseResponse,
    CaseStatusUpdate,
)
from app.services.case_service import CaseService
from app.services.sar_service import SARService

router = APIRouter(
    prefix="/cases", tags=["Cases"], dependencies=[Depends(verify_token)]
)

_TERMINAL_CASE_STATUSES = {"closed_legitimate", "closed_fraud", "sar_filed"}


@router.post("", response_model=CaseResponse, status_code=status.HTTP_201_CREATED)
async def create_case(
    data: CaseCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new investigation case."""
    service = CaseService(db)
    case = await service.create_case(data)
    return CaseResponse.model_validate(case)


@router.get("", response_model=CaseListResponse)
async def list_cases(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    status_filter: CaseStatus | None = Query(None, alias="status"),
    db: AsyncSession = Depends(get_db),
):
    """List investigation cases."""
    service = CaseService(db)
    cases, total = await service.list_cases(page, page_size, status_filter)
    return CaseListResponse(
        items=[CaseResponse.model_validate(c) for c in cases],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{case_id}", response_model=CaseResponse)
async def get_case(case_id: UUID, db: AsyncSession = Depends(get_db)):
    service = CaseService(db)
    case = await service.get_case(case_id)
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Case not found"
        )
    return CaseResponse.model_validate(case)


@router.patch("/{case_id}/status", response_model=CaseResponse, dependencies=[Depends(require_role(UserRole.ANALYST, UserRole.MLRO, UserRole.ADMIN))])
async def update_case_status(
    case_id: UUID,
    data: CaseStatusUpdate,
    current_user: dict = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    service = CaseService(db)
    case = await service.update_status(case_id, data.status)

    # COBAC audit trail — record who closed the case and when
    new_status = data.status.value if hasattr(data.status, "value") else str(data.status)
    if new_status.lower() in _TERMINAL_CASE_STATUSES:
        case.closed_at = datetime.now(UTC)
        case.closed_by = current_user.get("username") or current_user.get("sub")
        await db.flush()

    return CaseResponse.model_validate(case)


@router.patch("/{case_id}/assign", response_model=CaseResponse)
async def assign_case(
    case_id: UUID,
    data: CaseAssign,
    db: AsyncSession = Depends(get_db),
):
    service = CaseService(db)
    case = await service.assign_case(case_id, data.assigned_to)
    return CaseResponse.model_validate(case)


@router.get("/{case_id}/sar/xml", response_class=Response, dependencies=[Depends(require_role(UserRole.MLRO, UserRole.ADMIN))])
async def export_sar_xml(case_id: UUID, db: AsyncSession = Depends(get_db)):
    """Export SAR document as COBAC-compatible XML."""
    sar_service = SARService(db)
    try:
        sar_data = await sar_service.generate_sar_document(case_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    xml_content = sar_service.export_to_xml(sar_data)
    filename = f"SAR_{sar_data.case_number}_{sar_data.filing_date}.xml"
    return Response(
        content=xml_content,
        media_type="application/xml",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{case_id}/sar/pdf", response_class=Response, dependencies=[Depends(require_role(UserRole.MLRO, UserRole.ADMIN))])
async def export_sar_pdf(case_id: UUID, db: AsyncSession = Depends(get_db)):
    """Export SAR document as a human-readable PDF for COBAC submission."""
    sar_service = SARService(db)
    try:
        sar_data = await sar_service.generate_sar_document(case_id)
        pdf_bytes = sar_service.export_to_pdf(sar_data)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    filename = f"SAR_{sar_data.case_number}_{sar_data.filing_date}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
