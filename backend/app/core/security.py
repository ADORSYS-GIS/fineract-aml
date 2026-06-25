"""Authentication and authorization utilities."""

import hashlib
import hmac
from datetime import UTC, datetime, timedelta
from enum import Enum

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from app.core.config import settings

security = HTTPBearer()

ALGORITHM = "HS256"


class UserRole(str, Enum):
    ADMIN = "admin"
    MLRO = "mlro"           # Money Laundering Reporting Officer — can access SARs
    ANALYST = "analyst"     # Can review alerts and cases
    OPERATOR = "operator"   # Can view transactions only
    VIEWER = "viewer"       # Read-only


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(UTC) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.secret_key, algorithm=ALGORITHM)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Verify and decode a JWT token.

    Returns a dict with at minimum ``sub``, ``username``, and ``role`` keys
    so that callers can use them for audit trails and RBAC checks.
    """
    try:
        payload = jwt.decode(
            credentials.credentials, settings.secret_key, algorithms=[ALGORITHM]
        )
        if payload.get("sub") is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing subject",
            )
        # Normalise: always expose role and username at the top level
        if "role" not in payload:
            payload["role"] = UserRole.ANALYST.value
        if "username" not in payload:
            payload["username"] = payload.get("sub", "")
        # user_id is a convenience alias for sub
        if "user_id" not in payload:
            payload["user_id"] = payload.get("sub")
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )


def require_role(*allowed_roles: UserRole):
    """FastAPI dependency: require the token bearer to have one of the specified roles."""

    async def role_checker(token_data: dict = Depends(verify_token)):
        user_role = token_data.get("role") or token_data.get("roles", [UserRole.VIEWER])
        if isinstance(user_role, list):
            user_role = user_role[0] if user_role else UserRole.VIEWER
        if user_role not in [r.value for r in allowed_roles]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {[r.value for r in allowed_roles]}",
            )
        return token_data

    return role_checker


def verify_webhook_signature(payload: bytes, signature: str) -> bool:
    """Verify Fineract webhook HMAC signature."""
    expected = hmac.new(
        settings.fineract_webhook_secret.encode(),
        payload,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)
