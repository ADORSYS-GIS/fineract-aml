"""FastAPI application entry point."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api import (
    alerts,
    auth,
    cases,
    credit,
    ctrs,
    graph,
    model_health,
    scoring,
    transactions,
    webhook,
)
from app.core.config import settings

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

# Validate critical secrets are not defaults in production
if not settings.debug:
    _insecure_defaults = {"change-me-in-production"}
    if settings.fineract_webhook_secret in _insecure_defaults:
        raise RuntimeError(
            "AML_FINERACT_WEBHOOK_SECRET is still set to the default value. "
            "Set a strong secret before running in production (AML_DEBUG=false)."
        )
    if settings.secret_key in _insecure_defaults:
        raise RuntimeError(
            "AML_SECRET_KEY is still set to the default value. "
            "Set a strong JWT signing key before running in production."
        )

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Anti-Money Laundering detection service for Apache Fineract",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — restrict origins from config (comma-separated)
allowed_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics instrumentation
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
except ImportError:
    logger.info("prometheus-fastapi-instrumentator not installed; /metrics not exposed")

# Register routers
app.include_router(webhook.router, prefix=settings.api_prefix)
app.include_router(auth.router, prefix=settings.api_prefix)
app.include_router(transactions.router, prefix=settings.api_prefix)
app.include_router(alerts.router, prefix=settings.api_prefix)
app.include_router(cases.router, prefix=settings.api_prefix)
app.include_router(credit.router, prefix=settings.api_prefix)
app.include_router(scoring.router, prefix=settings.api_prefix)
app.include_router(graph.router, prefix=settings.api_prefix)
app.include_router(model_health.router, prefix=settings.api_prefix)
app.include_router(ctrs.router, prefix=settings.api_prefix)


@app.get("/health")
async def health_check():
    """Enhanced health check — verifies database and Redis connectivity."""
    health = {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
        "checks": {},
    }

    # Check database
    try:
        from sqlalchemy import text

        from app.core.database import async_session

        async with async_session() as db:
            await db.execute(text("SELECT 1"))
        health["checks"]["database"] = "ok"
    except Exception as e:
        health["checks"]["database"] = f"error: {e}"
        health["status"] = "degraded"

    # Check Redis
    try:
        import redis

        r = redis.from_url(settings.redis_url, socket_connect_timeout=2)
        r.ping()
        health["checks"]["redis"] = "ok"
    except Exception as e:
        health["checks"]["redis"] = f"error: {e}"
        health["status"] = "degraded"

    return health


@app.get("/")
async def root():
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
    }
