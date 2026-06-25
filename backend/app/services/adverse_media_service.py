"""Adverse media screening service — checks for negative news about entities.

Inspired by Marble's compliance platform approach to negative news screening.
FATF Recommendation 12 increasingly requires adverse media checks for PEPs
and high-risk customers beyond standard sanctions list screening.
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# Keywords indicating negative media relevance to AML/financial crime
_NEGATIVE_KEYWORDS = [
    "money laundering",
    "fraud",
    "corruption",
    "bribery",
    "cartel",
    "drug trafficking",
    "terrorism",
    "sanctions",
    "financial crime",
    "embezzlement",
    "tax evasion",
    "kickback",
    "illicit",
    "criminal",
    "indicted",
    "convicted",
    "arrested",
    "investigated",
]


@dataclass
class AdverseMediaResult:
    entity_name: str
    hit_count: int
    highest_relevance_score: float   # 0.0–1.0 (fraction of keywords matched)
    article_snippets: list[str] = field(default_factory=list)
    screened_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    error: str | None = None


class AdverseMediaService:
    """Screens an entity name against a news API for adverse media coverage."""

    async def screen_entity(
        self, name: str, entity_type: str = "individual"
    ) -> AdverseMediaResult:
        """Search for negative news coverage of an entity.

        Args:
            name: Full name of the individual or entity to screen.
            entity_type: "individual" or "entity".

        Returns:
            AdverseMediaResult with hit count and article snippets.
        """
        if not settings.adverse_media_enabled or not settings.adverse_media_api_key:
            return AdverseMediaResult(
                entity_name=name,
                hit_count=0,
                highest_relevance_score=0.0,
                error="Adverse media screening disabled or API key not configured",
            )

        try:
            articles = await self._fetch_articles(name)
        except Exception as exc:
            logger.warning("Adverse media API call failed for '%s': %s", name, exc)
            return AdverseMediaResult(
                entity_name=name,
                hit_count=0,
                highest_relevance_score=0.0,
                error=str(exc),
            )

        hits: list[tuple[float, str]] = []
        for article in articles:
            text = " ".join(filter(None, [
                article.get("title", ""),
                article.get("description", ""),
            ])).lower()
            matched_keywords = [kw for kw in _NEGATIVE_KEYWORDS if kw in text]
            if matched_keywords:
                relevance = len(matched_keywords) / len(_NEGATIVE_KEYWORDS)
                snippet = article.get("title", "")[:200]
                hits.append((relevance, snippet))

        if not hits:
            return AdverseMediaResult(
                entity_name=name,
                hit_count=0,
                highest_relevance_score=0.0,
            )

        hits.sort(key=lambda x: x[0], reverse=True)
        top_snippets = [snippet for _, snippet in hits[:3]]
        max_relevance = hits[0][0]

        logger.info(
            "Adverse media: %d hits for '%s' (max relevance=%.2f)",
            len(hits), name, max_relevance,
        )

        return AdverseMediaResult(
            entity_name=name,
            hit_count=len(hits),
            highest_relevance_score=max_relevance,
            article_snippets=top_snippets,
        )

    async def _fetch_articles(self, query: str) -> list[dict]:
        """Fetch news articles from the configured API."""
        params = {
            "q": f'"{query}"',  # Exact name match
            "language": "en",
            "pageSize": 20,
            "sortBy": "relevancy",
            "apiKey": settings.adverse_media_api_key,
        }
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(settings.adverse_media_api_url, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("articles", [])
