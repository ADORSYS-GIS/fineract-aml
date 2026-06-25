"""Watchlist synchronization — downloads and refreshes sanctions/PEP lists.

Sources:
- OFAC SDN (Specially Designated Nationals): US Treasury
  URL: https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/SDN.XML
  Format: XML (Advanced Sanctions Data Model)
  Update frequency: Irregular, but should be checked daily

- EU Consolidated Sanctions:
  URL: https://webgate.ec.europa.eu/fsd/fsf/public/files/xmlFullSanctionsList_1_1/content
  Format: XML
  Update frequency: Updated as needed

- UN Security Council Consolidated List:
  URL: https://scsanctions.un.org/resources/xml/en/consolidated.xml
  Format: XML
  Update frequency: Updated after each Security Council decision

Best practice: Refresh all lists daily (OFAC can update multiple times per day).
Our schedule: Every 6 hours via Celery Beat.
"""

import asyncio
import logging
import xml.etree.ElementTree as ET

import httpx

from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

# OFAC SDN download URL (publicly available, no auth required)
OFAC_SDN_XML_URL = (
    "https://sanctionslistservice.ofac.treas.gov"
    "/api/PublicationPreview/exports/SDN.XML"
)

# EU Consolidated Sanctions
EU_SANCTIONS_XML_URL = (
    "https://webgate.ec.europa.eu/fsd/fsf"
    "/public/files/xmlFullSanctionsList_1_1/content"
)

# UN Security Council
UN_SANCTIONS_XML_URL = (
    "https://scsanctions.un.org/resources/xml/en/consolidated.xml"
)


def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _parse_ofac_sdn_xml(xml_content: bytes) -> list[dict]:
    """Parse OFAC SDN XML into watchlist entry dicts.

    The SDN XML uses the namespace:
    https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/ADVANCED/SDN

    Each <sdnEntry> contains:
    - <uid>: Unique ID
    - <sdnType>: Individual, Entity, Vessel, Aircraft
    - <lastName> / <firstName>: Primary name
    - <programList><program>: Sanctions programs (e.g. SDGT, IRAN, UKRAINE-EO13661)
    - <akaList><aka>: Aliases
    - <idList><id>: Identification documents
    - <addressList><address>: Addresses with country
    - <nationalityList><nationality>: Nationalities

    Example real entries from the OFAC SDN list:
    - ISLAMIC REVOLUTIONARY GUARD CORPS (IRGC), Entity, Program: IRAN
    - HIZBALLAH, Entity, Program: SDGT (Specially Designated Global Terrorist)
    - BANCO DELTA ASIA, Entity, Program: NPWMD (North Korea)
    - Individual entries with passport numbers, DOB, nationalities
    """
    entries = []

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        logger.error("Failed to parse OFAC SDN XML: %s", e)
        return entries

    # Handle XML namespaces — OFAC uses a default namespace
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    for sdn_entry in root.iter(f"{ns}sdnEntry"):
        uid = _get_text(sdn_entry, f"{ns}uid")
        sdn_type = _get_text(sdn_entry, f"{ns}sdnType", "Individual").lower()
        last_name = _get_text(sdn_entry, f"{ns}lastName", "")
        first_name = _get_text(sdn_entry, f"{ns}firstName", "")

        name = f"{first_name} {last_name}".strip() if first_name else last_name
        if not name:
            continue

        # Programs
        programs = []
        program_list = sdn_entry.find(f"{ns}programList")
        if program_list is not None:
            for prog in program_list.iter(f"{ns}program"):
                if prog.text:
                    programs.append(prog.text.strip())

        # Aliases
        aliases = []
        aka_list = sdn_entry.find(f"{ns}akaList")
        if aka_list is not None:
            for aka in aka_list.iter(f"{ns}aka"):
                alias_last = _get_text(aka, f"{ns}lastName", "")
                alias_first = _get_text(aka, f"{ns}firstName", "")
                alias_name = f"{alias_first} {alias_last}".strip() if alias_first else alias_last
                if alias_name:
                    aliases.append(alias_name)

        # Country from address
        country = None
        addr_list = sdn_entry.find(f"{ns}addressList")
        if addr_list is not None:
            for addr in addr_list.iter(f"{ns}address"):
                country_text = _get_text(addr, f"{ns}country")
                if country_text:
                    country = country_text[:2].upper()  # Approximate ISO code
                    break

        # Identification documents
        identifiers = {}
        id_list = sdn_entry.find(f"{ns}idList")
        if id_list is not None:
            for id_elem in id_list.iter(f"{ns}id"):
                id_type = _get_text(id_elem, f"{ns}idType", "unknown")
                id_number = _get_text(id_elem, f"{ns}idNumber", "")
                if id_number:
                    identifiers[id_type] = id_number

        entries.append({
            "name": name,
            "entity_type": sdn_type,
            "country": country,
            "aliases": aliases,
            "identifiers": identifiers,
            "program": "; ".join(programs) if programs else None,
            "uid": uid,
        })

    return entries


def _get_text(element, tag, default=""):
    """Safely get text content from an XML element."""
    child = element.find(tag)
    if child is not None and child.text:
        return child.text.strip()
    return default


async def _sync_ofac_sdn():
    """Download and sync the OFAC SDN list."""
    from sqlalchemy import delete

    from app.core.database import async_session
    from app.models.sanctions import WatchlistEntry, WatchlistSource
    from app.services.audit_service import AuditService

    logger.info("Starting OFAC SDN list sync...")

    # Download the XML
    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            response = await client.get(OFAC_SDN_XML_URL)
            if response.status_code != 200:
                logger.error("OFAC SDN download failed: HTTP %d", response.status_code)
                return
            xml_content = response.content
    except httpx.RequestError as e:
        logger.error("OFAC SDN download failed: %s", e)
        return

    # Parse entries
    entries = _parse_ofac_sdn_xml(xml_content)
    if not entries:
        logger.warning("OFAC SDN parse returned 0 entries — skipping update")
        return

    logger.info("Parsed %d OFAC SDN entries", len(entries))

    async with async_session() as db:
        audit = AuditService(db)

        # Clear existing OFAC entries and reload (atomic refresh)
        await db.execute(
            delete(WatchlistEntry).where(
                WatchlistEntry.source == WatchlistSource.OFAC_SDN
            )
        )

        import json

        for entry in entries:
            db.add(WatchlistEntry(
                source=WatchlistSource.OFAC_SDN,
                entity_name=entry["name"],
                entity_type=entry["entity_type"],
                country=entry.get("country"),
                aliases=json.dumps(entry.get("aliases", [])),
                identifiers=json.dumps(entry.get("identifiers", {})),
                program=entry.get("program"),
                is_active=True,
            ))

        await audit.log(
            action="watchlist_sync",
            resource_type="ofac_sdn",
            details={
                "entries_loaded": len(entries),
                "source_url": OFAC_SDN_XML_URL,
            },
        )

        await db.commit()
        logger.info("OFAC SDN sync complete: %d entries loaded", len(entries))


@celery_app.task(name="app.tasks.watchlist_sync.sync_ofac_sdn")
def sync_ofac_sdn():
    """Download and refresh the OFAC SDN watchlist. Runs every 6 hours."""
    _run_async(_sync_ofac_sdn())


@celery_app.task(name="app.tasks.watchlist_sync.sync_all_watchlists")
def sync_all_watchlists():
    """Sync all watchlist sources. Runs every 6 hours."""
    _run_async(_sync_ofac_sdn())
    # Future: add EU and UN list sync here
    logger.info("All watchlist syncs complete")
