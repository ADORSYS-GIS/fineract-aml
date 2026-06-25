"""IBM AML dataset import adapter — WeBank-enriched.

Ingests Kaggle HI-Small/HI-Large and IBM AMLSim CSV files into the
Fineract AML database as Transaction + Alert records enriched with
WeBank-specific context (XAF currency, actor_type, KYC level, agent/merchant IDs).

Usage
-----
# Stats only (no DB required)
python -m app.scripts.import_ibm_aml_data --file HI-Small_Trans.csv --stats-only

# Dry run (parse + enrich, no DB writes)
python -m app.scripts.import_ibm_aml_data --file HI-Small_Trans.csv --dry-run

# Full import
python -m app.scripts.import_ibm_aml_data --file HI-Small_Trans.csv --sample-legit 15000
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import random
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from sqlalchemy import insert
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.models.alert import Alert, AlertSource, AlertStatus
from app.models.transaction import Transaction, TransactionType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

XAF_PER_USD = 600.0  # CEMAC fixed peg: 1 USD ≈ 600 XAF

KAGGLE_COLUMNS = {
    "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
    "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency",
    "Payment Format", "Is Laundering",
}

AMLSIM_COLUMNS = {
    "tx_id", "sender_account", "receiver_account", "amount", "currency",
    "transaction_type", "timestamp", "is_sar",
}

PAYMENT_FORMAT_MAP: dict[str, TransactionType] = {
    "cheque": TransactionType.TRANSFER,
    "ach": TransactionType.TRANSFER,
    "wire": TransactionType.TRANSFER,
    "bitcoin": TransactionType.TRANSFER,
    "cash": TransactionType.TRANSFER,
    "reinvestment": TransactionType.TRANSFER,
    "credit card": TransactionType.CHARGE,
}

# Country distribution — CEMAC-weighted
_COUNTRY_BUCKETS = [
    (14, "CM"),   # Cameroon  70%
    (16, "NG"),   # Nigeria   10%
    (18, "CI"),   # Côte d'Ivoire 10%
    (20, "SN"),   # Senegal   10%
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ImportStats:
    total_rows_read: int = 0
    fraud_imported: int = 0
    legit_imported: int = 0
    skipped: int = 0
    duration_seconds: float = 0.0

    def summary(self) -> str:
        total = self.fraud_imported + self.legit_imported
        return (
            f"rows_read={self.total_rows_read}  "
            f"fraud={self.fraud_imported}  legit={self.legit_imported}  "
            f"total_imported={total}  skipped={self.skipped}  "
            f"elapsed={self.duration_seconds:.1f}s"
        )


@dataclass
class EnrichedTransaction:
    transaction: dict          # kwargs for Transaction(...)
    label: int                 # 0 = legit, 1 = fraud
    tx_id: str = field(default="")   # fineract_transaction_id for deduplication


# ---------------------------------------------------------------------------
# Importer
# ---------------------------------------------------------------------------

class IBMAMLImporter:
    """Stream-parse IBM AML CSV files and insert enriched records into the DB."""

    def __init__(self) -> None:
        # Precomputed after Pass 1 — reused by every _kyc_level call in Pass 2
        self._sorted_ts_list: list[datetime] = []
        self._kyc_cache: dict[str, int] = {}

    def detect_format(self, path: Path) -> Literal["kaggle", "amlsim"]:
        """Peek at the header row to determine the CSV schema."""
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            headers = set(reader.fieldnames or [])
        if KAGGLE_COLUMNS.issubset(headers):
            return "kaggle"
        if AMLSIM_COLUMNS.issubset(headers):
            return "amlsim"
        raise ValueError(
            f"Unrecognised CSV schema. Headers: {headers}\n"
            f"Expected Kaggle columns: {KAGGLE_COLUMNS}\n"
            f"  or AMLSim columns: {AMLSIM_COLUMNS}"
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def import_file(
        self,
        path: Path,
        fmt: Literal["kaggle", "amlsim", "auto"] = "auto",
        sample_legit: int | None = None,
        batch_size: int = 500,
        dry_run: bool = False,
        stats_only: bool = False,
    ) -> ImportStats:
        """Import a single CSV file; returns ImportStats."""
        if fmt == "auto":
            fmt = self.detect_format(path)
        logger.info("Detected format: %s", fmt)

        # Pass 1 — build account_first_seen + count fraud
        logger.info("Pass 1: scanning timestamps…")
        account_first_seen, fraud_count, total_count = self._pass1(path, fmt)
        logger.info(
            "Pass 1 complete: %d accounts, %d fraud rows out of %d total",
            len(account_first_seen), fraud_count, total_count,
        )

        # Precompute sorted timestamp list for O(log n) KYC lookups in Pass 2
        self._sorted_ts_list = sorted(account_first_seen.values())
        self._kyc_cache = {}

        if stats_only:
            return self._stats_only_report(account_first_seen, fraud_count, total_count)

        # Resolve sample_legit
        if sample_legit is None:
            sample_legit = fraud_count * 3

        # Pass 2 — full import
        logger.info(
            "Pass 2: importing (sample_legit=%d, batch_size=%d, dry_run=%s)…",
            sample_legit, batch_size, dry_run,
        )
        t0 = time.monotonic()
        stats = asyncio.run(
            self._pass2(path, fmt, account_first_seen, sample_legit, batch_size, dry_run)
        )
        stats.duration_seconds = time.monotonic() - t0
        return stats

    # ------------------------------------------------------------------
    # Pass 1 — lightweight scan
    # ------------------------------------------------------------------

    def _pass1(
        self, path: Path, fmt: Literal["kaggle", "amlsim"]
    ) -> tuple[dict[str, datetime], int, int]:
        """Return (account_first_seen, fraud_count, total_count)."""
        account_first_seen: dict[str, datetime] = {}
        fraud_count = 0
        total_count = 0

        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                total_count += 1
                try:
                    if fmt == "kaggle":
                        account_id = f"{row['From Bank']}_{row['Account']}"
                        ts = _parse_timestamp(row["Timestamp"])
                        label = int(row["Is Laundering"])
                    else:
                        account_id = row["sender_account"]
                        ts = _parse_timestamp(row["timestamp"])
                        label = int(row["is_sar"])
                except (KeyError, ValueError):
                    continue

                if ts is not None and account_id not in account_first_seen:
                    account_first_seen[account_id] = ts

                fraud_count += label

                if total_count % 10_000 == 0:
                    _progress(f"Pass 1: {total_count:,} rows scanned…")

        return account_first_seen, fraud_count, total_count

    # ------------------------------------------------------------------
    # Pass 2 — full enrichment + insert
    # ------------------------------------------------------------------

    async def _pass2(
        self,
        path: Path,
        fmt: Literal["kaggle", "amlsim"],
        account_first_seen: dict[str, datetime],
        sample_legit: int,
        batch_size: int,
        dry_run: bool,
    ) -> ImportStats:
        stats = ImportStats()

        # Reservoir for legitimate rows (Algorithm R)
        reservoir: list[EnrichedTransaction] = []
        reservoir_idx = 0  # counts how many legit rows have been seen

        fraud_batch: list[EnrichedTransaction] = []

        # Lazy DB import — not needed for dry-run
        db_ctx = None
        db = None
        if not dry_run:
            from app.core.database import async_session  # noqa: PLC0415
            db_ctx = async_session()
            db = await db_ctx.__aenter__()

        try:
            with path.open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for idx, row in enumerate(reader):
                    stats.total_rows_read += 1

                    if fmt == "kaggle":
                        enriched = self._parse_kaggle_row(row, idx, account_first_seen)
                    else:
                        enriched = self._parse_amlsim_row(row, idx, account_first_seen)

                    if enriched is None:
                        stats.skipped += 1
                        continue

                    if enriched.label == 1:
                        fraud_batch.append(enriched)
                        if len(fraud_batch) >= batch_size:
                            if db is not None:
                                inserted, dupes = await self._batch_insert(db, fraud_batch)
                                stats.fraud_imported += inserted
                                stats.skipped += dupes
                            else:
                                stats.fraud_imported += len(fraud_batch)
                            fraud_batch.clear()
                    else:
                        # Reservoir sampling (Algorithm R)
                        if reservoir_idx < sample_legit:
                            reservoir.append(enriched)
                        else:
                            j = random.randint(0, reservoir_idx)
                            if j < sample_legit:
                                reservoir[j] = enriched
                        reservoir_idx += 1

                    if stats.total_rows_read % 10_000 == 0:
                        _progress(
                            f"Pass 2: {stats.total_rows_read:,} rows read, "
                            f"fraud={stats.fraud_imported}, "
                            f"reservoir={min(reservoir_idx, sample_legit)}/{sample_legit}…"
                        )

            # Flush remaining fraud batch
            if fraud_batch:
                if db is not None:
                    inserted, dupes = await self._batch_insert(db, fraud_batch)
                    stats.fraud_imported += inserted
                    stats.skipped += dupes
                else:
                    stats.fraud_imported += len(fraud_batch)

            # Insert reservoir (legitimate sample) in batches
            for i in range(0, len(reservoir), batch_size):
                chunk = reservoir[i : i + batch_size]
                if db is not None:
                    inserted, dupes = await self._batch_insert(db, chunk)
                    stats.legit_imported += inserted
                    stats.skipped += dupes
                else:
                    stats.legit_imported += len(chunk)

            if db is not None:
                await db.commit()

        finally:
            if db_ctx is not None:
                await db_ctx.__aexit__(None, None, None)

        return stats

    # ------------------------------------------------------------------
    # Row parsers
    # ------------------------------------------------------------------

    def _parse_kaggle_row(
        self,
        row: dict,
        idx: int,
        account_first_seen: dict[str, datetime],
    ) -> EnrichedTransaction | None:
        try:
            ts = _parse_timestamp(row["Timestamp"])
            if ts is None:
                return None
            amount_usd = float(row["Amount Paid"])
            account_id = f"{row['From Bank']}_{row['Account']}"
            counterparty_id = f"{row['To Bank']}_{row['Account.1']}"
            label = int(row["Is Laundering"])
            tx_type = self._map_payment_format(row["Payment Format"])
        except (KeyError, ValueError):
            return None

        enrichment = self._enrich(account_id, amount_usd, account_first_seen)
        tx_id = f"IBM-{idx}"

        transaction = {
            "id": uuid.uuid4(),
            "fineract_transaction_id": tx_id,
            "fineract_account_id": account_id,
            "fineract_client_id": account_id,
            "transaction_type": tx_type,
            "transaction_date": ts,
            "counterparty_account_id": counterparty_id,
            **enrichment,
        }
        return EnrichedTransaction(transaction=transaction, label=label, tx_id=tx_id)

    def _parse_amlsim_row(
        self,
        row: dict,
        idx: int,
        account_first_seen: dict[str, datetime],
    ) -> EnrichedTransaction | None:
        try:
            ts = _parse_timestamp(row["timestamp"])
            if ts is None:
                return None
            amount_usd = float(row["amount"])
            account_id = row["sender_account"]
            counterparty_id = row["receiver_account"]
            label = int(row["is_sar"])
            tx_type = _amlsim_tx_type(row.get("transaction_type", "transfer"))
        except (KeyError, ValueError):
            return None

        enrichment = self._enrich(account_id, amount_usd, account_first_seen)
        tx_id = f"IBM-SIM-{row.get('tx_id', idx)}"

        transaction = {
            "id": uuid.uuid4(),
            "fineract_transaction_id": tx_id,
            "fineract_account_id": account_id,
            "fineract_client_id": account_id,
            "transaction_type": tx_type,
            "transaction_date": ts,
            "counterparty_account_id": counterparty_id,
            **enrichment,
        }
        return EnrichedTransaction(transaction=transaction, label=label, tx_id=tx_id)

    # ------------------------------------------------------------------
    # Enrichment helpers
    # ------------------------------------------------------------------

    def _enrich(
        self,
        account_id: str,
        amount_usd: float,
        account_first_seen: dict[str, datetime],
    ) -> dict:
        """Return WeBank-specific enrichment fields for a transaction dict."""
        actor = self._actor_type(account_id)
        return {
            "amount": round(amount_usd * XAF_PER_USD, 2),
            "currency": "XAF",
            "actor_type": actor,
            "agent_id": f"AGT-{_stable_hash(account_id) % 50:04d}" if actor == "agent" else None,
            "branch_id": f"BRN-{_stable_hash(account_id) % 10:04d}" if actor == "agent" else None,
            "merchant_id": f"MERCH-{_stable_hash(account_id) % 20:04d}" if actor == "merchant" else None,
            "kyc_level": self._kyc_level(account_id, account_first_seen),
            "country_code": self._country_code(account_id),
        }

    def _map_payment_format(self, fmt: str) -> TransactionType:
        return PAYMENT_FORMAT_MAP.get(fmt.lower().strip(), TransactionType.TRANSFER)

    def _actor_type(self, account_id: str) -> str:
        bucket = _stable_hash(account_id) % 100
        if bucket < 80:
            return "customer"
        if bucket < 95:
            return "agent"
        return "merchant"

    def _kyc_level(self, account_id: str, account_first_seen: dict[str, datetime]) -> int:
        cached = self._kyc_cache.get(account_id)
        if cached is not None:
            return cached

        ts = account_first_seen.get(account_id)
        if ts is None or not self._sorted_ts_list:
            return 1

        n = len(self._sorted_ts_list)
        pos = _bisect_left(self._sorted_ts_list, ts)
        pct = pos / n  # 0.0 = earliest = oldest account (most established)

        if pct <= 0.25:
            level = 4   # established
        elif pct <= 0.50:
            level = 3
        elif pct <= 0.75:
            level = 2
        else:
            level = 1   # new account

        self._kyc_cache[account_id] = level
        return level

    def _country_code(self, account_id: str) -> str:
        bucket = _stable_hash(account_id + "country") % 20
        for threshold, code in _COUNTRY_BUCKETS:
            if bucket < threshold:
                return code
        return "CM"

    # ------------------------------------------------------------------
    # DB insertion
    # ------------------------------------------------------------------

    async def _batch_insert(
        self, db, batch: list[EnrichedTransaction]
    ) -> tuple[int, int]:
        """Insert transactions + alerts; returns (inserted, duplicates)."""
        if not batch:
            return 0, 0

        # Build transaction rows
        tx_rows = [e.transaction for e in batch]

        # ON CONFLICT DO NOTHING for idempotency
        stmt = (
            pg_insert(Transaction)
            .values(tx_rows)
            .on_conflict_do_nothing(index_elements=["fineract_transaction_id"])
            .returning(Transaction.id, Transaction.fineract_transaction_id)
        )
        result = await db.execute(stmt)
        inserted_rows = result.fetchall()

        # Map fineract_transaction_id → new UUID for alert FK
        tx_id_map = {row[1]: row[0] for row in inserted_rows}
        inserted_count = len(tx_id_map)
        duplicate_count = len(batch) - inserted_count

        # Build alert rows only for successfully inserted transactions
        alert_rows = []
        for e in batch:
            tx_uuid = tx_id_map.get(e.tx_id)
            if tx_uuid is None:
                continue  # was a duplicate — skip
            status = AlertStatus.CONFIRMED_FRAUD if e.label == 1 else AlertStatus.FALSE_POSITIVE
            risk_score = 1.0 if e.label == 1 else 0.0
            title = (
                "IBM AML: Confirmed Laundering" if e.label == 1
                else "IBM AML: Verified Legitimate"
            )
            alert_rows.append({
                "id": uuid.uuid4(),
                "transaction_id": tx_uuid,
                "status": status,
                "source": AlertSource.ML_MODEL,
                "risk_score": risk_score,
                "title": title,
                "description": "Imported from IBM AML dataset for classifier training.",
            })

        if alert_rows:
            await db.execute(insert(Alert).values(alert_rows))
        await db.flush()

        return inserted_count, duplicate_count

    # ------------------------------------------------------------------
    # Stats-only report (no DB)
    # ------------------------------------------------------------------

    def _stats_only_report(
        self,
        account_first_seen: dict[str, datetime],
        fraud_count: int,
        total_count: int,
    ) -> ImportStats:
        legit_count = total_count - fraud_count
        actor_counts = {"customer": 0, "agent": 0, "merchant": 0}
        kyc_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        country_counts: dict[str, int] = {}

        for account_id in account_first_seen:
            a = self._actor_type(account_id)
            actor_counts[a] = actor_counts.get(a, 0) + 1
            k = self._kyc_level(account_id, account_first_seen)
            kyc_counts[k] = kyc_counts.get(k, 0) + 1
            c = self._country_code(account_id)
            country_counts[c] = country_counts.get(c, 0) + 1

        n_accounts = len(account_first_seen) or 1
        print(f"\n{'='*60}")
        print(f"  Total rows  : {total_count:>10,}")
        print(f"  Fraud rows  : {fraud_count:>10,}  ({fraud_count/total_count*100:.1f}%)")
        print(f"  Legit rows  : {legit_count:>10,}  ({legit_count/total_count*100:.1f}%)")
        print(f"  Unique accts: {n_accounts:>10,}")
        print("\n  Actor distribution (sender accounts):")
        for a, cnt in sorted(actor_counts.items()):
            print(f"    {a:<12}: {cnt:>7,}  ({cnt/n_accounts*100:.1f}%)")
        print("\n  KYC level distribution:")
        for k in sorted(kyc_counts):
            cnt = kyc_counts[k]
            print(f"    KYC {k}: {cnt:>7,}  ({cnt/n_accounts*100:.1f}%)")
        print("\n  Country distribution:")
        for c, cnt in sorted(country_counts.items(), key=lambda x: -x[1]):
            print(f"    {c}: {cnt:>7,}  ({cnt/n_accounts*100:.1f}%)")
        print(f"{'='*60}\n")

        stats = ImportStats()
        stats.total_rows_read = total_count
        return stats


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _stable_hash(s: str) -> int:
    """Return a non-negative, process-stable hash (Python's built-in hash
    is randomised across processes with PYTHONHASHSEED).  We use a simple
    djb2 variant so the enrichment is truly deterministic across runs."""
    h = 5381
    for ch in s:
        h = ((h << 5) + h) ^ ord(ch)
    return h & 0x7FFF_FFFF  # keep positive, 31-bit


def _parse_timestamp(raw: str) -> datetime | None:
    raw = raw.strip()
    for fmt in ("%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def _amlsim_tx_type(raw: str) -> TransactionType:
    mapping = {
        "transfer": TransactionType.TRANSFER,
        "deposit": TransactionType.DEPOSIT,
        "withdrawal": TransactionType.WITHDRAWAL,
        "cash_in": TransactionType.DEPOSIT,
        "cash_out": TransactionType.WITHDRAWAL,
        "loan": TransactionType.LOAN_DISBURSEMENT,
        "repayment": TransactionType.LOAN_REPAYMENT,
    }
    return mapping.get(raw.lower().strip(), TransactionType.TRANSFER)


def _bisect_left(sorted_list: list, value) -> int:
    """Binary search — index of first element >= value."""
    lo, hi = 0, len(sorted_list)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_list[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _progress(msg: str) -> None:
    try:
        from tqdm import tqdm  # type: ignore
        tqdm.write(msg)
    except ImportError:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Import IBM AML datasets into Fineract AML (WeBank-enriched).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--file", required=True, type=Path, help="Path to the IBM AML CSV file")
    p.add_argument(
        "--format",
        choices=["kaggle", "amlsim", "auto"],
        default="auto",
        dest="fmt",
        help="CSV schema: 'kaggle' (HI-Small/Large), 'amlsim', or 'auto' (default)",
    )
    p.add_argument(
        "--sample-legit",
        type=int,
        default=None,
        metavar="N",
        help="Number of legitimate transactions to import as FALSE_POSITIVE alerts "
             "(default: 3× fraud count)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=500,
        metavar="N",
        help="DB insert batch size (default: 500)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse + enrich but do not write to the database",
    )
    p.add_argument(
        "--stats-only",
        action="store_true",
        help="Print dataset statistics and exit without importing",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    path = args.file.expanduser().resolve()
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    importer = IBMAMLImporter()
    stats = importer.import_file(
        path=path,
        fmt=args.fmt,
        sample_legit=args.sample_legit,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        stats_only=args.stats_only,
    )

    if not args.stats_only:
        mode = "[DRY RUN] " if args.dry_run else ""
        print(f"\n{mode}Import complete: {stats.summary()}")


if __name__ == "__main__":
    main()
