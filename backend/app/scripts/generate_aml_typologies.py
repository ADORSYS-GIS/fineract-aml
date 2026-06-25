"""Synthetic AML typology data generator.

Generates labeled transaction datasets covering WeBank-specific and
IBM AMLSim typologies for ML model training and backtesting.

Usage:
    python -m app.scripts.generate_aml_typologies \\
        --n-typologies 100 \\
        --fraud-rate 0.05 \\
        --output aml_synthetic.csv

Typologies covered:
    - scatter_gather    : N collectors → 1 consolidator → 1 disbursement
    - bipartite_layering: fan-out (1→N) followed by fan-in (N→1)
    - stacking          : rapid sequential A→B→C→D hops
    - agent_structuring : agent handling sub-threshold deposits in bulk
    - loan_and_run      : loan disbursement → immediate transfer cascade
    - random_normal     : legitimate transactions (fraud=0)
"""

import argparse
import csv
import random
import uuid
from datetime import UTC, datetime, timedelta


def _now() -> datetime:
    return datetime.now(UTC)


def _rnd_account() -> str:
    return f"ACC{uuid.uuid4().hex[:8].upper()}"


def _rnd_agent() -> str:
    return f"AGT{random.randint(100, 999)}"


def _rnd_amount(lo: float, hi: float) -> float:
    return round(random.uniform(lo, hi), 2)


# ── Base record template ──────────────────────────────────────────────────────

def _tx(
    *,
    src: str,
    dst: str,
    amount: float,
    tx_type: str,
    offset_minutes: float = 0,
    actor_type: str = "customer",
    agent_id: str | None = None,
    kyc_level: int = 2,
    label: int = 0,
    typology: str = "normal",
) -> dict:
    ts = _now() - timedelta(days=random.randint(1, 30)) + timedelta(minutes=offset_minutes)
    return {
        "transaction_id": uuid.uuid4().hex,
        "account_id": src,
        "counterparty_account_id": dst,
        "amount": amount,
        "currency": "XAF",
        "transaction_type": tx_type,
        "timestamp": ts.isoformat(),
        "actor_type": actor_type,
        "agent_id": agent_id or "",
        "kyc_level": kyc_level,
        "fraud_label": label,
        "typology": typology,
    }


# ── Typology generators ────────────────────────────────────────────────────────

class ScatterGatherGenerator:
    """N collectors each send small amounts to a consolidator → one disbursement."""

    def generate(self, n_collectors: int = 10) -> list[dict]:
        consolidator = _rnd_account()
        final_dst = _rnd_account()
        base_amount = _rnd_amount(50_000, 200_000)
        records = []

        # Fan-in: N collectors → consolidator
        for i in range(n_collectors):
            collector = _rnd_account()
            records.append(_tx(
                src=collector,
                dst=consolidator,
                amount=_rnd_amount(base_amount * 0.08, base_amount * 0.15),
                tx_type="transfer",
                offset_minutes=-120 + i * 5,
                label=1,
                typology="scatter_gather",
            ))

        # Fan-out: consolidator → final destination (single large transfer)
        records.append(_tx(
            src=consolidator,
            dst=final_dst,
            amount=sum(r["amount"] for r in records) * random.uniform(0.85, 0.95),
            tx_type="transfer",
            offset_minutes=0,
            label=1,
            typology="scatter_gather",
        ))
        return records


class BipartiteLayeringGenerator:
    """Fan-out (1→N) followed by fan-in (N→1) — classic layering sandwich."""

    def generate(self, n_middle: int = 6) -> list[dict]:
        source = _rnd_account()
        sink = _rnd_account()
        middle = [_rnd_account() for _ in range(n_middle)]
        base = _rnd_amount(500_000, 2_000_000)
        records = []

        # Fan-out: source → each middle node
        for i, mid in enumerate(middle):
            records.append(_tx(
                src=source,
                dst=mid,
                amount=base / n_middle * random.uniform(0.9, 1.1),
                tx_type="transfer",
                offset_minutes=-180 + i * 2,
                label=1,
                typology="bipartite_layering",
            ))

        # Fan-in: each middle node → sink
        for i, mid in enumerate(middle):
            records.append(_tx(
                src=mid,
                dst=sink,
                amount=base / n_middle * random.uniform(0.88, 1.0),
                tx_type="transfer",
                offset_minutes=-60 + i * 3,
                label=1,
                typology="bipartite_layering",
            ))
        return records


class StackingGenerator:
    """Sequential A→B→C→D rapid hops where each amount is 80–120% of prior."""

    def generate(self, n_hops: int = 4) -> list[dict]:
        accounts = [_rnd_account() for _ in range(n_hops + 1)]
        amount = _rnd_amount(300_000, 1_500_000)
        records = []

        for i in range(n_hops):
            amount = amount * random.uniform(0.8, 1.2)
            records.append(_tx(
                src=accounts[i],
                dst=accounts[i + 1],
                amount=round(amount, 2),
                tx_type="transfer",
                offset_minutes=i * random.uniform(3, 10),
                label=1,
                typology="stacking",
            ))
        return records


class AgentStructuringGenerator:
    """WeBank-specific: agent handles many sub-threshold deposits in 1 hour."""

    def generate(self, n_deposits: int = 7, threshold: float = 5_000_000) -> list[dict]:
        agent_id = _rnd_agent()
        agent_account = _rnd_account()
        records = []

        # Structuring threshold: stay just below 5M XAF
        for i in range(n_deposits):
            records.append(_tx(
                src=_rnd_account(),
                dst=agent_account,
                amount=_rnd_amount(threshold * 0.7, threshold * 0.95),
                tx_type="deposit",
                offset_minutes=i * random.uniform(4, 9),
                actor_type="agent",
                agent_id=agent_id,
                kyc_level=random.choice([1, 2]),
                label=1,
                typology="agent_structuring",
            ))
        return records


class LoanAndRunGenerator:
    """Loan disbursement → immediate large transfer out."""

    def generate(self) -> list[dict]:
        borrower = _rnd_account()
        loan_amount = _rnd_amount(500_000, 5_000_000)
        records = []

        # Disbursement
        records.append(_tx(
            src="BANK_ACCOUNT",
            dst=borrower,
            amount=loan_amount,
            tx_type="loan_disbursement",
            offset_minutes=-5,
            label=1,
            typology="loan_and_run",
        ))

        # Rapid extraction (>80% within 30 min)
        records.append(_tx(
            src=borrower,
            dst=_rnd_account(),
            amount=loan_amount * random.uniform(0.82, 0.95),
            tx_type="transfer",
            offset_minutes=0,
            label=1,
            typology="loan_and_run",
        ))
        return records


class NormalTransactionGenerator:
    """Generates legitimate transaction patterns (label=0)."""

    def generate(self, n: int = 20) -> list[dict]:
        account = _rnd_account()
        records = []
        for _ in range(n):
            tx_type = random.choice(["deposit", "withdrawal", "transfer"])
            amount = _rnd_amount(5_000, 500_000)
            records.append(_tx(
                src=account if tx_type != "deposit" else _rnd_account(),
                dst=_rnd_account() if tx_type == "transfer" else account,
                amount=amount,
                tx_type=tx_type,
                offset_minutes=random.uniform(-43200, 0),
                label=0,
                typology="normal",
            ))
        return records


# ── CLI ────────────────────────────────────────────────────────────────────────

def generate(
    n_typologies: int = 100,
    fraud_rate: float = 0.05,
    output: str = "aml_synthetic.csv",
) -> int:
    """Generate synthetic AML dataset.

    Args:
        n_typologies: Number of fraud typology instances to generate.
        fraud_rate: Target ratio of fraudulent vs total transactions.
        output: Output CSV file path.

    Returns:
        Total number of records written.
    """
    generators = [
        ScatterGatherGenerator(),
        BipartiteLayeringGenerator(),
        StackingGenerator(),
        AgentStructuringGenerator(),
        LoanAndRunGenerator(),
    ]

    all_records: list[dict] = []

    # Generate fraud typologies
    for i in range(n_typologies):
        gen = generators[i % len(generators)]
        all_records.extend(gen.generate())

    fraud_count = len(all_records)
    # Fill with normal transactions to achieve target fraud rate
    target_total = int(fraud_count / fraud_rate)
    n_normal = target_total - fraud_count
    normal_gen = NormalTransactionGenerator()
    while len(all_records) < target_total:
        batch = min(20, target_total - len(all_records))
        all_records.extend(normal_gen.generate(n=batch))

    random.shuffle(all_records)

    fieldnames = [
        "transaction_id", "account_id", "counterparty_account_id",
        "amount", "currency", "transaction_type", "timestamp",
        "actor_type", "agent_id", "kyc_level", "fraud_label", "typology",
    ]

    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    actual_fraud_rate = sum(1 for r in all_records if r["fraud_label"] == 1) / len(all_records)
    print(
        f"Generated {len(all_records):,} records "
        f"({fraud_count:,} fraud, {len(all_records) - fraud_count:,} normal) "
        f"— fraud rate: {actual_fraud_rate:.1%} → {output}"
    )
    return len(all_records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic AML typology data")
    parser.add_argument("--n-typologies", type=int, default=100)
    parser.add_argument("--fraud-rate", type=float, default=0.05)
    parser.add_argument("--output", type=str, default="aml_synthetic.csv")
    args = parser.parse_args()
    generate(
        n_typologies=args.n_typologies,
        fraud_rate=args.fraud_rate,
        output=args.output,
    )
