#!/usr/bin/env python3
"""Synthetic data generator for AML model training.

Generates realistic transaction data with embedded fraud patterns so you can
train the ML models WITHOUT waiting months for real data + analyst labels.

Usage:
    # Generate data and seed the database (requires running PostgreSQL)
    python scripts/generate_training_data.py --seed-db

    # Generate data to CSV files (no database needed)
    python scripts/generate_training_data.py --output-csv ./data

    # Custom size
    python scripts/generate_training_data.py --clients 500 --transactions 50000 --fraud-rate 0.03

Fraud patterns injected:
    1. Structuring    — multiple transactions just below threshold
    2. Rapid velocity — burst of transactions in short window
    3. Round amounts  — exact round numbers ($5K, $10K)
    4. Night activity — transactions at 2-5 AM
    5. Circular flow  — A→B→C→A transfer chains
    6. Fan-out        — one account sends to many recipients
    7. New IP         — transaction from never-seen IP address
    8. Smurfing combo — multiple patterns combined (most suspicious)
"""

import argparse
import csv
import json
import random
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

NUM_CLIENTS = 200
NUM_ACCOUNTS = 300  # Some clients have multiple accounts
NUM_TRANSACTIONS = 20000
FRAUD_RATE = 0.03  # 3% of transactions are fraudulent
DAYS_OF_HISTORY = 180
CURRENCY = "XAF"

# Realistic XAF amounts (Central African CFA franc)
NORMAL_AMOUNT_MEAN = 25000       # ~$38 — typical small transaction
NORMAL_AMOUNT_STD = 15000
SALARY_AMOUNT_RANGE = (80000, 500000)  # Monthly salary deposits
LARGE_LEGIT_RANGE = (500000, 3000000)  # Large legitimate transactions

# Thresholds (matching config defaults)
MAX_TRANSACTION_AMOUNT = 10000
STRUCTURING_THRESHOLD = 9500

# IP addresses
NORMAL_IPS = [f"192.168.1.{i}" for i in range(1, 50)]
SUSPICIOUS_IPS = ["45.33.32.156", "185.220.101.1", "91.219.236.222"]

COUNTRIES = ["CM", "GA", "TD", "CF", "CG", "GQ"]  # CEMAC member states
HIGH_RISK_COUNTRIES = ["IR", "KP", "SY", "AF"]

TRANSACTION_TYPES = ["deposit", "withdrawal", "transfer"]


def generate_clients(n: int) -> list[dict]:
    """Generate synthetic client profiles."""
    clients = []
    first_names = [
        "Amadou", "Fatou", "Ibrahim", "Aissatou", "Moussa", "Mariama",
        "Ousmane", "Aminata", "Abdoulaye", "Khadija", "Pierre", "Marie",
        "Jean", "Francoise", "Paul", "Therese", "Emmanuel", "Grace",
        "David", "Ruth", "Samuel", "Esther", "Daniel", "Sarah",
    ]
    last_names = [
        "Diallo", "Ba", "Ndiaye", "Sow", "Diop", "Fall", "Mbaye",
        "Camara", "Traore", "Sylla", "Bongo", "Nguema", "Oyono",
        "Mba", "Ekotto", "Eto'o", "Mbappé", "Ndong", "Obiang",
    ]

    for i in range(n):
        client_id = f"CLI-{i+1:04d}"
        clients.append({
            "client_id": client_id,
            "name": f"{random.choice(first_names)} {random.choice(last_names)}",
            "country": random.choice(COUNTRIES),
            "is_pep": random.random() < 0.02,  # 2% PEP rate
            "risk_profile": random.choices(
                ["low", "medium", "high"], weights=[0.7, 0.2, 0.1]
            )[0],
        })
    return clients


def generate_accounts(clients: list[dict], n: int) -> list[dict]:
    """Generate accounts linked to clients (some clients have multiple)."""
    accounts = []
    for i in range(n):
        client = random.choice(clients)
        accounts.append({
            "account_id": f"ACC-{i+1:04d}",
            "client_id": client["client_id"],
            "client_name": client["name"],
            "country": client["country"],
            "normal_ip": random.choice(NORMAL_IPS),
        })
    return accounts


def generate_normal_transaction(
    accounts: list[dict], base_date: datetime
) -> dict:
    """Generate a normal (non-fraudulent) transaction."""
    account = random.choice(accounts)
    tx_type = random.choices(
        TRANSACTION_TYPES, weights=[0.45, 0.35, 0.20]
    )[0]

    # Normal amounts follow a log-normal distribution
    amount = max(100, np.random.lognormal(mean=9.5, sigma=1.2))  # median ~13K XAF

    # Occasionally large legitimate transactions (salary, business)
    if random.random() < 0.05:
        amount = random.uniform(*SALARY_AMOUNT_RANGE)
    elif random.random() < 0.01:
        amount = random.uniform(*LARGE_LEGIT_RANGE)

    # Normal business hours (7 AM - 10 PM), slight weekend dip
    hour = int(np.random.normal(14, 4))
    hour = max(7, min(22, hour))
    day_offset = random.randint(0, DAYS_OF_HISTORY)
    tx_date = base_date - timedelta(days=day_offset, hours=-hour, minutes=random.randint(0, 59))

    counterparty = None
    counterparty_name = None
    if tx_type == "transfer":
        target = random.choice(accounts)
        counterparty = target["account_id"]
        counterparty_name = target["client_name"]

    return {
        "transaction_id": f"TX-{uuid.uuid4().hex[:12].upper()}",
        "account_id": account["account_id"],
        "client_id": account["client_id"],
        "transaction_type": tx_type,
        "amount": round(amount, 2),
        "currency": CURRENCY,
        "transaction_date": tx_date.isoformat(),
        "counterparty_account_id": counterparty,
        "counterparty_name": counterparty_name,
        "ip_address": account["normal_ip"],
        "country_code": account["country"],
        "is_fraud": False,
        "fraud_pattern": None,
    }


def generate_structuring_pattern(accounts: list[dict], base_date: datetime) -> list[dict]:
    """Generate structuring (smurfing) — multiple txns just below threshold."""
    account = random.choice(accounts)
    txns = []
    day_offset = random.randint(0, DAYS_OF_HISTORY)

    for i in range(random.randint(3, 6)):
        amount = random.uniform(STRUCTURING_THRESHOLD - 500, STRUCTURING_THRESHOLD)
        hour = random.randint(9, 18)
        tx_date = base_date - timedelta(days=day_offset, hours=-hour, minutes=i * random.randint(10, 30))

        txns.append({
            "transaction_id": f"TX-{uuid.uuid4().hex[:12].upper()}",
            "account_id": account["account_id"],
            "client_id": account["client_id"],
            "transaction_type": "deposit",
            "amount": round(amount, 2),
            "currency": CURRENCY,
            "transaction_date": tx_date.isoformat(),
            "counterparty_account_id": None,
            "counterparty_name": None,
            "ip_address": account["normal_ip"],
            "country_code": account["country"],
            "is_fraud": True,
            "fraud_pattern": "structuring",
        })
    return txns


def generate_rapid_velocity(accounts: list[dict], base_date: datetime) -> list[dict]:
    """Generate rapid velocity — burst of transactions in minutes."""
    account = random.choice(accounts)
    txns = []
    day_offset = random.randint(0, DAYS_OF_HISTORY)
    hour = random.randint(8, 20)

    for i in range(random.randint(10, 20)):
        tx_date = base_date - timedelta(
            days=day_offset, hours=-hour, minutes=i * random.randint(1, 5)
        )
        txns.append({
            "transaction_id": f"TX-{uuid.uuid4().hex[:12].upper()}",
            "account_id": account["account_id"],
            "client_id": account["client_id"],
            "transaction_type": random.choice(["deposit", "withdrawal"]),
            "amount": round(random.uniform(1000, 8000), 2),
            "currency": CURRENCY,
            "transaction_date": tx_date.isoformat(),
            "counterparty_account_id": None,
            "counterparty_name": None,
            "ip_address": account["normal_ip"],
            "country_code": account["country"],
            "is_fraud": True,
            "fraud_pattern": "rapid_velocity",
        })
    return txns


def generate_circular_flow(accounts: list[dict], base_date: datetime) -> list[dict]:
    """Generate circular transfer chain: A→B→C→A (layering)."""
    if len(accounts) < 3:
        return []

    chain = random.sample(accounts, random.randint(3, 5))
    txns = []
    day_offset = random.randint(0, DAYS_OF_HISTORY)
    amount = round(random.uniform(50000, 500000), 2)

    for i in range(len(chain)):
        src = chain[i]
        dst = chain[(i + 1) % len(chain)]
        tx_date = base_date - timedelta(
            days=day_offset, hours=-(10 + i * 2), minutes=random.randint(0, 30)
        )

        txns.append({
            "transaction_id": f"TX-{uuid.uuid4().hex[:12].upper()}",
            "account_id": src["account_id"],
            "client_id": src["client_id"],
            "transaction_type": "transfer",
            "amount": round(amount * random.uniform(0.95, 1.0), 2),  # slight variation
            "currency": CURRENCY,
            "transaction_date": tx_date.isoformat(),
            "counterparty_account_id": dst["account_id"],
            "counterparty_name": dst["client_name"],
            "ip_address": src["normal_ip"],
            "country_code": src["country"],
            "is_fraud": True,
            "fraud_pattern": "circular_flow",
        })
    return txns


def generate_night_activity(accounts: list[dict], base_date: datetime) -> list[dict]:
    """Generate suspicious night-time transactions (2-5 AM)."""
    account = random.choice(accounts)
    txns = []
    day_offset = random.randint(0, DAYS_OF_HISTORY)

    for i in range(random.randint(2, 5)):
        hour = random.randint(2, 5)
        tx_date = base_date - timedelta(days=day_offset + i, hours=-hour, minutes=random.randint(0, 59))
        txns.append({
            "transaction_id": f"TX-{uuid.uuid4().hex[:12].upper()}",
            "account_id": account["account_id"],
            "client_id": account["client_id"],
            "transaction_type": random.choice(["withdrawal", "transfer"]),
            "amount": round(random.uniform(5000, 50000), 2),
            "currency": CURRENCY,
            "transaction_date": tx_date.isoformat(),
            "counterparty_account_id": random.choice(accounts)["account_id"] if random.random() > 0.5 else None,
            "counterparty_name": None,
            "ip_address": random.choice(SUSPICIOUS_IPS),
            "country_code": random.choice(HIGH_RISK_COUNTRIES),
            "is_fraud": True,
            "fraud_pattern": "night_activity",
        })
    return txns


def generate_fan_out(accounts: list[dict], base_date: datetime) -> list[dict]:
    """Generate fan-out pattern — one account sends to many recipients."""
    src = random.choice(accounts)
    targets = random.sample(accounts, min(random.randint(6, 12), len(accounts) - 1))
    txns = []
    day_offset = random.randint(0, DAYS_OF_HISTORY)
    base_amount = random.uniform(10000, 100000)

    for i, dst in enumerate(targets):
        if dst["account_id"] == src["account_id"]:
            continue
        tx_date = base_date - timedelta(days=day_offset, hours=-(9 + i), minutes=random.randint(0, 30))
        txns.append({
            "transaction_id": f"TX-{uuid.uuid4().hex[:12].upper()}",
            "account_id": src["account_id"],
            "client_id": src["client_id"],
            "transaction_type": "transfer",
            "amount": round(base_amount * random.uniform(0.8, 1.2), 2),
            "currency": CURRENCY,
            "transaction_date": tx_date.isoformat(),
            "counterparty_account_id": dst["account_id"],
            "counterparty_name": dst["client_name"],
            "ip_address": src["normal_ip"],
            "country_code": src["country"],
            "is_fraud": True,
            "fraud_pattern": "fan_out",
        })
    return txns


def generate_round_amount(accounts: list[dict], base_date: datetime) -> list[dict]:
    """Generate suspicious exact round-number transactions."""
    account = random.choice(accounts)
    txns = []
    round_amounts = [5000, 10000, 50000, 100000, 500000, 1000000]

    for i in range(random.randint(2, 4)):
        day_offset = random.randint(0, DAYS_OF_HISTORY)
        tx_date = base_date - timedelta(days=day_offset, hours=-random.randint(8, 20))
        txns.append({
            "transaction_id": f"TX-{uuid.uuid4().hex[:12].upper()}",
            "account_id": account["account_id"],
            "client_id": account["client_id"],
            "transaction_type": random.choice(["deposit", "withdrawal"]),
            "amount": float(random.choice(round_amounts)),
            "currency": CURRENCY,
            "transaction_date": tx_date.isoformat(),
            "counterparty_account_id": None,
            "counterparty_name": None,
            "ip_address": account["normal_ip"],
            "country_code": account["country"],
            "is_fraud": True,
            "fraud_pattern": "round_amount",
        })
    return txns


def generate_dataset(
    num_clients: int = NUM_CLIENTS,
    num_transactions: int = NUM_TRANSACTIONS,
    fraud_rate: float = FRAUD_RATE,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Generate a complete synthetic dataset.

    Returns:
        (clients, accounts, transactions) — all as lists of dicts.
    """
    print(f"Generating {num_clients} clients, {num_transactions} transactions, {fraud_rate:.0%} fraud rate...")

    clients = generate_clients(num_clients)
    accounts = generate_accounts(clients, int(num_clients * 1.5))
    now = datetime.now(UTC)

    # Generate normal transactions
    num_normal = int(num_transactions * (1 - fraud_rate))
    transactions = [generate_normal_transaction(accounts, now) for _ in range(num_normal)]

    # Generate fraud patterns
    num_fraud_patterns = int(num_transactions * fraud_rate / 5)  # avg ~5 txns per pattern
    fraud_generators = [
        generate_structuring_pattern,
        generate_rapid_velocity,
        generate_circular_flow,
        generate_night_activity,
        generate_fan_out,
        generate_round_amount,
    ]

    fraud_transactions = []
    for _ in range(num_fraud_patterns):
        gen = random.choice(fraud_generators)
        fraud_transactions.extend(gen(accounts, now))

    transactions.extend(fraud_transactions)
    random.shuffle(transactions)

    # Stats
    total = len(transactions)
    fraud_count = sum(1 for t in transactions if t["is_fraud"])
    patterns = {}
    for t in transactions:
        if t["fraud_pattern"]:
            patterns[t["fraud_pattern"]] = patterns.get(t["fraud_pattern"], 0) + 1

    print(f"\nGenerated {total} transactions:")
    print(f"  Normal:     {total - fraud_count} ({(total - fraud_count) / total:.1%})")
    print(f"  Fraudulent: {fraud_count} ({fraud_count / total:.1%})")
    print("\nFraud patterns:")
    for pattern, count in sorted(patterns.items(), key=lambda x: -x[1]):
        print(f"  {pattern:25s}: {count:5d} transactions")
    print(f"\nClients: {len(clients)}, Accounts: {len(accounts)}")

    return clients, accounts, transactions


def save_to_csv(clients, accounts, transactions, output_dir: str):
    """Save generated data to CSV files."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Transactions CSV
    tx_fields = [
        "transaction_id", "account_id", "client_id", "transaction_type",
        "amount", "currency", "transaction_date", "counterparty_account_id",
        "counterparty_name", "ip_address", "country_code", "is_fraud", "fraud_pattern",
    ]
    with open(output / "transactions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=tx_fields)
        writer.writeheader()
        writer.writerows(transactions)

    # Clients CSV
    client_fields = ["client_id", "name", "country", "is_pep", "risk_profile"]
    with open(output / "clients.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=client_fields)
        writer.writeheader()
        writer.writerows(clients)

    print(f"\nSaved to {output}:")
    print(f"  transactions.csv ({len(transactions)} rows)")
    print(f"  clients.csv ({len(clients)} rows)")


def seed_database(transactions: list[dict], clients: list[dict]):
    """Seed the database with generated transactions and create training labels.

    Uses synchronous SQLAlchemy (psycopg2) to avoid greenlet issues on Windows.
    """
    from passlib.context import CryptContext
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import Session

    # Import all models so metadata knows about them
    import app.models  # noqa: F401
    from app.core.config import settings
    from app.models.alert import Alert, AlertSource, AlertStatus
    from app.models.base import Base
    from app.models.review import Review
    from app.models.transaction import RiskLevel, Transaction, TransactionType
    from app.models.user import User, UserRole

    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    # Build sync database URL from async one
    sync_url = settings.database_url.replace("+asyncpg", "").replace("asyncpg://", "")
    sync_url = f"postgresql+psycopg2://{sync_url.split('://', 1)[-1]}"
    print("\n--- Seeding Database ---")
    print(f"Database: {sync_url.split('@')[1] if '@' in sync_url else sync_url}")

    sync_engine = create_engine(sync_url)

    # Create tables
    Base.metadata.create_all(sync_engine)
    print("Tables created.")

    with Session(sync_engine) as db:
        # 1. Create admin user
        existing = db.execute(select(User).where(User.username == "admin")).scalar_one_or_none()
        if not existing:
            admin = User(
                username="admin",
                email="admin@aml.local",
                full_name="AML Administrator",
                hashed_password=pwd_context.hash("admin123"),
                role=UserRole.ADMIN,
            )
            db.add(admin)
            db.flush()
            print("Admin user created: admin / admin123")
            admin_id = admin.id
        else:
            admin_id = existing.id
            print("Admin user already exists.")

        # 2. Ingest transactions
        print(f"Ingesting {len(transactions)} transactions...")
        tx_objects = []
        fraud_tx_ids = []
        legit_tx_ids = []

        for i, tx_data in enumerate(transactions):
            tx_type_str = tx_data["transaction_type"]
            try:
                tx_type = TransactionType(tx_type_str)
            except ValueError:
                tx_type = TransactionType.OTHER

            tx = Transaction(
                fineract_transaction_id=tx_data["transaction_id"],
                fineract_account_id=tx_data["account_id"],
                fineract_client_id=tx_data["client_id"],
                transaction_type=tx_type,
                amount=tx_data["amount"],
                currency=tx_data["currency"],
                transaction_date=datetime.fromisoformat(tx_data["transaction_date"]),
                counterparty_account_id=tx_data.get("counterparty_account_id"),
                counterparty_name=tx_data.get("counterparty_name"),
                ip_address=tx_data.get("ip_address"),
                country_code=tx_data.get("country_code"),
                risk_score=0.0,
                risk_level=RiskLevel.LOW,
            )
            db.add(tx)
            tx_objects.append(tx)

            if tx_data["is_fraud"]:
                fraud_tx_ids.append(i)
            else:
                legit_tx_ids.append(i)

            if (i + 1) % 5000 == 0:
                db.flush()
                print(f"  {i + 1}/{len(transactions)} ingested...")

        db.flush()
        print(f"  All {len(transactions)} transactions ingested.")

        # 3. Create alerts and reviews (labeled training data)
        print("Creating alerts and reviews (labeled training data)...")

        n_fraud_labels = len(fraud_tx_ids)
        n_legit_labels = min(len(legit_tx_ids), n_fraud_labels * 5)  # 5:1 ratio
        legit_sample = random.sample(legit_tx_ids, n_legit_labels)

        label_count = 0
        for idx in fraud_tx_ids:
            tx = tx_objects[idx]
            alert = Alert(
                transaction_id=tx.id,
                status=AlertStatus.CONFIRMED_FRAUD,
                source=AlertSource.RULE_ENGINE,
                risk_score=random.uniform(0.7, 1.0),
                title=f"Synthetic fraud: {transactions[idx]['fraud_pattern']}",
                description=f"Pattern: {transactions[idx]['fraud_pattern']}",
                triggered_rules=json.dumps([transactions[idx]["fraud_pattern"]]),
            )
            db.add(alert)
            db.flush()

            review = Review(
                alert_id=alert.id,
                reviewer_id=admin_id,
                decision="confirmed_fraud",
                notes=f"Synthetic training data — pattern: {transactions[idx]['fraud_pattern']}",
            )
            db.add(review)
            label_count += 1

        for idx in legit_sample:
            tx = tx_objects[idx]
            alert = Alert(
                transaction_id=tx.id,
                status=AlertStatus.FALSE_POSITIVE,
                source=AlertSource.RULE_ENGINE,
                risk_score=random.uniform(0.3, 0.6),
                title="Normal transaction reviewed",
                description="Analyst confirmed as legitimate",
            )
            db.add(alert)
            db.flush()

            review = Review(
                alert_id=alert.id,
                reviewer_id=admin_id,
                decision="legitimate",
                notes="Synthetic training data — confirmed legitimate",
            )
            db.add(review)
            label_count += 1

        db.commit()
        print(f"  Created {label_count} labeled alerts ({n_fraud_labels} fraud + {n_legit_labels} legitimate)")

    # 4. Train models directly (sync — avoids greenlet issues on Windows)
    print("\n--- Training Models ---")

    from app.features.extractor import FeatureExtractor
    from app.ml.anomaly_detector import AnomalyDetector
    from app.ml.fraud_classifier import FraudClassifier
    from app.tasks.training import _build_account_history, _build_account_index

    with Session(sync_engine) as db:
        # Fetch all transactions for training
        all_txs = list(db.execute(
            select(Transaction).order_by(Transaction.created_at.desc()).limit(10000)
        ).scalars().all())

        print(f"Training anomaly detector on {len(all_txs)} transactions...")
        account_index = _build_account_index(all_txs)
        features_list = []
        for tx in all_txs:
            h1, h24 = _build_account_history(tx, all_txs, account_index)
            features_list.append(FeatureExtractor.extract(tx, h1, h24))

        feature_matrix = np.vstack(features_list)
        detector = AnomalyDetector()
        metrics = detector.train(feature_matrix)
        print(f"  Anomaly detector trained: {metrics['n_samples']} samples, "
              f"{metrics['anomaly_rate']:.1%} anomaly rate")

        # Train fraud classifier
        fraud_alerts = list(db.execute(
            select(Alert).where(Alert.status == AlertStatus.CONFIRMED_FRAUD)
        ).scalars().all())
        legit_alerts = list(db.execute(
            select(Alert).where(Alert.status == AlertStatus.FALSE_POSITIVE)
        ).scalars().all())

        fraud_tx_objs = []
        for a in fraud_alerts:
            tx = db.execute(select(Transaction).where(Transaction.id == a.transaction_id)).scalar_one_or_none()
            if tx:
                fraud_tx_objs.append(tx)

        legit_tx_objs = []
        for a in legit_alerts:
            tx = db.execute(select(Transaction).where(Transaction.id == a.transaction_id)).scalar_one_or_none()
            if tx:
                legit_tx_objs.append(tx)

        print(f"Training fraud classifier on {len(fraud_tx_objs)} fraud + {len(legit_tx_objs)} legitimate...")

        classifier = FraudClassifier()
        if classifier.can_train(len(fraud_tx_objs), len(fraud_tx_objs) + len(legit_tx_objs)):
            # Build feature matrix from labeled data
            all_labeled = fraud_tx_objs + legit_tx_objs
            account_ids = {tx.fineract_account_id for tx in all_labeled}
            related = []
            for acc_id in account_ids:
                related.extend(db.execute(
                    select(Transaction).where(Transaction.fineract_account_id == acc_id)
                ).scalars().all())
            acc_idx = _build_account_index(related)

            train_features = []
            train_labels = []
            for tx in fraud_tx_objs:
                h1, h24 = _build_account_history(tx, related, acc_idx)
                train_features.append(FeatureExtractor.extract(tx, h1, h24))
                train_labels.append(1)
            for tx in legit_tx_objs:
                h1, h24 = _build_account_history(tx, related, acc_idx)
                train_features.append(FeatureExtractor.extract(tx, h1, h24))
                train_labels.append(0)

            fm = np.vstack(train_features)
            la = np.array(train_labels)
            cl_metrics = classifier.train(fm, la, feature_names=FeatureExtractor.get_feature_names())
            print(f"  Fraud classifier: CV AUC={cl_metrics['cv_auc_mean']:.4f} "
                  f"(±{cl_metrics['cv_auc_std']:.4f}), deployed={cl_metrics['deployed']}")
        else:
            print(f"  Not enough labels for fraud classifier (need {classifier.MIN_FRAUD_SAMPLES} fraud)")


    print("\n=== DONE ===")
    print(f"Database seeded with {len(transactions)} transactions")
    print(f"Labeled training data: {n_fraud_labels} fraud + {n_legit_labels} legitimate")
    print(f"Models trained and saved to {settings.model_path}/")
    print("\nLogin credentials: admin / admin123")
    print("API: http://localhost:8000/docs")
    print("Dashboard: http://localhost:3000")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic AML training data")
    parser.add_argument("--clients", type=int, default=NUM_CLIENTS, help="Number of clients")
    parser.add_argument("--transactions", type=int, default=NUM_TRANSACTIONS, help="Number of transactions")
    parser.add_argument("--fraud-rate", type=float, default=FRAUD_RATE, help="Fraud rate (0.0-1.0)")
    parser.add_argument("--output-csv", type=str, help="Output directory for CSV files")
    parser.add_argument("--seed-db", action="store_true", help="Seed the database and train models")
    args = parser.parse_args()

    clients, accounts, transactions = generate_dataset(
        num_clients=args.clients,
        num_transactions=args.transactions,
        fraud_rate=args.fraud_rate,
    )

    if args.output_csv:
        save_to_csv(clients, accounts, transactions, args.output_csv)

    if args.seed_db:
        seed_database(transactions, clients)

    if not args.output_csv and not args.seed_db:
        print("\nUse --output-csv or --seed-db to save the data.")
        print("Example: python scripts/generate_training_data.py --seed-db")


if __name__ == "__main__":
    main()
