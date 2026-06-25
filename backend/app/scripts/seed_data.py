"""Seed the database with sample data for development.

Usage:
    python -m app.scripts.seed_data
    # or via Docker:
    docker compose exec api python -m app.scripts.seed_data
"""

import asyncio
import random
import uuid
from datetime import UTC, datetime, timedelta

from passlib.context import CryptContext
from sqlalchemy import select

from app.core.database import async_session, engine
from app.models.base import Base
from app.models.transaction import Transaction, TransactionType
from app.models.user import User, UserRole

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


async def seed():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session() as db:
        # Create demo users
        existing = await db.execute(select(User).where(User.username == "admin"))
        if not existing.scalar_one_or_none():
            users = [
                User(
                    username="admin",
                    email="admin@fineract-aml.local",
                    full_name="AML Administrator",
                    hashed_password=pwd_context.hash("admin123"),
                    role=UserRole.ADMIN,
                ),
                User(
                    username="analyst1",
                    email="analyst1@fineract-aml.local",
                    full_name="Jane Smith",
                    hashed_password=pwd_context.hash("analyst123"),
                    role=UserRole.ANALYST,
                ),
                User(
                    username="senior_analyst",
                    email="senior@fineract-aml.local",
                    full_name="John Davis",
                    hashed_password=pwd_context.hash("senior123"),
                    role=UserRole.SENIOR_ANALYST,
                ),
            ]
            for u in users:
                db.add(u)
            print(f"Created {len(users)} users")

        # Generate sample transactions
        rng = random.Random(42)
        now = datetime.now(UTC)
        accounts = [f"ACC-{i:03d}" for i in range(1, 21)]
        clients = [f"CLI-{i:03d}" for i in range(1, 16)]
        currencies = ["USD", "EUR", "XAF"]

        sample_ips = [
            "192.168.1.100", "10.0.0.50", "172.16.0.10",
            "41.202.207.14", "41.210.45.67",  # Cameroon IPs
            "197.239.5.100", "154.72.166.30",  # Central Africa IPs
            "8.8.8.8", "185.220.101.1",  # Suspicious/Tor exit
            None,
        ]
        sample_countries = ["CM", "GA", "CF", "US", "FR", "NG", None]

        transactions = []
        for i in range(500):
            tx_type = rng.choice(list(TransactionType))
            amount = round(rng.lognormvariate(6, 2), 2)  # Log-normal: mostly small, some large
            amount = min(amount, 100000)

            # Inject some suspicious patterns
            if i % 50 == 0:
                amount = rng.uniform(9500, 9999)  # Structuring
            elif i % 70 == 0:
                amount = rng.choice([5000, 10000, 20000])  # Round numbers

            hours_ago = rng.uniform(0, 720)  # Last 30 days
            tx_date = now - timedelta(hours=hours_ago)

            # Some night transactions
            if i % 30 == 0:
                tx_date = tx_date.replace(hour=rng.randint(2, 5))

            tx = Transaction(
                fineract_transaction_id=f"FIN-TX-{uuid.uuid4().hex[:10].upper()}",
                fineract_account_id=rng.choice(accounts),
                fineract_client_id=rng.choice(clients),
                transaction_type=tx_type,
                amount=round(amount, 2),
                currency=rng.choice(currencies),
                transaction_date=tx_date,
                counterparty_account_id=(
                    rng.choice(accounts) if tx_type == TransactionType.TRANSFER else None
                ),
                description=rng.choice([
                    "Salary deposit",
                    "ATM withdrawal",
                    "Wire transfer",
                    "Mobile payment",
                    "Cash deposit",
                    "Loan repayment",
                    "Utility bill",
                    None,
                ]),
                ip_address=rng.choice(sample_ips),
                country_code=rng.choice(sample_countries),
            )
            transactions.append(tx)
            db.add(tx)

        # Add loan transactions for some clients
        loan_clients = clients[:8]  # First 8 clients get loan history
        loan_transactions = []
        for cli in loan_clients:
            num_loans = rng.randint(1, 3)
            for _ in range(num_loans):
                disbursement_amount = rng.choice([200000, 500000, 1000000, 2000000])
                days_ago = rng.randint(30, 180)
                tx = Transaction(
                    fineract_transaction_id=f"FIN-TX-{uuid.uuid4().hex[:10].upper()}",
                    fineract_account_id=rng.choice(accounts),
                    fineract_client_id=cli,
                    transaction_type=TransactionType.LOAN_DISBURSEMENT,
                    amount=disbursement_amount,
                    currency="XAF",
                    transaction_date=now - timedelta(days=days_ago),
                    description="Loan disbursement",
                    ip_address=rng.choice(sample_ips),
                    country_code="CM",
                )
                loan_transactions.append(tx)
                db.add(tx)

                # Corresponding repayments (varying repayment rates)
                repayment_rate = rng.uniform(0.3, 1.1)
                num_repayments = rng.randint(1, 6)
                repayment_per = (disbursement_amount * repayment_rate) / num_repayments
                for r in range(num_repayments):
                    repay_days_ago = days_ago - (r + 1) * rng.randint(7, 30)
                    if repay_days_ago < 0:
                        repay_days_ago = rng.randint(0, 5)
                    tx = Transaction(
                        fineract_transaction_id=f"FIN-TX-{uuid.uuid4().hex[:10].upper()}",
                        fineract_account_id=rng.choice(accounts),
                        fineract_client_id=cli,
                        transaction_type=TransactionType.LOAN_REPAYMENT,
                        amount=round(repayment_per, 2),
                        currency="XAF",
                        transaction_date=now - timedelta(days=max(repay_days_ago, 0)),
                        description="Loan repayment",
                        ip_address=rng.choice(sample_ips),
                        country_code="CM",
                    )
                    loan_transactions.append(tx)
                    db.add(tx)

        # Add suspicious transfer patterns
        transfer_transactions = []
        # Circular transfer: CLI-001 → CLI-002 → CLI-001
        for pair in [("ACC-001", "ACC-002", "CLI-001"), ("ACC-002", "ACC-001", "CLI-002")]:
            tx = Transaction(
                fineract_transaction_id=f"FIN-TX-{uuid.uuid4().hex[:10].upper()}",
                fineract_account_id=pair[0],
                fineract_client_id=pair[2],
                transaction_type=TransactionType.TRANSFER,
                amount=rng.uniform(500000, 900000),
                currency="XAF",
                transaction_date=now - timedelta(hours=rng.randint(1, 12)),
                counterparty_account_id=pair[1],
                description="Internal transfer",
                ip_address="41.202.207.14",
                country_code="CM",
            )
            transfer_transactions.append(tx)
            db.add(tx)

        # Rapid pair transfers: multiple ACC-003 ↔ ACC-004 in short window
        for i in range(4):
            tx = Transaction(
                fineract_transaction_id=f"FIN-TX-{uuid.uuid4().hex[:10].upper()}",
                fineract_account_id="ACC-003",
                fineract_client_id="CLI-003",
                transaction_type=TransactionType.TRANSFER,
                amount=rng.uniform(100000, 300000),
                currency="XAF",
                transaction_date=now - timedelta(hours=rng.randint(0, 6)),
                counterparty_account_id="ACC-004",
                description="Transfer",
                ip_address="41.210.45.67",
                country_code="CM",
            )
            transfer_transactions.append(tx)
            db.add(tx)

        await db.commit()
        print(f"Created {len(transactions)} sample transactions")
        print(f"Created {len(loan_transactions)} loan transactions (disbursements + repayments)")
        print(f"Created {len(transfer_transactions)} suspicious transfer patterns")

        # Now run analysis on some transactions to generate alerts
        print("\nTo generate alerts, start the Celery worker and run:")
        print("  docker compose exec api python -m app.scripts.analyze_all")
        print("\nTo compute credit scores:")
        print('  docker compose exec api python -c "from app.tasks.credit_scoring import compute_all_credit_scores; compute_all_credit_scores.delay()"')
        print("\nOr test the webhook with:")
        print('  curl -X POST http://localhost:8000/api/v1/webhook/fineract \\')
        print('    -H "Content-Type: application/json" \\')
        print('    -d \'{"transaction_id":"TEST-001","account_id":"ACC-001",')
        print('          "client_id":"CLI-001","transaction_type":"deposit",')
        print('          "amount":9800,"currency":"USD",')
        print('          "transaction_date":"2025-06-15T03:00:00Z"}\'')

    print("\nSeed complete!")
    print("Login credentials:")
    print("  admin / admin123")
    print("  analyst1 / analyst123")
    print("  senior_analyst / senior123")


if __name__ == "__main__":
    asyncio.run(seed())
