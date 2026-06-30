"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application
    app_name: str = "Fineract AML Service"
    app_version: str = "0.1.0"
    debug: bool = False
    api_prefix: str = "/api/v1"

    # Database
    database_url: str = "postgresql+asyncpg://aml:aml@localhost:5432/fineract_aml"
    database_echo: bool = False

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # CORS
    cors_origins: str = "http://localhost:3000"  # Comma-separated allowed origins

    # Fineract
    fineract_base_url: str = "https://localhost:8443/fineract-provider/api/v1"
    fineract_webhook_secret: str = "change-me-in-production"

    # ML
    model_path: str = "./models"
    anomaly_threshold: float = 0.7
    anomaly_contamination: float = 0.01  # Expected fraud rate (~0.1-0.5% in real data)
    risk_score_high: float = 0.8
    risk_score_medium: float = 0.5

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "fineract-aml"

    # Auth
    secret_key: str = "change-me-in-production"
    access_token_expire_minutes: int = 30
    # Internal server-to-server API key (BFF → AML for /kyc-sync and /aml-status).
    # Must be set in production — app refuses to start with the default value.
    internal_api_key: str = "change-me-in-production"

    # Celery
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # Currency
    default_currency: str = "XAF"

    # Credit scoring tiers
    credit_tier_a_min_score: float = 0.8
    credit_tier_b_min_score: float = 0.65
    credit_tier_c_min_score: float = 0.5
    credit_tier_d_min_score: float = 0.35
    # Below 0.35 = Tier E (Very Poor)

    credit_tier_a_max_amount: float = 2_000_000.0  # XAF
    credit_tier_b_max_amount: float = 1_000_000.0
    credit_tier_c_max_amount: float = 400_000.0
    credit_tier_d_max_amount: float = 100_000.0
    credit_tier_e_max_amount: float = 25_000.0

    # Credit scoring weights (must sum to 1.0)
    credit_weight_deposit_consistency: float = 0.20
    credit_weight_net_flow: float = 0.20
    credit_weight_savings_rate: float = 0.15
    credit_weight_tx_frequency: float = 0.10
    credit_weight_account_age: float = 0.10
    credit_weight_repayment_rate: float = 0.15
    credit_weight_fraud_history: float = 0.10

    # Credit batch scoring
    credit_scoring_batch_size: int = 500
    credit_min_transactions: int = 5

    # Rule engine
    max_transaction_amount: float = 500_000.0
    rapid_transaction_window_minutes: int = 60
    rapid_transaction_count: int = 20
    rapid_transaction_count_agent: int = 30
    structuring_threshold: float = 490_000.0
    new_account_age_days: int = 30
    agent_float_volume_minimum: float = 500_000.0
    scatter_gather_min_senders: int = 15

    # CTR (Currency Transaction Report) — auto-file threshold
    ctr_threshold: float = 5_000_000.0  # XAF — CEMAC regulatory threshold

    # Sanctions screening
    sanctions_screening_enabled: bool = True
    sanctions_match_threshold: float = 0.85

    # Agent network fraud rules
    agent_structuring_min_count: int = 5
    agent_collusion_window_minutes: int = 60
    agent_float_imbalance_threshold: float = 0.95
    agent_float_volume_minimum: float = 50_000.0
    agent_new_account_threshold: int = 8

    # Network typology rules (from IBM AMLSim)
    scatter_gather_min_senders: int = 8
    bipartite_fan_threshold: int = 5
    stacking_window_minutes: int = 30
    stacking_amount_ratio_min: float = 0.8
    stacking_amount_ratio_max: float = 1.2

    # Merchant fraud rules
    anonymous_payment_alert_threshold: float = 100_000.0

    # Post-disbursement loan monitoring
    loan_run_threshold: float = 0.8          # fraction of loan transferred = flag
    loan_immediate_cashout_minutes: int = 30  # cash-out within this window = critical
    loan_dispersal_counterparty_min: int = 5  # sent to this many parties = flag

    # Credit score gaming
    credit_gaming_inflow_multiplier: float = 3.0
    credit_gaming_score_penalty: float = 0.15

    # Adverse media screening
    adverse_media_enabled: bool = False
    adverse_media_api_key: str = ""
    adverse_media_api_url: str = "https://newsapi.org/v2/everything"
    adverse_media_min_risk_score: float = 0.6  # Only screen if risk score >= this

    # LLM / AI agent
    anthropic_api_key: str = ""
    llm_investigation_enabled: bool = False
    llm_model: str = "claude-opus-4-6"
    llm_investigation_min_risk_level: str = "high"  # "medium" | "high" | "critical"

    # Synchronous scoring endpoint
    sync_scoring_timeout_ms: int = 400
    sync_scoring_enabled: bool = True

    # Graph cache (Redis-backed)
    graph_cache_ttl_seconds: int = 1800
    graph_refresh_interval_minutes: int = 15
    graph_lookback_hours: int = 48

    # Graph fraud layer — persistent graph DB (Memgraph/Bolt) — ADR 0007
    graph_enabled: bool = False                       # master flag; off until verified per-env
    graph_database_url: str = "bolt://localhost:7687"
    graph_database_user: str = ""                     # empty → anonymous (Memgraph default)
    graph_database_password: str = ""                 # set both (+ bolt+s:// URL) in production
    graph_scoring_timeout_ms: int = 80                # bounded live-path budget (fail-open)
    graph_k_hops: int = 2                             # guilt-by-association hop limit (max 3)
    graph_score_weight: float = 0.15                  # weight of graph term in combined score
    graph_backfill_lookback_days: int = 30            # batch backfill window from Postgres

    # Shadow/canary ML deployment
    shadow_model_enabled: bool = False
    shadow_model_promotion_auc_delta: float = 0.02
    shadow_model_promotion_days: int = 7

    # Anomaly detector ensemble
    anomaly_ensemble_enabled: bool = False

    model_config = {"env_file": ".env", "env_prefix": "AML_"}


settings = Settings()
