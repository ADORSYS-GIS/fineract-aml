"""Tests for the webhook API endpoint."""


from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

SAMPLE_WEBHOOK_PAYLOAD = {
    "transaction_id": "TX-12345",
    "account_id": "ACC-001",
    "client_id": "CLI-001",
    "transaction_type": "deposit",
    "amount": 5000.00,
    "currency": "USD",
    "transaction_date": "2025-06-15T14:30:00Z",
    "description": "Salary deposit",
}


class TestWebhookEndpoint:
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("healthy", "degraded")
        assert "checks" in data

    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "docs" in response.json()

    def test_webhook_payload_validation(self):
        """Test that invalid payloads are rejected."""
        bad_payload = {"transaction_id": "TX-1"}  # Missing required fields
        response = client.post("/api/v1/webhook/fineract", json=bad_payload)
        assert response.status_code == 422  # Validation error
