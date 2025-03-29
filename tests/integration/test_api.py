import pytest
from fastapi.testclient import TestClient
from main import app
from database import Session
from datetime import datetime, timedelta
import jwt

client = TestClient(app)

@pytest.fixture(scope="module")
def auth_token():
    payload = {
        "sub": "test_user",
        "exp": datetime.utcnow() + timedelta(hours=1)
    }
    return jwt.encode(payload, "secret_key", algorithm="HS256")

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_unauthorized_access():
    response = client.get("/api/sensors")
    assert response.status_code == 401
    assert "WWW-Authenticate" in response.headers

def test_sensor_data_submission(auth_token):
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "kwh": 150.25,
        "source": "test"
    }
    
    response = client.post(
        "/api/sensors/data",
        json=payload,
        headers={"Authorization": f"Bearer {auth_token}"}
    )
    
    assert response.status_code == 201
    assert "record_id" in response.json()

def test_historical_data_query(auth_token):
    params = {
        "start": (datetime.utcnow() - timedelta(days=1)).isoformat(),
        "end": datetime.utcnow().isoformat(),
        "resolution": "hourly"
    }
    
    response = client.get(
        "/api/analytics/historical",
        params=params,
        headers={"Authorization": f"Bearer {auth_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "timestamps" in data
    assert "values" in data
    assert len(data["timestamps"]) == len(data["values"])

def test_prediction_endpoint(auth_token):
    test_data = {
        "features": {
            "energy_usage": [100, 120, 110],
            "temperature": [22.5, 23.0, 22.8]
        }
    }
    
    response = client.post(
        "/api/predict",
        json=test_data,
        headers={"Authorization": f"Bearer {auth_token}"}
    )
    
    assert response.status_code == 200
    prediction = response.json()
    assert "timestamp" in prediction
    assert "predicted_co2" in prediction
    assert isinstance(prediction["predicted_co2"], float)

def test_rate_limiting(auth_token):
    for _ in range(5):
        response = client.get(
            "/api/sensors",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
    
    assert response.status_code == 429
    assert "Retry-After" in response.headers

def test_database_rollback_on_failure(auth_token):
    # Test transaction integrity with invalid data
    payload = {
        "timestamp": "invalid-date",
        "kwh": "not-a-number"
    }
    
    response = client.post(
        "/api/sensors/data",
        json=payload,
        headers={"Authorization": f"Bearer {auth_token}"}
    )
    
    assert response.status_code == 422
    
    # Verify no data persisted
    with Session() as session:
        count = session.query(EnergyUsage).count()
        assert count == 0
