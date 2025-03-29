import pytest
from fastapi import status
from fastapi.testclient import TestClient
from main import app
from security.auth import validate_jwt

client = TestClient(app)

def test_jwt_validation():
    # Test valid token
    valid_token = create_test_token()
    assert validate_jwt(valid_token) is True
    
    # Test expired token
    expired_token = create_test_token(exp_delta=-3600)
    assert validate_jwt(expired_token) is False

def test_role_based_access():
    user_token = create_test_token(roles=['user'])
    admin_token = create_test_token(roles=['admin'])
    
    # Test admin-only endpoint
    response = client.get(
        "/api/admin/metrics",
        headers={"Authorization": f"Bearer {user_token}"}
    )
    assert response.status_code == status.HTTP_403_FORBIDDEN
    
    response = client.get(
        "/api/admin/metrics",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == status.HTTP_200_OK

def test_sql_injection_protection():
    malicious_payload = {"filter": "1; DROP TABLE users;"}
    response = client.post(
        "/api/data/query",
        json=malicious_payload,
        headers={"Authorization": f"Bearer {valid_token}"}
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
