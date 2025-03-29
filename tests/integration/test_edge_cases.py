def test_massive_data_ingestion():
    # Generate 1M data points
    data = [{"timestamp": f"2023-10-10T{hour:02}:00:00Z", "kwh": hour*10} 
            for hour in range(1_000_000)]
    
    response = client.post(
        "/api/sensors/batch",
        json=data,
        headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code == 202
    assert "batch_id" in response.json()

def test_network_failure_recovery():
    with patch('requests.post') as mock_post:
        mock_post.side_effect = ConnectionError
        response = client.post("/api/sensors/data", json=valid_payload)
        
        # Verify retry mechanism
        assert mock_post.call_count == 3
        assert response.status_code == 503

def test_clock_drift_handling():
    future_time = datetime.now() + timedelta(days=1)
    payload = {"timestamp": future_time.isoformat(), "kwh": 150}
    
    response = client.post("/api/sensors/data", json=payload)
    assert response.status_code == 400
    assert "future timestamp" in response.text
