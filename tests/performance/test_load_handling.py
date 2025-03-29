import pytest
from locust import HttpUser, task, between

class SustainabilityLoadTest(HttpUser):
    wait_time = between(1, 5)
    
    @task
    def submit_sensor_data(self):
        payload = {
            "timestamp": "2023-10-10T12:00:00Z",
            "kwh": 150.25,
            "source": "solar"
        }
        self.client.post(
            "/api/sensors/data",
            json=payload,
            headers={"Authorization": "Bearer $TOKEN"}
        )
    
    @task(3)
    def get_predictions(self):
        self.client.get("/api/predict?days=7")
    
    def on_start(self):
        self.token = self.login()
    
    def login(self):
        response = self.client.post("/auth/token", json={
            "username": "load_user",
            "password": "test_password"
        })
        return response.json()["access_token"]

@pytest.mark.performance
def test_high_throughput(locust_env):
    runner = locust_env.create_local_runner()
    runner.start(500, spawn_rate=100)
    assert runner.stats.total.fail_ratio < 0.01
