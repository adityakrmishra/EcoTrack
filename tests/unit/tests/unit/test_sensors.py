import pytest
from unittest.mock import Mock, patch
from sensors.energy_monitor import EnergyMonitor
from sensors.exceptions import SensorConnectionError

@pytest.fixture
def mock_sensor():
    mock = Mock()
    mock.read.return_value = {
        'timestamp': '2023-10-10T12:00:00Z',
        'kwh': 150.25,
        'voltage': 220.0
    }
    return mock

def test_sensor_initialization():
    monitor = EnergyMonitor(ip_address="192.168.1.100", port=8080)
    assert monitor.ip == "192.168.1.100"
    assert monitor.sampling_interval == 5.0

@patch('sensors.energy_monitor.requests.get')
def test_successful_reading(mock_get):
    mock_response = Mock()
    mock_response.json.return_value = {
        'readings': {'kwh': 150.0},
        'status': 'OK'
    }
    mock_get.return_value = mock_response
    
    monitor = EnergyMonitor(ip_address="192.168.1.100")
    reading = monitor.get_reading()
    
    assert reading['kwh'] == 150.0
    assert 'timestamp' in reading

@patch('sensors.energy_monitor.requests.get')
def test_sensor_timeout(mock_get):
    mock_get.side_effect = TimeoutError("Connection timed out")
    
    monitor = EnergyMonitor(ip_address="192.168.1.100")
    
    with pytest.raises(SensorConnectionError) as excinfo:
        monitor.get_reading()
    
    assert "Connection timed out" in str(excinfo.value)

def test_data_normalization(mock_sensor):
    from sensors.data_processing import normalize_reading
    
    raw_data = {
        'kwh': '150.25',  # String input
        'voltage': 220,
        'extra_field': 'ignore'
    }
    
    normalized = normalize_reading(raw_data)
    
    assert normalized['kwh'] == 150.25
    assert 'voltage' not in normalized
    assert 'extra_field' not in normalized

def test_concurrent_sensor_polling():
    from concurrent.futures import ThreadPoolExecutor
    from sensors.sensor_manager import SensorManager
    
    manager = SensorManager(poll_interval=0.1)
    
    with ThreadPoolExecutor() as executor:
        future = executor.submit(manager.start_polling)
        # Allow some time for polling
        manager.stop_polling()
        result = future.result(timeout=1)
    
    assert len(result) >= 1
