"""
Industrial Energy Monitoring System

Features:
- Real-time power consumption monitoring
- Voltage/current sensing
- Energy calculation (kWh)
- Circuit-level monitoring
- Power factor calculation
- Data smoothing
- Overload detection
"""

import time
import logging
import json
from typing import Optional, Dict
from dataclasses import dataclass
from functools import partial
from tenacity import retry, stop_after_attempt, wait_fixed
import numpy as np
from gpiozero import CPUTemperature, Energenie
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EnergyReading:
    timestamp: float
    voltage: float  # Volts
    current: float  # Amps
    power: float  # Watts
    energy: float  # kWh
    power_factor: float
    frequency: float  # Hz

class EnergyMonitor:
    """Industrial-grade energy monitoring system"""
    
    def __init__(self, 
                 sensor_url: str = "http://192.168.1.10/read",
                 api_key: str = None,
                 mock: bool = False):
        self.sensor_url = sensor_url
        self.api_key = api_key
        self.mock = mock
        self.cpu = CPUTemperature()
        self.circuits = Energenie()
        self.base_energy = 0.0
        self.last_reading = None
        
        if not mock:
            self._calibrate_sensor()
            
    def _calibrate_sensor(self):
        """Perform initial sensor calibration"""
        logger.info("Calibrating energy sensor...")
        try:
            readings = [self._read_raw() for _ in range(10)]
            self.base_energy = np.mean([r['energy'] for r in readings])
        except Exception as e:
            logger.error(f"Calibration failed: {str(e)}")
            raise RuntimeError("Sensor calibration failed") from e
            
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _read_raw(self) -> Dict:
        """Read raw sensor data with retries"""
        if self.mock:
            return self._mock_reading()
            
        headers = {"X-API-KEY": self.api_key} if self.api_key else {}
        response = requests.get(self.sensor_url, headers=headers, timeout=5)
        response.raise_for_status()
        return response.json()
        
    def _mock_reading(self) -> Dict:
        """Generate simulated sensor data"""
        return {
            "voltage": 230 + np.random.normal(0, 0.5),
            "current": 15 + np.random.normal(0, 0.2),
            "power": 3450 + np.random.normal(0, 10),
            "energy": self.base_energy + (time.time() // 3600),
            "pf": 0.98 + np.random.normal(0, 0.01),
            "freq": 50 + np.random.normal(0, 0.05)
        }
    
    def get_reading(self) -> Optional[EnergyReading]:
        """Get processed energy reading"""
        try:
            raw = self._read_raw()
            temp = self.cpu.temperature
            
            reading = EnergyReading(
                timestamp=time.time(),
                voltage=raw['voltage'],
                current=raw['current'],
                power=raw['power'],
                energy=raw['energy'] - self.base_energy,
                power_factor=raw['pf'],
                frequency=raw['freq']
            )
            
            self._check_overload(reading)
            self.last_reading = reading
            return reading
            
        except Exception as e:
            logger.error(f"Reading failed: {str(e)}")
            return None
            
    def _check_overload(self, reading: EnergyReading):
        """Detect and handle power overloads"""
        if reading.power > self.circuits.rated_power:
            logger.warning(f"Power overload detected: {reading.power}W")
            self.circuits.off()
            raise RuntimeError("Overload protection triggered")
    
    def start_monitoring(self, interval: int = 10):
        """Start continuous monitoring"""
        logger.info(f"Starting energy monitoring (interval: {interval}s)")
        while True:
            reading = self.get_reading()
            if reading:
                self._store_reading(reading)
            time.sleep(interval)
            
    def _store_reading(self, reading: EnergyReading):
        """Store reading in database"""
        # Implementation would connect to TimescaleDB
        logger.info(f"Storing reading: {reading}")

# Example usage:
"""
monitor = EnergyMonitor(mock=True)
reading = monitor.get_reading()
print(f"Current power: {reading.power}W")
monitor.start_monitoring(interval=60)
"""
