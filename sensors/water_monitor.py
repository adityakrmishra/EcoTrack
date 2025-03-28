"""
Industrial Water Monitoring System

Features:
- Flow rate measurement
- Total consumption tracking
- Leak detection
- Pressure monitoring
- Temperature sensing
- Usage pattern analysis
- Valve control
"""

import time
import math
import logging
from typing import Optional, Dict
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_fixed
from gpiozero import DigitalInputDevice, DigitalOutputDevice
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WaterReading:
    timestamp: float
    flow_rate: float  # L/min
    total_volume: float  # L
    pressure: float  # bar
    temperature: float  # °C
    valve_status: bool

class WaterMonitor:
    """Industrial water monitoring and control system"""
    
    def __init__(self,
                 flow_pin: int = 17,
                 pressure_url: str = "http://192.168.1.11/read",
                 mock: bool = False):
        self.mock = mock
        self.flow_sensor = None
        self.pressure_url = pressure_url
        self.valve = DigitalOutputDevice(24) if not mock else None
        self.volume = 0.0
        self.flow_rate = 0.0
        self.last_pulse = time.time()
        
        if not mock:
            self.flow_sensor = DigitalInputDevice(flow_pin)
            self.flow_sensor.when_activated = self._pulse_handler
            
    def _pulse_handler(self):
        """Handle flow sensor pulses"""
        now = time.time()
        delta = now - self.last_pulse
        self.last_pulse = now
        
        # Calculate instantaneous flow rate (1 pulse = 2.25 mL)
        self.flow_rate = (0.00225 / delta) * 60  # L/min
        self.volume += 0.00225
        
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _read_pressure(self) -> Dict:
        """Read pressure sensor data"""
        if self.mock:
            return {
                "pressure": 3.2 + (0.1 * math.sin(time.time())),
                "temp": 18 + (2 * math.sin(time.time()/3600))
            }
            
        response = requests.get(self.pressure_url, timeout=5)
        response.raise_for_status()
        return response.json()
        
    def get_reading(self) -> Optional[WaterReading]:
        """Get current water system status"""
        try:
            pressure_data = self._read_pressure()
            
            return WaterReading(
                timestamp=time.time(),
                flow_rate=self.flow_rate,
                total_volume=self.volume,
                pressure=pressure_data['pressure'],
                temperature=pressure_data['temp'],
                valve_status=self.valve.is_active if self.valve else False
            )
        except Exception as e:
            logger.error(f"Reading failed: {str(e)}")
            return None
            
    def control_valve(self, state: bool):
        """Control main water valve"""
        if self.valve:
            self.valve.value = state
        logger.info(f"Valve {'open' if state else 'closed'}")
        
    def leak_detection(self) -> bool:
        """Check for potential leaks"""
        if self.mock:
            return False
            
        # Detect continuous flow when valve is closed
        if not self.valve.is_active and self.flow_rate > 0.1:
            logger.warning("Potential leak detected!")
            return True
        return False
        
    def start_monitoring(self, interval: int = 10):
        """Start continuous monitoring"""
        logger.info(f"Starting water monitoring (interval: {interval}s)")
        while True:
            reading = self.get_reading()
            if reading:
                self._store_reading(reading)
                if self.leak_detection():
                    self.control_valve(False)
            time.sleep(interval)
            
    def _store_reading(self, reading: WaterReading):
        """Store reading in database"""
        # Implementation would connect to TimescaleDB
        logger.info(f"Storing reading: {reading}")

# Example usage:
"""
monitor = WaterMonitor(mock=True)
monitor.control_valve(True)
reading = monitor.get_reading()
print(f"Current flow: {reading.flow_rate}L/min")
monitor.start_monitoring(interval=30)
