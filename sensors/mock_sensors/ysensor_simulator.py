import random
import time
from datetime import datetime
from flask import Flask, jsonify
from flask_limiter import Limiter

app = Flask(__name__)
limiter = Limiter(app, default_limits=["100 per minute"])

class SensorSimulator:
    def __init__(self):
        self.energy_base = 5000  # kWh
        self.water_base = 200    # L
        self.failure_mode = False
        
    def generate_energy_data(self):
        if self.failure_mode:
            return {"error": "Sensor offline"}
            
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "voltage": 230 + random.uniform(-0.5, 0.5),
            "current": 15 + random.uniform(-0.2, 0.2),
            "power": 3450 + random.uniform(-10, 10),
            "energy": self.energy_base + (time.time() // 3600),
            "pf": 0.98 + random.uniform(-0.01, 0.01)
        }
        
    def generate_water_data(self):
        if self.failure_mode:
            return {"error": "Valve stuck"}
            
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "flow_rate": 2.5 + random.uniform(-0.1, 0.1),
            "total_volume": self.water_base + (time.time() // 60),
            "pressure": 3.2 + random.uniform(-0.05, 0.05),
            "temp": 18 + random.uniform(-0.5, 0.5)
        }

simulator = SensorSimulator()

@app.route('/api/energy')
def energy_endpoint():
    return jsonify(simulator.generate_energy_data())

@app.route('/api/water')
def water_endpoint():
    return jsonify(simulator.generate_water_data())

@app.route('/api/failure/<mode>')
def set_failure(mode):
    simulator.failure_mode = mode.lower() == "true"
    return jsonify(status="Failure mode " + ("enabled" if simulator.failure_mode else "disabled"))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
