# EcoTrack Installation Guide

## Prerequisites
- Python 3.8+
- PostgreSQL 12+ with TimescaleDB extension
- Docker Engine 20.10+
- Node.js 16.x (for frontend)
- IoT Sensors (Optional for development)

## Installation Steps

1. Clone Repository:
```bash
git clone https://github.com/adityakrmishra/EcoTrack.git
cd EcoTrack
```
2. Create Virtual Environment:
```
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .\.venv\Scripts\activate  # Windows
```
3. Install Dependencies:
```
pip install -r requirements.txt
```
4. Database Setup:
```
# Create .env file from template
cp .env.example .env

# Initialize TimescaleDB
python database/init_db.py

# Run migrations
alembic upgrade head
```
5.   Configure IoT Sensors:
```
# Edit sensor configuration
nano config/sensors.yaml
```

## Running the System
```
# Start backend server
uvicorn api.main:app --reload --port 8000

# Start frontend
cd frontend
npm install
npm start
```

## Testing
```
# Run unit tests
pytest tests/unit

# Run integration tests
pytest tests/integration```
```

Note: For production deployment, use Docker containers and configure proper security settings.


---

### File: `docs/api-reference.md`
```markdown
# API Reference

## Base URL
`http:`//localhost:8000/api/v1`
```
## Authentication
```http
POST /auth/token
Content-Type: application/json

{
  "username": "admin",
  "password": "securepassword"
}
```
## Sensor Data Endpoints
Get Real-time Metrics
```
GET /sensors/current
Headers:
  Authorization: Bearer <your-api-key>
```

Response:
```
{
  "timestamp": "2023-10-10T12:00:00Z",
  "energy_kwh": 150.25,
  "water_m3": 12.8,
  "co2_kg": 245.7
}
```
### Submit Sensor Batch
```
POST /sensors/batch
Headers:
  Content-Type: application/json
  Authorization: Bearer <your-api-key>

Body:
[
  {
    "sensor_id": "energy-01",
    "timestamp": "2023-10-10T12:00:00Z",
    "value": 150.25
  }
]
```

## Predictions Endpoints
Get Emission Forecast
```HTTP
GET /predictions/forecast?days=7
Headers:
  Authorization: Bearer <your-api-key>
```

```JSON
{
  "predictions": [
    {
      "date": "2023-10-11",
      "predicted_co2": 230.5,
      "confidence_interval": [215.2, 245.8]
    }
  ]
}
```

## OpenAPI Specification
```YAML
openapi: 3.0.0
info:
  title: EcoTrack API
  version: 1.0.0
servers:
  - url: http://localhost:8000/api/v1
paths:
  /sensors/current:
    get:
      summary: Get current sensor readings
      responses:
        200:
          description: Successful response
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/SensorData'
```
