import pytest
from sqlalchemy.exc import IntegrityError
from datetime import datetime, timedelta
from database.models import Facility, EnergyUsage, Session

@pytest.fixture(scope="module")
def test_session():
    session = Session()
    yield session
    session.rollback()
    session.close()

def test_facility_creation(test_session):
    facility = Facility(
        name="Test Facility",
        location={"lat": 40.7128, "lng": -74.0060},
        industry_type="manufacturing"
    )
    test_session.add(facility)
    test_session.commit()
    
    assert facility.id is not None
    assert facility.created_at is not None
    assert facility.industry_type == "manufacturing"

def test_energy_usage_constraints(test_session):
    invalid_usage = EnergyUsage(
        timestamp=datetime.now(),
        kwh=-50.0,  # Negative energy should fail
        source="invalid_source"
    )
    test_session.add(invalid_usage)
    
    with pytest.raises(IntegrityError):
        test_session.commit()

def test_timescale_hypertable(test_session):
    # Test time-series specific functions
    result = test_session.execute("""
        SELECT create_hypertable(
            'energy_usage', 
            'timestamp',
            if_not_exists => TRUE
        )
    """)
    assert result.scalar() is not None

def test_energy_usage_aggregation(test_session):
    # Insert test data
    for i in range(5):
        usage = EnergyUsage(
            timestamp=datetime.now() - timedelta(hours=i),
            facility_id=1,
            kwh=100.0 * (i+1),
            source="test"
        )
        test_session.add(usage)
    test_session.commit()
    
    # Query hourly aggregates
    result = test_session.execute("""
        SELECT time_bucket('1 hour', timestamp) AS bucket,
               SUM(kwh) AS total_energy
        FROM energy_usage
        GROUP BY bucket
    """)
    
    assert len(result.fetchall()) >= 1

def test_model_relationships(test_session):
    facility = test_session.query(Facility).first()
    usage = EnergyUsage(
        timestamp=datetime.now(),
        facility
