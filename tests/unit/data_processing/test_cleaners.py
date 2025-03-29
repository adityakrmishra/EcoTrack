import pytest
import numpy as np
from data_processing.cleaners import OutlierHandler, MissingDataImputer

@pytest.fixture
def sample_data():
    return [10, 12, 14, 1000, 15, np.nan, 16]

def test_outlier_detection(sample_data):
    handler = OutlierHandler(method='iqr')
    cleaned = handler.fit_transform(sample_data)
    
    assert 1000 not in cleaned
    assert len(cleaned) == 5

def test_imputation_strategies(sample_data):
    imputer = MissingDataImputer(strategy='time_linear')
    result = imputer.fit_transform(sample_data)
    
    assert not np.isnan(result).any()
    assert np.allclose(result[4:7], [15, 15.5, 16])

def test_sensor_boundary_validation():
    validator = SensorDataValidator(
        min_energy=0,
        max_energy=10000,
        allowed_sources=['solar', 'grid']
    )
    
    assert validator.validate({'energy': 5000, 'source': 'solar'})
    assert not validator.validate({'energy': -50, 'source': 'invalid'})
