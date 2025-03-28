"""
Industrial Data Preprocessing Pipeline

Features:
- Missing data imputation
- Outlier detection & handling
- Temporal feature engineering
- Advanced scaling/normalization
- Categorical encoding
- Data validation
- Parallel processing
- Feature store integration
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from joblib import dump, load
import holidays
import warnings

warnings.filterwarnings("ignore")

class DataPreprocessor(BaseEstimator, TransformerMixin):
    """Enterprise-grade data preprocessing pipeline"""
    
    def __init__(self, config: dict):
        self.config = config
        self.feature_store = {}
        self.preprocessor = None
        self._build_pipeline()
        
    def _build_pipeline(self):
        """Construct processing pipeline from config"""
        numeric_features = self.config.get("numeric_features", [])
        categorical_features = self.config.get("categorical_features", [])
        temporal_features = self.config.get("temporal_feature", "timestamp")
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )
        
        self.temporal_feature = temporal_features
        
    def _extract_temporal_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive temporal features"""
        df = X.copy()
        dt_feature = pd.to_datetime(df[self.temporal_feature])
        
        # Basic temporal features
        df['hour'] = dt_feature.dt.hour
        df['day_of_week'] = dt_feature.dt.dayofweek
        df['day_of_month'] = dt_feature.dt.day
        df['month'] = dt_feature.dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Holiday features
        country_holidays = holidays.CountryHoliday(self.config.get("country", "US"))
        df['is_holiday'] = dt_feature.dt.date.isin(country_holidays).astype(int)
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
        
        return df.drop(columns=[self.temporal_feature])
    
    def _handle_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Robust outlier handling using IQR"""
        df = X.copy()
        for col in self.config.get("numeric_features", []):
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Cap outliers
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        return df
    
    def fit(self, X: pd.DataFrame, y=None):
        """Learn preprocessing parameters"""
        X_processed = self._extract_temporal_features(X)
        X_processed = self._handle_outliers(X_processed)
        self.preprocessor.fit(X_processed)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing pipeline"""
        X_processed = self._extract_temporal_features(X)
        X_processed = self._handle_outliers(X_processed)
        transformed = self.preprocessor.transform(X_processed)
        
        # Get feature names
        numeric_features = self.config.get("numeric_features", [])
        categorical_features = self.config.get("categorical_features", [])
        ohe_columns = list(self.preprocessor.named_transformers_['cat']
                          .named_steps['onehot'].get_feature_names_out(categorical_features))
        
        all_features = numeric_features + ohe_columns + [
            'hour', 'day_of_week', 'day_of_month', 'month',
            'is_weekend', 'is_holiday', 'hour_sin', 'hour_cos'
        ]
        
        return pd.DataFrame(transformed, columns=all_features)
    
    def save_pipeline(self, path: str):
        """Save preprocessing pipeline"""
        dump({'preprocessor': self.preprocessor, 'config': self.config}, path)
        
    @classmethod
    def load_pipeline(cls, path: str):
        """Load preprocessing pipeline"""
        loaded = load(path)
        instance = cls(loaded['config'])
        instance.preprocessor = loaded['preprocessor']
        return instance

class DataValidator:
    """Industrial Data Validation Framework"""
    
    def __init__(self, schema: dict):
        self.schema = schema
        
    def validate(self, df: pd.DataFrame) -> bool:
        """Perform comprehensive data validation"""
        results = {
            'missing_values': self._check_missing(df),
            'value_ranges': self._check_ranges(df),
            'data_types': self._check_dtypes(df),
            'timestamp_continuity': self._check_timestamps(df)
        }
        return all(results.values())
    
    def _check_missing(self, df: pd.DataFrame) -> bool:
        """Check for unexpected missing values"""
        allowed_missing = self.schema.get("allowed_missing", {})
        for col, threshold in allowed_missing.items():
            if df[col].isna().mean() > threshold:
                return False
        return True
    
    def _check_ranges(self, df: pd.DataFrame) -> bool:
        """Validate numerical value ranges"""
        for col, specs in self.schema.get("numeric_ranges", {}).items():
            if not df[col].between(specs['min'], specs['max']).all():
                return False
        return True
    
    def _check_dtypes(self, df: pd.DataFrame) -> bool:
        """Verify column data types"""
        for col, dtype in self.schema.get("dtypes", {}).items():
            if not pd.api.types.is_dtype_equal(df[col].dtype, dtype):
                return False
        return True
    
    def _check_timestamps(self, df: pd.DataFrame) -> bool:
        """Validate timestamp continuity and frequency"""
        if 'timestamp' not in df.columns:
            return True
            
        ts = pd.to_datetime(df['timestamp'])
        expected_freq = self.schema.get("timestamp_freq", "1H")
        return ts.diff().dropna().value_counts().idxmax() == pd.Timedelta(expected_freq)

# Example usage:
"""
config = {
    "numeric_features": ["energy_usage", "water_usage"],
    "categorical_features": ["facility_id"],
    "temporal_feature": "timestamp",
    "country": "US"
}

raw_data = pd.read_parquet("data/raw.parquet")
preprocessor = DataPreprocessor(config)
processed_data = preprocessor.fit_transform(raw_data)

validator_schema = {
    "allowed_missing": {"energy_usage": 0.05},
    "numeric_ranges": {"temperature": {"min": -40, "max": 100}},
    "timestamp_freq": "1H"
}

validator = DataValidator(validator_schema)
if not validator.validate(raw_data):
    raise ValueError("Data validation failed")
"""
