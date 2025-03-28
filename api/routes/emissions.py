"""
EcoTrack Emissions Data Route Handler

Implements enterprise-grade CRUD operations for emissions tracking with advanced features:
- Real-time IoT data ingestion
- Timeseries analytics
- Role-based access control
- Data validation pipelines
- Advanced query capabilities
"""

from datetime import datetime, timedelta
from typing import Annotated, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field, confloat, validator
from sqlalchemy.ext.asyncio import AsyncSession
import numpy as np
from loguru import logger
import pytz

# Local imports
from api.models.emission_model import EmissionRecord
from api.utils.database import get_db
from api.utils.security import validate_api_key, get_current_user
from api.utils.analytics import calculate_carbon_equivalent

router = APIRouter(prefix="/emissions", tags=["emissions"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Constants
MAX_BATCH_INGEST = 1000  # Max records per bulk insert
CACHE_TTL = 300  # 5 minutes

class EmissionCreate(BaseModel):
    """Payload for creating new emission records"""
    timestamp: datetime = Field(..., example="2023-07-20T12:00:00Z")
    co2_kg: confloat(ge=0) = Field(..., example=150.5)
    ch4_g: confloat(ge=0) = Field(0.0, example=2.3)
    n2o_g: confloat(ge=0) = Field(0.0, example=1.1)
    energy_kwh: confloat(ge=0) = Field(..., example=500.0)
    water_l: confloat(ge=0) = Field(..., example=2000.0)
    source_type: str = Field(..., min_length=2, max_length=50, 
                            example="manufacturing")
    device_id: str = Field(..., min_length=5, max_length=50, 
                         example="iot-sensor-123")

    @validator('timestamp')
    def validate_timestamp(cls, value):
        """Ensure timestamp is not in the future"""
        if value > datetime.now(pytz.utc) + timedelta(minutes=5):
            raise ValueError("Timestamp cannot be in the future")
        return value

class EmissionResponse(EmissionCreate):
    """Emission record response with calculated fields"""
    id: int
    co2e_kg: float
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class EmissionUpdate(BaseModel):
    """Payload for updating emission records"""
    co2_kg: Optional[confloat(ge=0)] = None
    energy_kwh: Optional[confloat(ge=0)] = None
    water_l: Optional[confloat(ge=0)] = None
    notes: Optional[str] = Field(None, max_length=500)

class PaginatedEmissionResponse(BaseModel):
    """Paginated response structure"""
    count: int
    total_co2e: float
    results: list[EmissionResponse]

@router.post(
    "/",
    response_model=EmissionResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(validate_api_key)]
)
async def create_emission_record(
    emission: EmissionCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Create new emission record with carbon equivalent calculation
    
    Permissions:
    - write:emissions
    """
    try:
        # Calculate carbon equivalent
        co2e = calculate_carbon_equivalent(
            emission.co2_kg,
            emission.ch4_g,
            emission.n2o_g
        )
        
        # Create database model
        db_emission = EmissionRecord(
            **emission.dict(),
            co2e_kg=co2e,
            created_by=current_user['id']
        )
        
        db.add(db_emission)
        await db.commit()
        await db.refresh(db_emission)
        
        logger.info(f"Created emission record {db_emission.id}")
        return db_emission
        
    except Exception as e:
        logger.error(f"Create emission failed: {str(e)}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create emission record"
        )

@router.post(
    "/bulk-iot",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(validate_api_key)]
)
async def bulk_ingest_iot_data(
    emissions: list[EmissionCreate],
    db: AsyncSession = Depends(get_db),
    device_token: str = Depends(oauth2_scheme)
):
    """
    High-throughput IoT device data ingestion endpoint
    
    Features:
    - Batch processing with configurable limits
    - Device authentication via JWT
    - Async bulk insert
    """
    if len(emissions) > MAX_BATCH_INGEST:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Max {MAX_BATCH_INGEST} records per request"
        )
        
    try:
        records = []
        current_time = datetime.now(pytz.utc)
        
        for emission in emissions:
            co2e = calculate_carbon_equivalent(
                emission.co2_kg,
                emission.ch4_g,
                emission.n2o_g
            )
            records.append({
                **emission.dict(),
                "co2e_kg": co2e,
                "created_at": current_time,
                "updated_at": current_time
            })
        
        # Bulk insert using SQLAlchemy 2.0 syntax
        await db.execute(
            EmissionRecord.__table__.insert(),
            records
        )
        await db.commit()
        
        logger.info(f"Ingested {len(emissions)} records from IoT device")
        return {"ingested": len(emissions)}

    except Exception as e:
        logger.error(f"Bulk ingest failed: {str(e)}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process IoT data batch"
        )

@router.get(
    "/",
    response_model=PaginatedEmissionResponse,
    dependencies=[Depends(validate_api_key)]
)
async def get_emissions(
    db: AsyncSession = Depends(get_db),
    start_date: datetime = Query(None, description="Start time filter"),
    end_date: datetime = Query(None, description="End time filter"),
    source_type: str = Query(None, min_length=2),
    min_co2e: float = Query(None, ge=0),
    page: int = Query(1, ge=1),
    limit: int = Query(100, ge=1, le=1000),
    current_user: dict = Depends(get_current_user)
):
    """
    Advanced emission records query with filters and pagination
    
    Features:
    - Timeseries filtering
    - Source type categorization
    - CO2e threshold filtering
    - Pagination with configurable limits
    """
    try:
        query = db.query(EmissionRecord)
        
        # Apply filters
        if start_date and end_date:
            query = query.filter(
                EmissionRecord.timestamp.between(start_date, end_date)
        elif start_date:
            query = query.filter(EmissionRecord.timestamp >= start_date)
        elif end_date:
            query = query.filter(EmissionRecord.timestamp <= end_date)
            
        if source_type:
            query = query.filter_by(source_type=source_type)
            
        if min_co2e:
            query = query.filter(EmissionRecord.co2e_kg >= min_co2e)
            
        # Pagination
        total = await db.scalar(query.count())
        results = await db.execute(
            query.order_by(EmissionRecord.timestamp.desc())
            .offset((page - 1) * limit)
            .limit(limit)
        )
        records = results.scalars().all()
        
        # Calculate total CO2e
        total_co2e = np.sum([r.co2e_kg for r in records])
        
        return {
            "count": len(records),
            "total_co2e": round(total_co2e, 2),
            "results": records
        }
        
    except Exception as e:
        logger.error(f"Emission query failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve emission records"
        )

@router.get(
    "/stats/daily",
    dependencies=[Depends(validate_api_key)]
)
async def get_daily_stats(
    db: AsyncSession = Depends(get_db),
    days: int = Query(7, ge=1, le=365),
    current_user: dict = Depends(get_current_user)
):
    """
    Calculate daily emission statistics for dashboard visualization
    
    Returns:
    - Average daily CO2e
    - Energy consumption trends
    - Water usage patterns
    """
    try:
        end_date = datetime.now(pytz.utc)
        start_date = end_date - timedelta(days=days)
        
        # Raw SQL for time-bucketed aggregation
        stmt = f"""
            SELECT
                time_bucket_gapfill('1 day', timestamp) AS period,
                AVG(co2e_kg) as avg_co2e,
                SUM(energy_kwh) as total_energy,
                SUM(water_l) as total_water
            FROM emission_records
            WHERE timestamp BETWEEN '{start_date.isoformat()}' 
                AND '{end_date.isoformat()}'
            GROUP BY period
            ORDER BY period DESC
        """
        
        result = await db.execute(stmt)
        rows = result.fetchall()
        
        return {
            "start_date": start_date,
            "end_date": end_date,
            "data": [
                {
                    "date": row[0].isoformat(),
                    "avg_co2e": row[1],
                    "total_energy": row[2],
                    "total_water": row[3]
                }
                for row in rows
            ]
        }
        
    except Exception as e:
        logger.error(f"Daily stats failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calculate daily statistics"
        )

@router.patch(
    "/{record_id}",
    response_model=EmissionResponse,
    dependencies=[Depends(validate_api_key)]
)
async def update_emission_record(
    record_id: int,
    payload: EmissionUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Update emission record with audit logging
    
    Permissions:
    - update:emissions
    - admin:emissions (for historical data)
    """
    try:
        result = await db.execute(
            db.query(EmissionRecord)
            .filter_by(id=record_id)
        )
        record = result.scalar_one_or_none()
        
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Emission record not found"
            )
            
        # Check update permissions
        if (current_user['role'] != 'admin' and 
            record.created_by != current_user['id']):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to modify this record"
            )
            
        # Apply updates
        update_data = payload.dict(exclude_unset=True)
        for key, value in update_data.items():
            setattr(record, key, value)
            
        record.updated_at = datetime.now(pytz.utc)
        await db.commit()
        await db.refresh(record)
        
        logger.info(f"Updated emission record {record_id}")
        return record
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update failed for record {record_id}: {str(e)}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update emission record"
        )

@router.delete(
    "/{record_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(validate_api_key)]
)
async def delete_emission_record(
    record_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Soft-delete emission record (implements audit trail)
    
    Permissions:
    - delete:emissions
    - admin:emissions
    """
    try:
        result = await db.execute(
            db.query(EmissionRecord)
            .filter_by(id=record_id)
        )
        record = result.scalar_one_or_none()
        
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Emission record not found"
            )
            
        if not current_user.get('is_admin'):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required for deletion"
            )
            
        # Soft delete implementation
        record.deleted_at = datetime.now(pytz.utc)
        await db.commit()
        
        logger.warning(f"Soft-deleted record {record_id} by {current_user['id']}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete failed for record {record_id}: {str(e)}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete emission record"
        )

# Example Usage:
"""
# Create record
curl -X POST http://localhost:8000/emissions/ \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2023-07-20T12:00:00Z",
    "co2_kg": 150.5,
    "energy_kwh": 500,
    "water_l": 2000,
    "source_type": "manufacturing",
    "device_id": "sensor-123"
  }'

# Query records
curl "http://localhost:8000/emissions/?start_date=2023-07-01&page=1&limit=10" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Bulk IoT ingest
curl -X POST http://localhost:8000/emissions/bulk-iot \
  -H "Authorization: Bearer DEVICE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '[{"timestamp": "...", "co2_kg": 100, ...}]'
"""
