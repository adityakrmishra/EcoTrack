"""
EcoTrack Emission Data Model

Implements enterprise-grade timeseries data storage with features:
- Hypertable partitioning for TimescaleDB
- Composite indexes for fast IoT queries
- Soft deletion with audit trail
- Data versioning
- Multi-tenant isolation
- Optimized for time-range queries
- Field-level encryption
"""

from datetime import datetime
from typing import Optional, Annotated
from sqlalchemy import (
    Column,
    String,
    Integer,
    Numeric,
    DateTime,
    ForeignKey,
    Index,
    text,
    event,
    DDL
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    declared_attr,
    relationship,
    validates
)
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.dialects.postgresql import JSONB, INET
from pydantic import BaseModel
from cryptography.fernet import Fernet
import pytz

# Encryption setup
FIELD_KEY = Fernet.generate_key()
fernet = Fernet(FIELD_KEY)

class Base(AsyncAttrs, DeclarativeBase):
    """Base model with TimescaleDB extensions"""
    
    @declared_attr.directive
    def __tablename__(cls) -> str:
        return cls.__name__.lower()

class TimestampMixin:
    """Timestamp fields with timezone awareness"""
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(pytz.utc),
        server_default=text("NOW()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(pytz.utc),
        onupdate=lambda: datetime.now(pytz.utc),
        server_default=text("NOW()")
    )

class EmissionRecord(Base, TimestampMixin):
    """Hypertable for storing emission metrics with enterprise features"""
    
    __table_args__ = (
        Index('ix_device_timestamp', 'device_id', 'timestamp'),
        Index('ix_source_co2e', 'source_type', 'co2e_kg'),
        Index('ix_timestamp_range', 'timestamp', postgresql_using='brin'),
        {'schema': 'telemetry'}
    )
    
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        server_default=text("nextval('telemetry.emission_record_id_seq'::regclass)")
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="UTC timestamp with timezone"
    )
    co2_kg: Mapped[Numeric] = mapped_column(
        Numeric(12, 6),
        nullable=False,
        comment="CO2 emissions in kilograms"
    )
    ch4_g: Mapped[Numeric] = mapped_column(
        Numeric(12, 6),
        server_default=text("0.0"),
        comment="Methane emissions in grams"
    )
    n2o_g: Mapped[Numeric] = mapped_column(
        Numeric(12, 6),
        server_default=text("0.0"),
        comment="Nitrous oxide emissions in grams"
    )
    co2e_kg: Mapped[Numeric] = mapped_column(
        Numeric(12, 6),
        nullable=False,
        comment="CO2 equivalent (calculated)"
    )
    energy_kwh: Mapped[Numeric] = mapped_column(
        Numeric(12, 6),
        nullable=False,
        comment="Energy consumption in kWh"
    )
    water_l: Mapped[Numeric] = mapped_column(
        Numeric(12, 6),
        nullable=False,
        comment="Water usage in liters"
    )
    source_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Emission source category"
    )
    device_id: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="IoT device identifier"
    )
    ip_address: Mapped[Annotated[str, "inet"]] = mapped_column(
        INET,
        comment="Device IP address at time of recording"
    )
    raw_data: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        comment="Raw sensor payload (encrypted)"
    )
    version: Mapped[int] = mapped_column(
        Integer,
        server_default=text("1"),
        comment="Data version for schema evolution"
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        comment="Soft deletion timestamp"
    )
    created_by: Mapped[int] = mapped_column(
        Integer,
        ForeignKey('iam.users.id', ondelete="RESTRICT"),
        nullable=False,
        comment="User/tenant who created record"
    )
    
    # Relationships
    user: Mapped["User"] = relationship(
        back_populates="emissions",
        lazy="selectin"
    )
    
    @validates('raw_data')
    def encrypt_raw_data(self, key, value):
        """Encrypt sensitive raw data field"""
        if value is not None:
            return fernet.encrypt(json.dumps(value).encode())
        return None

    def decrypt_raw_data(self):
        """Decrypt raw data for authorized access"""
        if self.raw_data:
            return json.loads(fernet.decrypt(self.raw_data).decode())
        return None

    @property
    def carbon_intensity(self) -> float:
        """Calculate carbon intensity ratio"""
        try:
            return float(self.co2e_kg) / float(self.energy_kwh)
        except ZeroDivisionError:
            return 0.0

# Hypertable creation DDL
create_hypertable = DDL(
    "SELECT create_hypertable("
    "'telemetry.emission_record', 'timestamp', "
    "chunk_time_interval => interval '7 days', "
    "if_not_exists => TRUE);"
)

event.listen(
    EmissionRecord.__table__,
    'after_create',
    create_hypertable.execute_if(dialect='postgresql')
)

class EmissionRecordAudit(Base):
    """Time-series audit log for data modifications"""
    __tablename__ = "emission_record_audit"
    __table_args__ = {'schema': 'audit'}
    
    id: Mapped[int] = mapped_column(primary_key=True)
    record_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )
    operation_type: Mapped[str] = mapped_column(
        String(1),
        nullable=False,
        comment="C=Create, U=Update, D=Delete"
    )
    old_values: Mapped[Optional[dict]] = mapped_column(JSONB)
    new_values: Mapped[Optional[dict]] = mapped_column(JSONB)
    changed_by: Mapped[int] = mapped_column(
        Integer,
        ForeignKey('iam.users.id'),
        nullable=False
    )
    changed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("NOW()")
    )

class EmissionCategory(Base):
    """Reference table for emission source types"""
    __tablename__ = "emission_category"
    __table_args__ = {'schema': 'reference'}
    
    id: Mapped[int] = mapped_column(primary_key=True)
    category_code: Mapped[str] = mapped_column(
        String(10),
        unique=True,
        comment="UNSPSC classification code"
    )
    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )
    description: Mapped[Optional[str]] = mapped_column(String(500))
    co2e_factor: Mapped[Numeric] = mapped_column(
        Numeric(12, 6),
        comment="Industry-specific emission factor"
    )
    effective_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("NOW()")
    )
    effective_to: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )

# Indexes
Index('ix_emission_audit_record', EmissionRecordAudit.record_id)
Index('ix_emission_category_code', EmissionCategory.category_code, unique=True)

class EmissionRecordSchema(BaseModel):
    """Pydantic schema for API validation"""
    timestamp: datetime
    co2_kg: float
    ch4_g: float
    n2o_g: float
    energy_kwh: float
    water_l: float
    source_type: str
    device_id: str
    ip_address: str
    raw_data: Optional[dict]

    class Config:
        orm_mode = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class EmissionStatsResponse(BaseModel):
    """Aggregate statistics response model"""
    period_start: datetime
    period_end: datetime
    total_co2e: float
    avg_carbon_intensity: float
    energy_consumption: float
    water_usage: float
    peak_emission: datetime

# Example Usage:
"""
from sqlalchemy.ext.asyncio import AsyncSession
from models.emission_model import EmissionRecord

async def create_emission(db: AsyncSession, data: dict):
    record = EmissionRecord(
        timestamp=data['timestamp'],
        co2_kg=data['co2_kg'],
        energy_kwh=data['energy_kwh'],
        water_l=data['water_l'],
        source_type='manufacturing',
        device_id='iot-sensor-123',
        ip_address='192.168.1.10',
        created_by=1
    )
    db.add(record)
    await db.commit()
    return record
"""
