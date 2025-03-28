from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

def upgrade():
    # Create core tables
    op.create_table('facilities',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('location', postgresql.JSONB),
        sa.Column('industry_type', sa.String(50)),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.func.now())
    )
    
    op.create_table('energy_usage',
        sa.Column('timestamp', sa.TIMESTAMP, server_default=sa.func.now()),
        sa.Column('facility_id', sa.String(36), sa.ForeignKey('facilities.id')),
        sa.Column('kwh', sa.Double),
        sa.Column('source', sa.String(20))
    
    op.create_table('water_usage',
        sa.Column('timestamp', sa.TIMESTAMP, server_default=sa.func.now()),
        sa.Column('facility_id', sa.String(36), sa.ForeignKey('facilities.id')),
        sa.Column('cubic_meters', sa.Double),
        sa.Column('source', sa.String(20))
    
    op.create_table('emission_metrics',
        sa.Column('timestamp', sa.TIMESTAMP, server_default=sa.func.now()),
        sa.Column('facility_id', sa.String(36), sa.ForeignKey('facilities.id')),
        sa.Column('co2_kg', sa.Double),
        sa.Column('predicted_co2_kg', sa.Double),
        sa.Column('anomaly_score', sa.Double)
    )

def downgrade():
    op.drop_table('emission_metrics')
    op.drop_table('water_usage')
    op.drop_table('energy_usage')
    op.drop_table('facilities')
