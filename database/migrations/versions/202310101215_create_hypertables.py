from alembic import op
import sqlalchemy as sa

def upgrade():
    # Convert tables to hypertables
    op.execute("""
        SELECT create_hypertable(
            'emission_metrics', 
            'timestamp',
            chunk_time_interval => INTERVAL '1 week',
            if_not_exists => TRUE
        );
    """)
    
    op.execute("""
        SELECT create_hypertable(
            'energy_usage', 
            'timestamp',
            chunk_time_interval => INTERVAL '1 day',
            if_not_exists => TRUE
        );
    """)
    
    # Create compression policies
    op.execute("""
        ALTER TABLE emission_metrics SET (
            timescaledb.compress,
            timescaledb.compress_orderby = 'timestamp DESC',
            timescaledb.compress_segmentby = 'facility_id'
        );
    """)
    
    op.execute("""
        SELECT add_compression_policy('emission_metrics', INTERVAL '30 days');
    """)

def downgrade():
    op.execute("SELECT remove_compression_policy('emission_metrics');")
    op.execute("SELECT remove_retention_policy('emission_metrics');")
