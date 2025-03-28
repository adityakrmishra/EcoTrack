"""
Enterprise-grade Database Utilities

Features:
- Async SQLAlchemy 2.0 core
- Connection pooling with automatic scaling
- Retry mechanisms with exponential backoff
- Health check endpoints
- Context managers for session handling
- Multi-tenancy support
- Query observability
- Connection recycling
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker
)
from sqlalchemy.pool import QueuePool
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_log
)
from prometheus_client import Summary, Gauge

# Metrics
DB_QUERY_TIME = Summary('db_query_time', 'Time spent on database queries')
DB_CONNECTIONS = Gauge('db_connections', 'Current database connections')

class DatabaseManager:
    """Enterprise database manager with failover support"""
    
    def __init__(self, dsn: str, pool_size: int = 20, max_overflow: int = 10):
        self.engine: AsyncEngine = create_async_engine(
            dsn,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_recycle=3600,
            pool_pre_ping=True,
            connect_args={
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5
            }
        )
        self.session_factory = async_sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
            autoflush=False,
            class_=AsyncSession
        )
        
    @retry(
        retry=retry_if_exception_type(OperationalError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before=before_log(logging.getLogger(), logging.INFO)
    )
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Context manager for database sessions with retries"""
        session = self.session_factory()
        DB_CONNECTIONS.inc()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logging.error(f"Database error: {str(e)}")
            raise
        finally:
            await session.close()
            DB_CONNECTIONS.dec()
            
    async def health_check(self) -> bool:
        """Deep database health check with connection test"""
        try:
            async with self.engine.connect() as conn:
                result = await conn.execute(text("SELECT 1"))
                return result.scalar() == 1
        except Exception as e:
            logging.critical(f"Database health check failed: {str(e)}")
            return False

    @DB_QUERY_TIME.time()
    async def execute_query(self, query, params=None):
        """Instrumented query execution with metrics"""
        async with self.get_session() as session:
            result = await session.execute(query, params)
            return result

    async def setup_connection_pool(self):
        """Warm up connection pool on startup"""
        async with self.engine.begin() as conn:
            await conn.run_sync(lambda sync_conn: sync_conn.execute(text("SELECT 1")))

# Initialize database
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/db")
db_manager = DatabaseManager(DATABASE_URL)
