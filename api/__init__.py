"""
EcoTrack FastAPI Core Configuration
"""

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request
from contextlib import asynccontextmanager
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
import os

# Database
from .utils.database import async_engine, Base
from .models.emission_model import EmissionRecord

# Security
from .utils.security import validate_api_key

# Routers
from .routes.emissions import router as emission_router
from .routes.predictions import router as prediction_router
from .routes.blockchain import router as blockchain_router

# Configuration
from config import settings

security = HTTPBearer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Initialize database connection
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database connection initialized")
    
    # Load ML model
    from ai.predict import load_model
    app.state.ml_model = load_model(settings.MODEL_PATH)
    logger.info(f"Loaded ML model from {settings.MODEL_PATH}")
    
    yield  # App is running
    
    # Cleanup
    await async_engine.dispose()
    logger.info("Database connection closed")

app = FastAPI(
    title="EcoTrack API",
    description="Core API for sustainability metrics tracking and analysis",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    SessionMiddleware,
    secret_key=settings.SECRET_KEY,
    session_cookie="ecotrack_session"
)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

# Error handling
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "success": False,
            "path": request.url.path
        }
    )

# Database dependency
async def get_db() -> AsyncSession:
    async with AsyncSession(async_engine) as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Database error: {str(e)}"
            )
        finally:
            await session.close()

# API routers
app.include_router(
    emission_router,
    prefix="/api/v1/emissions",
    tags=["emissions"],
    dependencies=[Depends(validate_api_key)]
)

app.include_router(
    prediction_router,
    prefix="/api/v1/predictions",
    tags=["predictions"],
    dependencies=[Depends(validate_api_key)]
)

app.include_router(
    blockchain_router,
    prefix="/api/v1/blockchain",
    tags=["blockchain"],
    dependencies=[Depends(validate_api_key)]
)

# Health check endpoint
@app.get("/health", include_in_schema=False)
async def health_check():
    return {
        "status": "healthy",
        "database": "connected" if async_engine else "disconnected",
        "ml_model": "loaded" if app.state.ml_model else "not_loaded"
    }

# Root endpoint
@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Welcome to EcoTrack API",
        "version": app.version,
        "docs": "/api/docs",
        "redoc": "/api/redoc"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", 8000)),
        log_level="info"
    )
