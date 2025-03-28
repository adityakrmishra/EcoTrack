"""
EcoTrack API Entry Point with Advanced Configuration

Handles server initialization, environment validation, and production-grade deployment setup.
"""

import os
import sys
import signal
import argparse
import time
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
import uvicorn
from loguru import logger
from pydantic import BaseModel, ValidationError
import psutil

# Custom imports
from .utils.database import test_db_connection
from .utils.security import check_ssl_files
from ai.predict import model_loader

class ServerConfig(BaseModel):
    """Pydantic model for environment validation"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    log_level: str = "info"
    debug: bool = False
    ssl_enabled: bool = False
    ssl_key_path: str = None
    ssl_cert_path: str = None
    timeout_keep_alive: int = 300
    max_request_size: int = 1024 * 1024 * 10  # 10MB

    class Config:
        env_prefix = "API_"
        case_sensitive = False

def configure_logging(debug: bool = False):
    """Configure structured logging with rotation and retention"""
    logger.remove()  # Remove default handler
    
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>"
    
    if debug:
        logger.add(
            sys.stderr,
            format=log_format,
            level="DEBUG",
            backtrace=True,
            diagnose=True
        )
    else:
        logger.add(
            "logs/api.log",
            rotation="500 MB",
            retention="30 days",
            compression="zip",
            format=log_format,
            level="INFO",
            enqueue=True
        )
        
        # JSON logging for production analysis
        logger.add(
            "logs/api.json.log",
            rotation="500 MB",
            retention="30 days",
            format=lambda record: record["extra"].get("json", ""),
            serialize=True,
            level="INFO",
            enqueue=True
        )

def handle_exit(signal_received, frame):
    """Graceful shutdown on SIGINT or SIGTERM"""
    logger.warning(f"Received exit signal {signal_received.name}...")
    try:
        # Cleanup resources
        if model_loader.is_model_loaded():
            model_loader.unload_model()
            logger.info("ML model unloaded successfully")
            
        # Report system status
        process = psutil.Process(os.getpid())
        logger.info(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        logger.error(f"Shutdown cleanup failed: {str(e)}")
    finally:
        sys.exit(0)

def validate_environment(config: ServerConfig):
    """Pre-flight environment validation"""
    errors = []
    
    # Check database connectivity
    try:
        test_db_connection()
    except Exception as e:
        errors.append(f"Database connection failed: {str(e)}")
    
    # Validate SSL configuration
    if config.ssl_enabled:
        if not check_ssl_files(config.ssl_key_path, config.ssl_cert_path):
            errors.append("Invalid SSL certificate configuration")
    
    # Check model file existence
    if not Path(model_loader.DEFAULT_MODEL_PATH).exists():
        errors.append(f"Model file not found at {model_loader.DEFAULT_MODEL_PATH}")
    
    if errors:
        logger.critical("Environment validation failed:")
        for error in errors:
            logger.error(error)
        sys.exit(1)

def parse_cli_args() -> Dict[str, Any]:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="EcoTrack API Server")
    parser.add_argument("--host", help="Override API host")
    parser.add_argument("--port", type=int, help="Override API port")
    parser.add_argument("--workers", type=int, help="Override worker count")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return vars(parser.parse_args())

def run_server():
    """Production-grade server initialization"""
    try:
        # Load environment variables
        load_dotenv(".env")
        cli_args = parse_cli_args()
        
        # Merge config sources
        config = ServerConfig(**cli_args)
        configure_logging(config.debug)
        
        # Register signal handlers
        signal.signal(signal.SIGINT, handle_exit)
        signal.signal(signal.SIGTERM, handle_exit)
        
        # Validate environment
        validate_environment(config)
        
        # Create PID file
        with open("api.pid", "w") as pid_file:
            pid_file.write(str(os.getpid()))
            
        # Startup banner
        logger.info(f"""
        ███████╗ ██████╗███████╗████████╗██████╗  ██████╗█████╗ ██╗  ██╗
        ██╔════╝██╔════╝██╔════╝╚══██╔══╝██╔══██╗██╔════╝██╔══██╗██║ ██╔╝
        █████╗  ██║     █████╗     ██║   ██████╔╝██║     ███████║█████╔╝ 
        ██╔══╝  ██║     ██╔══╝     ██║   ██╔══██╗██║     ██╔══██║██╔═██╗ 
        ███████╗╚██████╗███████╗   ██║   ██║  ██║╚██████╗██║  ██║██║  ██╗
        ╚══════╝ ╚═════╝╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝
        
        Environment: {'DEVELOPMENT' if config.debug else 'PRODUCTION'}
        Version: {os.getenv('APP_VERSION', '1.0.0')}
        Host: {config.host}
        Port: {config.port}
        Workers: {config.workers}
        SSL Enabled: {config.ssl_enabled}
        """)
        
        # Server configuration
        server_args = {
            "app": "api.main:app",
            "host": config.host,
            "port": config.port,
            "workers": config.workers,
            "log_level": config.log_level.lower(),
            "reload": config.debug,
            "proxy_headers": True,
            "timeout_keep_alive": config.timeout_keep_alive,
            "limit_max_requests": 1000 if not config.debug else None,
            "factory": True
        }
        
        if config.ssl_enabled:
            server_args.update({
                "ssl_keyfile": config.ssl_key_path,
                "ssl_certfile": config.ssl_cert_path,
                "ssl_version": 2
            })
            
        # Start server
        uvicorn.run(**server_args)
        
    except ValidationError as e:
        logger.critical(f"Configuration validation failed: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Fatal startup error: {str(e)}")
        sys.exit(1)
    finally:
        # Cleanup PID file
        if Path("api.pid").exists():
            os.remove("api.pid")

if __name__ == "__main__":
    run_server()
