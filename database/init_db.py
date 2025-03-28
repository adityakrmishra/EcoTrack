import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from psycopg2.sql import SQL, Identifier
from dotenv import load_dotenv

load_dotenv()

def init_timescale():
    """Initialize TimescaleDB database and enable extension"""
    try:
        # Connect to maintenance database
        conn = psycopg2.connect(
            dbname='postgres',
            user=os.getenv('TSDB_USER'),
            password=os.getenv('TSDB_PASSWORD'),
            host=os.getenv('TSDB_HOST'),
            port=os.getenv('TSDB_PORT')
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Create database if not exists
        cursor.execute(SQL("""
            SELECT 1 FROM pg_database WHERE datname = %s
        """), (os.getenv('TSDB_NAME'),))
        
        if not cursor.fetchone():
            cursor.execute(SQL(
                "CREATE DATABASE {}"
            ).format(Identifier(os.getenv('TSDB_NAME'))))
        
        # Connect to target database
        conn.close()
        conn = psycopg2.connect(
            dbname=os.getenv('TSDB_NAME'),
            user=os.getenv('TSDB_USER'),
            password=os.getenv('TSDB_PASSWORD'),
            host=os.getenv('TSDB_HOST'),
            port=os.getenv('TSDB_PORT')
        )
        cursor = conn.cursor()
        
        # Enable TimescaleDB extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
        
        # Create tables
        with open('database/schema/tables.sql') as f:
            cursor.execute(f.read())
        
        # Create hypertables
        with open('database/schema/hypertables.sql') as f:
            cursor.execute(f.read())
        
        conn.commit()
        print("Database initialized successfully")
        
    except Exception as e:
        print(f"Initialization failed: {str(e)}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    init_timescale()
