#!/usr/bin/env python3
import os
import csv
import json
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_ingestion.log'),
        logging.StreamHandler()
    ]
)

class DataIngestor:
    def __init__(self):
        self.conn = psycopg2.connect(
            dbname=os.getenv('TSDB_NAME'),
            user=os.getenv('TSDB_USER'),
            password=os.getenv('TSDB_PASSWORD'),
            host=os.getenv('TSDB_HOST'),
            port=os.getenv('TSDB_PORT')
        )
        self.cursor = self.conn.cursor()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def _create_temp_table(self):
        """Create temporary staging table"""
        self.cursor.execute("""
            CREATE TEMPORARY TABLE IF NOT EXISTS staging_metrics (
                raw_data JSONB,
                file_name TEXT,
                ingested_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
    def _parse_timestamp(self, ts_str):
        """Handle multiple timestamp formats"""
        for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%SZ', '%m/%d/%Y %I:%M %p'):
            try:
                return datetime.strptime(ts_str, fmt)
            except ValueError:
                continue
        raise ValueError(f"Unable to parse timestamp: {ts_str}")

    async def _process_file(self, file_path):
        """Process different file formats"""
        _, ext = os.path.splitext(file_path)
        try:
            if ext == '.csv':
                return self._process_csv(file_path)
            elif ext == '.json':
                return self._process_json(file_path)
            else:
                logging.warning(f"Unsupported file format: {file_path}")
                return 0
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")
            return 0

    def _process_csv(self, file_path):
        """Process CSV files with chunking"""
        inserted = 0
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            chunk = []
            for row in reader:
                try:
                    chunk.append({
                        'timestamp': self._parse_timestamp(row['Timestamp']),
                        'facility_id': row['FacilityID'],
                        'kwh': float(row['Energy_kWh']),
                        'source': row['Source']
                    })
                    if len(chunk) >= 1000:
                        inserted += self._insert_energy_chunk(chunk)
                        chunk = []
                except KeyError as e:
                    logging.warning(f"Missing column in {file_path}: {str(e)}")
            if chunk:
                inserted += self._insert_energy_chunk(chunk)
        return inserted

    def _insert_energy_chunk(self, chunk):
        """Batch insert energy data"""
        query = """
            INSERT INTO energy_usage (timestamp, facility_id, kwh, source)
            VALUES (%(timestamp)s, %(facility_id)s, %(kwh)s, %(source)s)
            ON CONFLICT (timestamp, facility_id) DO NOTHING
        """
        try:
            execute_batch(self.cursor, query, chunk)
            self.conn.commit()
            return len(chunk)
        except psycopg2.Error as e:
            self.conn.rollback()
            logging.error(f"Insert failed: {str(e)}")
            return 0

    def run(self, data_dir):
        """Main ingestion workflow"""
        self._create_temp_table()
        total = 0
        
        for root, _, files in os.walk(data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                future = self.executor.submit(
                    self._process_file, 
                    file_path
                )
                total += future.result()
        
        logging.info(f"Ingested {total} records total")
        self.cursor.close()
        self.conn.close()

if __name__ == "__main__":
    ingestor = DataIngestor()
    ingestor.run('data/historical/')
