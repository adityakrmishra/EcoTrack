#!/usr/bin/env python3
import os
import sys
import pdfkit
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from jinja2 import Environment, FileSystemLoader
import psycopg2
from dotenv import load_dotenv

load_dotenv()

class ESGReporter:
    def __init__(self, facility_id):
        self.facility_id = facility_id
        self.conn = psycopg2.connect(
            dbname=os.getenv('TSDB_NAME'),
            user=os.getenv('TSDB_USER'),
            password=os.getenv('TSDB_PASSWORD'),
            host=os.getenv('TSDB_HOST'),
            port=os.getenv('TSDB_PORT')
        )
        self.env = Environment(loader=FileSystemLoader('templates/'))
        
    def _query_metrics(self, start_date, end_date):
        """Retrieve report data from TimescaleDB"""
        query = """
            SELECT 
                time_bucket('1 day', timestamp) AS bucket,
                SUM(kwh) AS energy,
                SUM(cubic_meters) AS water,
                SUM(co2_kg) AS emissions,
                AVG(anomaly_score) AS anomaly
            FROM combined_metrics
            WHERE facility_id = %s
                AND timestamp BETWEEN %s AND %s
            GROUP BY bucket
            ORDER BY bucket
        """
        df = pd.read_sql(query, self.conn, 
                        params=(self.facility_id, start_date, end_date))
        return df

    def _generate_charts(self, df, output_dir):
        """Create visualization assets"""
        plt.style.use('ggplot')
        
        # Energy Usage Trend
        plt.figure(figsize=(10, 6))
        df.plot(x='bucket', y='energy', kind='line', title='Daily Energy Consumption')
        plt.savefig(f"{output_dir}/energy_trend.png")
        plt.close()
        
        # Emissions Distribution
        plt.figure(figsize=(10, 6))
        df['emissions'].plot(kind='hist', title='CO2 Emissions Distribution')
        plt.savefig(f"{output_dir}/emissions_hist.png")
        plt.close()
        
        return [
            'energy_trend.png',
            'emissions_hist.png'
        ]

    def _generate_pdf(self, context, output_path):
        """Render PDF report using Jinja template"""
        template = self.env.get_template('esg_report.html')
        html = template.render(context)
        pdfkit.from_string(html, output_path, options={
            'encoding': 'UTF-8',
            'quiet': ''
        })

    def generate_report(self, output_dir, period='month'):
        """Main report generation workflow"""
        end_date = datetime.now()
        start_date = end_date - timedelta(
            days=30 if period == 'month' else 7
        )
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get metrics data
        df = self._query_metrics(start_date, end_date)
        
        # Generate visualizations
        charts = self._generate_charts(df, output_dir)
        
        # Prepare report context
        context = {
            'facility_id': self.facility_id,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'total_energy': df['energy'].sum(),
            'total_water': df['water'].sum(),
            'total_emissions': df['emissions'].sum(),
            'charts': charts,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Generate outputs
        report_base = f"ESG_Report_{self.facility_id}_{end_date.date()}"
        
        # PDF Report
        self._generate_pdf(context, f"{output_dir}/{report_base}.pdf")
        
        # Excel Export
        df.to_excel(f"{output_dir}/{report_base}.xlsx", index=False)
        
        # JSON Metadata
        with open(f"{output_dir}/metadata.json", 'w') as f:
            json.dump(context, f)
            
        self.conn.close()
        return output_dir

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_reports.py <facility_id>")
        sys.exit(1)
        
    reporter = ESGReporter(sys.argv[1])
    output = reporter.generate_report('reports/')
    print(f"Generated reports in: {output}")
