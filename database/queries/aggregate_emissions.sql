-- Daily emissions aggregation
SELECT 
    time_bucket('1 day', timestamp) AS bucket,
    facility_id,
    SUM(co2_kg) AS total_co2,
    AVG(anomaly_score) AS avg_anomaly
FROM emission_metrics
WHERE 
    timestamp > NOW() - INTERVAL '30 days'
GROUP BY bucket, facility_id
ORDER BY bucket DESC;
