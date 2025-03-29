-- Find emissions anomalies using MAD
WITH stats AS (
    SELECT
        facility_id,
        median(co2_kg) AS median_co2,
        mad(co2_kg) AS mad_co2
    FROM emission_metrics
    WHERE timestamp > NOW() - INTERVAL '7 days'
    GROUP BY facility_id
)
SELECT
    e.timestamp,
    e.facility_id,
    e.co2_kg,
    ABS(e.co2_kg - s.median_co2) / s.mad_co2 AS z_score
FROM emission_metrics e
JOIN stats s USING (facility_id)
WHERE 
    ABS(e.co2_kg - s.median_co2) / s.mad_co2 > 3
    AND timestamp > NOW() - INTERVAL '1 day';
