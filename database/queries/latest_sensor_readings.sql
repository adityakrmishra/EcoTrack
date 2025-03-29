-- Last reading from each sensor type
WITH energy AS (
    SELECT 
        facility_id,
        kwh,
        timestamp
    FROM energy_usage
    ORDER BY timestamp DESC
    LIMIT 1
),
water AS (
    SELECT 
        facility_id,
        cubic_meters,
        timestamp
    FROM water_usage
    ORDER BY timestamp DESC
    LIMIT 1
)
SELECT 
    e.facility_id,
    e.kwh AS last_energy,
    w.cubic_meters AS last_water,
    GREATEST(e.timestamp, w.timestamp) AS last_reading
FROM energy e
JOIN water w USING (facility_id);
