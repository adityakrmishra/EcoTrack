import React, { useState, useEffect } from 'react';
import { VictoryLine, VictoryChart, VictoryTheme, VictoryAxis, VictoryTooltip, VictoryVoronoiContainer } from 'victory';
import { Card, Progress, Switch } from '@material-ui/core';
import axios from 'axios';
import useStyles from '../styles';
import { formatCO2, useInterval } from '../utils';

const LiveMetrics = ({ facilityId }) => {
  const classes = useStyles();
  const [metrics, setMetrics] = useState({ 
    energy: [], 
    water: [], 
    co2: [] 
  });
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [loading, setLoading] = useState(true);

  const fetchMetrics = async () => {
    try {
      const { data } = await axios.get(`/api/live-metrics/${facilityId}`);
      setMetrics({
        energy: [...metrics.energy.slice(-29), data.energy],
        water: [...metrics.water.slice(-29), data.water],
        co2: [...metrics.co2.slice(-29), data.co2]
      });
      setLoading(false);
    } catch (error) {
      console.error('Error fetching metrics:', error);
    }
  };

  useInterval(() => {
    if (autoRefresh) fetchMetrics();
  }, 5000);

  return (
    <Card className={classes.metricCard}>
      <div className={classes.cardHeader}>
        <h2>Real-time Monitoring</h2>
        <Switch 
          checked={autoRefresh}
          onChange={() => setAutoRefresh(!autoRefresh)}
          color="primary"
        />
      </div>
      
      {!loading ? (
        <div className={classes.chartContainer}>
          <VictoryChart
            theme={VictoryTheme.material}
            containerComponent={<VictoryVoronoiContainer />}
            height={300}
          >
            <VictoryAxis
              tickFormat={(t) => new Date(t).toLocaleTimeString()}
              style={{ tickLabels: { angle: -45 } }}
            />
            <VictoryAxis dependentAxis />
            
            <VictoryLine
              data={metrics.co2}
              x="timestamp"
              y="value"
              style={{ data: { stroke: "#ff4081" } }}
              labels={({ datum }) => `CO₂: ${formatCO2(datum.value)}`}
              labelComponent={<VictoryTooltip />}
            />
          </VictoryChart>

          <div className={classes.kpiContainer}>
            <div className={classes.kpi}>
              <h3>Energy Usage</h3>
              <Progress 
                variant="determinate" 
                value={metrics.energy.slice(-1)[0]?.value || 0}
              />
              <span>{metrics.energy.slice(-1)[0]?.value.toFixed(2)} kWh</span>
            </div>
            <div className={classes.kpi}>
              <h3>Water Consumption</h3>
              <Progress 
                variant="determinate" 
                value={metrics.water.slice(-1)[0]?.value || 0}
              />
              <span>{metrics.water.slice(-1)[0]?.value.toFixed(2)} m³</span>
            </div>
          </div>
        </div>
      ) : (
        <div className={classes.loading}>Initializing real-time data...</div>
      )}
    </Card>
  );
};

export default LiveMetrics;
