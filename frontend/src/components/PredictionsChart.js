import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Card, Select, MenuItem, Typography } from '@material-ui/core';
import useStyles from '../styles';
import { formatDate, co2Formatter } from '../utils';

const PredictionsChart = ({ predictions }) => {
  const classes = useStyles();
  const [timeRange, setTimeRange] = useState('7d');
  const [chartData, setChartData] = useState([]);

  useEffect(() => {
    const processData = () => {
      return predictions.map(p => ({
        date: new Date(p.timestamp),
        actual: p.actual_emissions,
        predicted: p.predicted_emissions,
        variance: Math.abs(p.actual_emissions - p.predicted_emissions)
      }));
    };
    setChartData(processData());
  }, [predictions, timeRange]);

  return (
    <Card className={classes.predictionCard}>
      <div className={classes.cardHeader}>
        <Typography variant="h5" component="h2">
          Emissions Forecast vs Actual
        </Typography>
        <Select
          value={timeRange}
          onChange={(e) => setTimeRange(e.target.value)}
          className={classes.select}
        >
          <MenuItem value="24h">Last 24 Hours</MenuItem>
          <MenuItem value="7d">Last 7 Days</MenuItem>
          <MenuItem value="30d">Last 30 Days</MenuItem>
        </Select>
      </div>

      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="date"
            tickFormatter={formatDate}
            angle={-45}
            textAnchor="end"
          />
          <YAxis
            tickFormatter={co2Formatter}
            width={100}
          />
          <Tooltip 
            formatter={(value) => [co2Formatter(value), 'CO₂']}
            labelFormatter={formatDate}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="actual"
            stroke="#82ca9d"
            strokeWidth={2}
            dot={false}
            name="Actual Emissions"
          />
          <Line
            type="monotone"
            dataKey="predicted"
            stroke="#8884d8"
            strokeWidth={2}
            strokeDasharray="5 5"
            dot={false}
            name="AI Prediction"
          />
        </LineChart>
      </ResponsiveContainer>

      <div className={classes.accuracyBadge}>
        Prediction Accuracy: {calculateAccuracy(chartData)}%
      </div>
    </Card>
  );
};

const calculateAccuracy = (data) => {
  if (!data.length) return 0;
  const totalVariance = data.reduce((sum, d) => sum + d.variance, 0);
  const avgVariance = totalVariance / data.length;
  return Math.max(0, 100 - (avgVariance * 100)).toFixed(1);
};

export default PredictionsChart;
