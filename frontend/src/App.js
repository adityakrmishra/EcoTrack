import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Switch, Route } from 'react-router-dom';
import { ThemeProvider, createMuiTheme, CssBaseline } from '@material-ui/core';
import { LiveMetrics, PredictionsChart } from './components';
import { Sidebar, AlertBar } from './layout';
import useStyles from './styles';

const App = () => {
  const classes = useStyles();
  const [darkMode, setDarkMode] = useState(false);
  const [facilityData, setFacilityData] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [alerts, setAlerts] = useState([]);

  const theme = createMuiTheme({
    palette: {
      type: darkMode ? 'dark' : 'light',
      primary: {
        main: '#4CAF50',
      },
      secondary: {
        main: '#FFC107',
      },
    },
  });

  useEffect(() => {
    const loadInitialData = async () => {
      try {
        const [facilityRes, predictionRes] = await Promise.all([
          axios.get('/api/facility/current'),
          axios.get('/api/predictions')
        ]);
        setFacilityData(facilityRes.data);
        setPredictions(predictionRes.data);
      } catch (error) {
        setAlerts([...alerts, {
          message: 'Failed to load initial data',
          severity: 'error'
        }]);
      }
    };
    loadInitialData();
  }, []);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <div className={classes.root}>
          <Sidebar darkMode={darkMode} setDarkMode={setDarkMode} />
          
          <main className={classes.content}>
            <AlertBar alerts={alerts} />
            
            <Switch>
              <Route exact path="/">
                {facilityData && (
                  <div className={classes.dashboard}>
                    <LiveMetrics facilityId={facilityData.id} />
                    <PredictionsChart predictions={predictions} />
                  </div>
                )}
              </Route>
            </Switch>
          </main>
        </div>
      </Router>
    </ThemeProvider>
  );
};

export default App;
