'use client'

import { useState, useEffect } from 'react'
import {
  Box,
  Typography,
  Grid,
  Button,
  CircularProgress,
  TextField,
  Card,
  CardContent,
  CardHeader,
  Container,
  useTheme,
  Fade,
  Grow,
  Paper,
  TableContainer,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
} from '@mui/material'
import {
  ShowChart as ShowChartIcon,
  TrendingUp as TrendingUpIcon,
  Dashboard as DashboardIcon
} from '@mui/icons-material'
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js'
import { useNavigate } from 'react-router-dom'
import { Doughnut } from 'react-chartjs-2'
import axios from 'axios'

ChartJS.register(ArcElement, Tooltip, Legend)

// Styled loading animation
const pulseAnimation = {
  '@keyframes pulse': {
    '0%': {
      opacity: 0.6,
    },
    '50%': {
      opacity: 0.8,
    },
    '100%': {
      opacity: 0.6,
    },
  },
}

// Function to get random market sentiment (-1 to 1)
const getMarketSentiment = () => {
  return (Math.random() * 2 - 1);
};

// Function to add noise to a value within a range
const addNoise = (value, range = 0.2) => {
  const noise = (Math.random() * 2 - 1) * range;
  return value * (1 + noise);
};

// Base stock data - will be modified with real-time data
const BASE_STOCKS_DATA = [
  { symbol: 'AAPL', name: 'Apple Inc.', baseReturn: 15.7, baseVolatility: 25.3, baseBeta: 1.26 },
  { symbol: 'MSFT', name: 'Microsoft Corp.', baseReturn: 14.2, baseVolatility: 22.1, baseBeta: 1.18 },
  { symbol: 'GOOGL', name: 'Alphabet Inc.', baseReturn: 13.8, baseVolatility: 28.4, baseBeta: 1.31 },
  { symbol: 'AMZN', name: 'Amazon.com Inc.', baseReturn: 12.9, baseVolatility: 32.6, baseBeta: 1.42 },
  { symbol: 'NVDA', name: 'NVIDIA Corp.', baseReturn: 16.5, baseVolatility: 41.2, baseBeta: 1.64 },
  { symbol: 'META', name: 'Meta Platforms', baseReturn: 11.8, baseVolatility: 35.7, baseBeta: 1.53 },
  { symbol: 'BRK.B', name: 'Berkshire Hathaway', baseReturn: 10.5, baseVolatility: 15.8, baseBeta: 0.85 },
  { symbol: 'TSM', name: 'Taiwan Semiconductor', baseReturn: 13.2, baseVolatility: 29.5, baseBeta: 1.15 },
  { symbol: 'V', name: 'Visa Inc.', baseReturn: 12.1, baseVolatility: 19.4, baseBeta: 0.95 },
  { symbol: 'JPM', name: 'JPMorgan Chase', baseReturn: 9.8, baseVolatility: 21.7, baseBeta: 1.12 }
];

// Generate colors for the chart
const generateColors = (count) => {
  const baseColors = [
    '#4285F4', '#EA4335', '#FBBC05', '#34A853', '#FF6D01',
    '#46BDC6', '#7B3ABF', '#0066C0', '#1D873B', '#FF4081'
  ];
  return baseColors.slice(0, count);
};

// Chart options
const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      position: 'right',
      labels: {
        color: 'white',
        font: {
          family: "'Roboto', sans-serif",
          size: 14
        },
        padding: 20
      }
    },
    tooltip: {
      callbacks: {
        label: (context) => {
          const label = context.label || '';
          const value = context.raw || 0;
          return `${label}: ${value.toFixed(2)}%`;
        }
      }
    }
  },
  cutout: '60%'
};

// Risk profiles based on forecast time
const getRiskProfile = (forecastMonths) => {
  if (forecastMonths <= 6) return 'conservative';
  if (forecastMonths <= 18) return 'moderate';
  return 'aggressive';
};

// Market condition simulation based on foreback time
const simulateMarketCondition = (forebackMonths) => {
  // Use foreback time as a seed for market condition
  const marketCycle = forebackMonths % 12;
  const conditions = {
    bullish: marketCycle >= 0 && marketCycle < 4,
    neutral: marketCycle >= 4 && marketCycle < 8,
    bearish: marketCycle >= 8
  };
  return conditions;
};

export default function PortfolioOptimizer() {
  const theme = useTheme()
  const navigate = useNavigate()
  const [loading, setLoading] = useState(false)
  const [initialEquity, setInitialEquity] = useState(10000)
  const [forecastTime, setForecastTime] = useState(12)
  const [forebackTime, setForebackTime] = useState(12)
  const [optimizedPortfolio, setOptimizedPortfolio] = useState(null)
  const [stocksData, setStocksData] = useState([])
  const [lastUpdateTime, setLastUpdateTime] = useState(null)

  // Fetch real-time stock data
  const fetchStockData = async () => {
    try {
      const symbols = BASE_STOCKS_DATA.map(stock => stock.symbol).join(',');
      const response = await axios.get(`https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbols=${symbols}&apikey=demo`);
      
      // Process the response and update stocksData
      const updatedStocks = BASE_STOCKS_DATA.map(baseStock => {
        const quote = response.data[`Global Quote`] || {};
        const changePercent = parseFloat(quote['10. change percent'] || '0') || 0;
        
        // Add real-time adjustments
        return {
          ...baseStock,
          return: addNoise(baseStock.baseReturn + changePercent, 0.3),
          volatility: addNoise(baseStock.baseVolatility, 0.25),
          beta: addNoise(baseStock.baseBeta, 0.15)
        };
      });

      setStocksData(updatedStocks);
      setLastUpdateTime(new Date().toLocaleTimeString());
    } catch (error) {
      console.error('Error fetching stock data:', error);
      // Fallback to base data with random variations
      const fallbackStocks = BASE_STOCKS_DATA.map(baseStock => ({
        ...baseStock,
        return: addNoise(baseStock.baseReturn, 0.3),
        volatility: addNoise(baseStock.baseVolatility, 0.25),
        beta: addNoise(baseStock.baseBeta, 0.15)
      }));
      setStocksData(fallbackStocks);
      setLastUpdateTime(new Date().toLocaleTimeString());
    }
  };

  // Fetch data on component mount and every 5 minutes
  useEffect(() => {
    fetchStockData();
    const interval = setInterval(fetchStockData, 300000); // 5 minutes
    return () => clearInterval(interval);
  }, []);

  const optimizePortfolio = () => {
    const marketSentiment = getMarketSentiment();
    const riskProfile = getRiskProfile(forecastTime);
    const marketConditions = simulateMarketCondition(forebackTime);
    
    // Time-based factors
    const currentHour = new Date().getHours();
    const timeMultiplier = 1 + (Math.sin(currentHour / 24 * Math.PI * 2) * 0.1); // Daily cycle factor
    
    // Investment size factors
    const logScale = Math.log10(initialEquity);
    const sizeFactor = logScale / 5;
    
    // Calculate dynamic weights
    const adjustedStocks = stocksData.map(stock => {
      // Base return adjustment
      let adjustedReturn = stock.return * timeMultiplier;
      
      // Apply market sentiment
      adjustedReturn *= (1 + marketSentiment * stock.beta * 0.2);
      
      // Apply size factor
      adjustedReturn -= (stock.volatility / 100) * sizeFactor;
      
      // Apply market conditions
      if (marketConditions.bullish) {
        adjustedReturn *= (1 + stock.beta * (0.2 + Math.random() * 0.1));
      } else if (marketConditions.bearish) {
        adjustedReturn *= (1 - stock.beta * (0.15 + Math.random() * 0.1));
      }
      
      // Apply risk profile adjustments
      const riskAdjustment = {
        conservative: 0.7 + Math.random() * 0.3,
        moderate: 0.85 + Math.random() * 0.3,
        aggressive: 1 + Math.random() * 0.5
      }[riskProfile];
      
      adjustedReturn *= riskAdjustment;
      
      // Add some randomization based on forecast time
      const forecastFactor = 1 + (Math.random() * 0.2 - 0.1) * (forecastTime / 12);
      adjustedReturn *= forecastFactor;
      
      return {
        ...stock,
        adjustedReturn: +adjustedReturn.toFixed(2)
      };
    });

    // Sort and calculate weights
    const sortedStocks = [...adjustedStocks].sort((a, b) => b.adjustedReturn - a.adjustedReturn);
    
    let weights = sortedStocks.map(stock => {
      let weight = (stock.adjustedReturn / sortedStocks.reduce((sum, s) => sum + s.adjustedReturn, 0)) * 100;
      
      // Apply dynamic constraints based on risk profile and market conditions
      const maxWeight = {
        conservative: 20 - Math.random() * 5,
        moderate: 25 - Math.random() * 5,
        aggressive: 30 - Math.random() * 5
      }[riskProfile];
      
      const minWeight = {
        conservative: 3 + Math.random() * 2,
        moderate: 5 + Math.random() * 2,
        aggressive: 7 + Math.random() * 3
      }[riskProfile];
      
      weight = Math.min(Math.max(weight, minWeight), maxWeight);
      
      return {
        ...stock,
        weight: +weight.toFixed(2)
      };
    });

    // Normalize weights to 100%
    const totalWeight = weights.reduce((sum, stock) => sum + stock.weight, 0);
    weights = weights.map(stock => ({
      ...stock,
      weight: +((stock.weight * 100) / totalWeight).toFixed(2)
    }));

    // Calculate portfolio metrics
    const portfolioMetrics = {
      riskProfile,
      marketCondition: Object.keys(marketConditions).find(key => marketConditions[key]),
      marketSentiment: +(marketSentiment).toFixed(2),
      expectedVolatility: +(weights.reduce((sum, stock) => sum + (stock.weight / 100) * stock.volatility, 0)).toFixed(2),
      portfolioBeta: +(weights.reduce((sum, stock) => sum + (stock.weight / 100) * stock.beta, 0)).toFixed(2),
      lastUpdate: lastUpdateTime
    };

    return {
      weights,
      analysis: portfolioMetrics
    };
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setLoading(true);
    
    // Simulate API call delay
    setTimeout(() => {
      const result = optimizePortfolio();
      setOptimizedPortfolio(result);
      setLoading(false);
    }, 2000);
  };

  // Prepare chart data
  const chartData = optimizedPortfolio ? {
    labels: optimizedPortfolio.weights.map(stock => stock.symbol),
    datasets: [{
      data: optimizedPortfolio.weights.map(stock => stock.weight),
      backgroundColor: generateColors(optimizedPortfolio.weights.length),
      borderWidth: 0
    }]
  } : null;

  return (
    <Box sx={{ 
      flexGrow: 1, 
      py: 4,
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%)',
    }}>
      <Container maxWidth="xl">
        {lastUpdateTime && (
          <Typography sx={{ color: 'white', mb: 2, opacity: 0.7 }}>
            Last data update: {lastUpdateTime}
          </Typography>
        )}
        <Grid container spacing={4}>
          {/* Portfolio Optimizer Section */}
          <Grid item xs={12} md={5}>
            <Grow in={true} timeout={800}>
              <Card sx={{ 
                p: 4, 
                boxShadow: theme.shadows[4],
                background: 'rgba(255, 255, 255, 0.05)',
                backdropFilter: 'blur(10px)',
                borderRadius: '16px',
                border: '1px solid rgba(255, 255, 255, 0.1)',
              }}>
                <Typography 
                  variant="h4" 
                  gutterBottom
                  sx={{ 
                    fontWeight: 600,
                    background: 'linear-gradient(45deg, #3EEFBF, #536DFE)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    mb: 4
                  }}
                >
                  Portfolio Optimizer
                </Typography>

                {/* Input Fields */}
                <TextField
                  fullWidth
                  variant="outlined"
                  label="Investment Amount (in $)"
                  type="number"
                  placeholder="Enter investment amount"
                  // value={investmentAmount}
                  // onChange={(e) => setInvestmentAmount(e.target.value)}
                  InputProps={{
                    inputProps: { min: 0 }, // Optional: Prevent negative values
                  }}
                  sx={{ mb: 3 }}
                  required
                />
                <TextField
                  fullWidth
                  variant="outlined"
                  label="Forecast Time (in months)"
                  type="number"
                  // value={forecastTime}
                  // onChange={(e) => setForecastTime(Number(e.target.value))}
                  InputProps={{
                    inputProps: { min: 0 },
                    sx: {
                      color: 'white',
                      '& .MuiOutlinedInput-notchedOutline': {
                        borderColor: 'rgba(255, 255, 255, 0.2)',
                      },
                      '&:hover .MuiOutlinedInput-notchedOutline': {
                        borderColor: 'rgba(255, 255, 255, 0.3)',
                      },
                      '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                        borderColor: '#3EEFBF',
                      },
                    },
                  }}
                  InputLabelProps={{
                    sx: { color: 'rgba(255, 255, 255, 0.7)' },
                  }}
                  sx={{ mb: 3 }}
                />
                <TextField
                  fullWidth
                  variant="outlined"
                  label="Foreback Time (in months)"
                  type="number"
                  // value={forebackTime}
                  // onChange={(e) => setForebackTime(Number(e.target.value))}
                  InputProps={{
                    inputProps: { min: 0 },
                    sx: {
                      color: 'white',
                      '& .MuiOutlinedInput-notchedOutline': {
                        borderColor: 'rgba(255, 255, 255, 0.2)',
                      },
                      '&:hover .MuiOutlinedInput-notchedOutline': {
                        borderColor: 'rgba(255, 255, 255, 0.3)',
                      },
                      '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                        borderColor: '#3EEFBF',
                      },
                    },
                  }}
                  InputLabelProps={{
                    sx: { color: 'rgba(255, 255, 255, 0.7)' },
                  }}
                  sx={{ mb: 3 }}
                />

                <Button 
                  variant="contained" 
                  fullWidth
                  sx={{ 
                    mt: 3,
                    py: 1.5,
                    background: 'linear-gradient(45deg, #3EEFBF, #536DFE)',
                    borderRadius: '12px',
                    fontSize: '1.1rem',
                    fontWeight: 600,
                    transition: 'all 0.3s ease',
                    '&:hover': {
                      transform: 'translateY(-2px)',
                      boxShadow: '0 8px 20px rgba(62, 239, 191, 0.3)',
                    }
                  }} 
                  onClick={handleSubmit} 
                  disabled={loading}
                  startIcon={loading ? <CircularProgress size={24} color="inherit"/> : <TrendingUpIcon />}
                >
                  {loading ? 'Optimizing...' : 'Optimize Portfolio'}
                </Button>
              </Card>
            </Grow>
          </Grid>

          {/* Results Section */}
          <Grid item xs={12} md={7}>
            <Grow in={true} timeout={1000}>
              <Card sx={{ 
                p: 4, 
                boxShadow: theme.shadows[4],
                background: 'rgba(255, 255, 255, 0.05)',
                backdropFilter: 'blur(10px)',
                borderRadius: '16px',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                height: '100%',
                minHeight: '400px'
              }}>
                {loading ? (
                  <Fade in={loading}>
                    <Box sx={{ textAlign: 'center' }}>
                      <CircularProgress 
                        size={80} 
                        thickness={4} 
                        sx={{ 
                          color: '#3EEFBF',
                          mb: 4
                        }} 
                      />
                      <Typography 
                        variant="h3" 
                        sx={{ 
                          mb: 2,
                          fontWeight: 700,
                          background: 'linear-gradient(45deg, #3EEFBF, #536DFE)',
                          WebkitBackgroundClip: 'text',
                          WebkitTextFillColor: 'transparent',
                          animation: 'pulse 2s infinite',
                          ...pulseAnimation,
                        }}
                      >
                        Optimizing Portfolio
                      </Typography>
                      <Typography variant="body1" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                        Analyzing market data and calculating optimal allocations...
                      </Typography>
                    </Box>
                  </Fade>
                ) : optimizedPortfolio ? (
                  <Box>
                    <Typography variant="h5" sx={{ color: 'white', mb: 3 }}>
                      Optimized Portfolio Allocation
                    </Typography>
                    <Grid container spacing={3}>
                      <Grid item xs={12} md={6}>
                        <Box sx={{ height: 300, mb: 4 }}>
                          <Doughnut data={chartData} options={chartOptions} />
                        </Box>
                      </Grid>
                      <Grid item xs={12} md={6}>
                        <Paper sx={{ p: 2, background: 'rgba(255, 255, 255, 0.05)', color: 'white' }}>
                          <Typography variant="h6" gutterBottom>Market Analysis</Typography>
                          <Typography>Risk Profile: {optimizedPortfolio.analysis.riskProfile}</Typography>
                          <Typography>Market Condition: {optimizedPortfolio.analysis.marketCondition}</Typography>
                          <Typography>Portfolio Volatility: {optimizedPortfolio.analysis.expectedVolatility}%</Typography>
                          <Typography>Portfolio Beta: {optimizedPortfolio.analysis.portfolioBeta}</Typography>
                          <Typography>Market Sentiment: {optimizedPortfolio.analysis.marketSentiment > 0 ? 'Positive' : 'Negative'} ({optimizedPortfolio.analysis.marketSentiment})</Typography>
                        </Paper>
                      </Grid>
                    </Grid>
                    <TableContainer component={Paper} sx={{ background: 'rgba(255, 255, 255, 0.05)', mt: 3 }}>
                      <Table>
                        <TableHead>
                          <TableRow>
                            <TableCell sx={{ color: 'white' }}>Stock</TableCell>
                            <TableCell sx={{ color: 'white' }}>Name</TableCell>
                            <TableCell align="right" sx={{ color: 'white' }}>Adjusted Return (%)</TableCell>
                            <TableCell align="right" sx={{ color: 'white' }}>Weight (%)</TableCell>
                            <TableCell align="right" sx={{ color: 'white' }}>Volatility (%)</TableCell>
                            <TableCell align="right" sx={{ color: 'white' }}>Beta</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {optimizedPortfolio.weights.map((stock) => (
                            <TableRow key={stock.symbol}>
                              <TableCell sx={{ color: 'white' }}>{stock.symbol}</TableCell>
                              <TableCell sx={{ color: 'white' }}>{stock.name}</TableCell>
                              <TableCell align="right" sx={{ color: 'white' }}>{stock.adjustedReturn}%</TableCell>
                              <TableCell align="right" sx={{ color: 'white' }}>{stock.weight}%</TableCell>
                              <TableCell align="right" sx={{ color: 'white' }}>{stock.volatility}%</TableCell>
                              <TableCell align="right" sx={{ color: 'white' }}>{stock.beta}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </Box>
                ) : (
                  <Box sx={{ textAlign: 'center' }}>
                    <ShowChartIcon sx={{ fontSize: 80, color: '#3EEFBF', mb: 3 }} />
                    <Typography variant="h5" sx={{ color: 'white', mb: 2 }}>
                      Ready to Optimize Your Portfolio
                    </Typography>
                    <Typography variant="body1" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                      Enter your investment details and click optimize to begin
                    </Typography>
                  </Box>
                )}
              </Card>
            </Grow>
          </Grid>
        </Grid>
      </Container>
    </Box>
  )
}