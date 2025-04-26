'use client'

import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  CardHeader,
  ListItemAvatar,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Button,
  Avatar,
  Container,
  useTheme,
  TextField,
  InputAdornment,
  IconButton,
  Chip,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  CircularProgress,
  LinearProgress,
  Tooltip
} from '@mui/material';
import {
  ArrowUpward as ArrowUpwardIcon,
  ArrowDownward as ArrowDownwardIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Assessment as AssessmentIcon,
  ShowChart as ShowChartIcon,
  Notifications as NotificationsIcon,
  AccountBalance as AccountBalanceIcon,
  MonetizationOn as MonetizationOnIcon,
  PieChart as PieChartIcon,
  Timeline as TimelineIcon,
  Diamond as DiamondIcon,
  CurrencyExchange as CurrencyExchangeIcon,
  Paid as PaidIcon,
  Person as PersonIcon,
  Email as EmailIcon,
  Phone as PhoneIcon,
  LocationOn as LocationIcon,
  CalendarToday as CalendarIcon,
  Save as SaveIcon,
  Cancel as CancelIcon,
  Star as StarIcon,
  StarOutline as StarOutlineIcon,
  Search as SearchIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  ErrorOutline as ErrorOutlineIcon,
  Lightbulb as LightbulbIcon,
  Article as ArticleIcon,
  Language as LanguageIcon,
  Speed as SpeedIcon,
  BarChart as BarChartIcon,
  Refresh as RefreshIcon,
  Analytics as AnalyticsIcon
} from '@mui/icons-material';
import { Chart as ChartJS, ArcElement, Tooltip as ChartTooltip, Legend, CategoryScale, LinearScale, PointElement, LineElement, BarElement } from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';

// Register ChartJS components
ChartJS.register(
  ArcElement, 
  ChartTooltip, 
  Legend, 
  CategoryScale, 
  LinearScale, 
  PointElement, 
  LineElement,
  BarElement
);

// API Configuration
const ALPHA_VANTAGE_API_KEY = 'XCEQN4K5LDG5OIAN';
const FINNHUB_API_KEY = 'cjvlb89r01qjv0um2q3gcjvlb89r01qjv0um2q40';

// Market indices configuration
const MARKET_INDICES = {
  SPY: { name: 'S&P 500', id: 1, description: 'Large-cap U.S. stocks' },
  DIA: { name: 'Dow Jones', id: 2, description: '30 major U.S. companies' },
  QQQ: { name: 'Nasdaq 100', id: 3, description: 'Tech-heavy index' },
  IWM: { name: 'Russell 2000', id: 4, description: 'Small-cap U.S. stocks' },
  VGK: { name: 'European Stocks', id: 5, description: 'European market index' },
  EEM: { name: 'Emerging Markets', id: 6, description: 'Emerging markets index' }
};

// Top gainers and losers configuration
const TRENDING_STOCKS = [
  'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'WMT',
  'PG', 'JNJ', 'XOM', 'BAC', 'HD', 'PFE', 'DIS', 'NFLX', 'ADBE', 'CRM'
];

// Chart options
const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      display: true,
      position: 'top',
    },
    tooltip: {
      mode: 'index',
      intersect: false,
    }
  },
  scales: {
    y: {
      ticks: {
        callback: (value) => value.toLocaleString(),
      },
    },
  },
  elements: {
    point: {
      radius: 2,
    },
    line: {
      tension: 0.4
    }
  }
};

const volumeChartOptions = {
  ...chartOptions,
  scales: {
    ...chartOptions.scales,
    y: {
      ...chartOptions.scales.y,
      title: {
        display: true,
        text: 'Volume (Millions)'
      }
    }
  }
};

// Fetch function for stock data using Finnhub
const fetchStockDataFinnhub = async (symbol) => {
  try {
    const response = await fetch(
      `https://finnhub.io/api/v1/quote?symbol=${symbol}&token=${FINNHUB_API_KEY}`
    );
    const data = await response.json();
    
    if (data && data.c) {
      return {
        symbol,
        price: data.c,
        change: data.d,
        changePercent: data.dp,
        high: data.h,
        low: data.l,
        volume: data.v
      };
    }
    throw new Error('Invalid data received from Finnhub API');
  } catch (error) {
    console.error(`Error fetching Finnhub data for ${symbol}:`, error);
    return null;
  }
};

// Backup fetch function using Alpha Vantage
const fetchStockDataAlphaVantage = async (symbol) => {
  try {
    const response = await fetch(
      `https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=${symbol}&apikey=${ALPHA_VANTAGE_API_KEY}`
    );
    const data = await response.json();
    
    if (data['Global Quote']) {
      const quote = data['Global Quote'];
      return {
        symbol,
        price: parseFloat(quote['05. price']),
        change: parseFloat(quote['09. change']),
        changePercent: parseFloat(quote['10. change percent'].replace('%', '')),
        volume: parseInt(quote['06. volume']),
        high: parseFloat(quote['03. high']),
        low: parseFloat(quote['04. low'])
      };
    }
    throw new Error('Invalid data received from Alpha Vantage API');
  } catch (error) {
    console.error(`Error fetching Alpha Vantage data for ${symbol}:`, error);
    return null;
  }
};

// Combined fetch function that tries multiple APIs
const fetchStockData = async (symbol) => {
  try {
    // Try Finnhub first
    let result = await fetchStockDataFinnhub(symbol);
    
    // If Finnhub fails, try Alpha Vantage
    if (!result) {
      result = await fetchStockDataAlphaVantage(symbol);
    }
    
    // If both fail, use fallback mock data for demo
    if (!result) {
      result = {
        symbol,
        price: Math.random() * 1000 + 100,
        change: Math.random() * 10 - 5,
        changePercent: Math.random() * 4 - 2,
        volume: Math.floor(Math.random() * 1000000),
        high: 0,
        low: 0
      };
      result.high = result.price + Math.random() * 10;
      result.low = result.price - Math.random() * 10;
      console.log('Using fallback data for', symbol);
    }
    
    return result;
  } catch (error) {
    console.error(`Error in combined fetch for ${symbol}:`, error);
    return null;
  }
};

// Fetch historical data with fallback
const fetchHistoricalData = async (symbol) => {
  try {
    // Try Alpha Vantage first
    const response = await fetch(
      `https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=${symbol}&apikey=${ALPHA_VANTAGE_API_KEY}`
    );
    const data = await response.json();
    
    if (data['Time Series (Daily)']) {
      const dailyData = data['Time Series (Daily)'];
      const labels = [];
      const prices = [];
      const volumes = [];
      
      Object.entries(dailyData)
        .slice(0, 30)
        .reverse()
        .forEach(([date, values]) => {
          labels.push(new Date(date).toLocaleDateString());
          prices.push(parseFloat(values['4. close']));
          volumes.push(parseInt(values['5. volume']) / 1000000);
        });

      return { labels, prices, volumes };
    }

    // If API fails, generate mock data for demo
    const mockData = {
      labels: [],
      prices: [],
      volumes: []
    };

    const basePrice = 100 + Math.random() * 100;
    const baseVolume = 1000000;
    
    for (let i = 30; i > 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      mockData.labels.push(date.toLocaleDateString());
      mockData.prices.push(basePrice + Math.random() * 20 - 10);
      mockData.volumes.push((baseVolume + Math.random() * baseVolume) / 1000000);
    }

    console.log('Using mock historical data for', symbol);
    return mockData;
  } catch (error) {
    console.error(`Error fetching historical data for ${symbol}:`, error);
    return null;
  }
};

export default function MarketInsights() {
  const theme = useTheme();
  const [marketIndices, setMarketIndices] = useState([]);
  const [trendingStocks, setTrendingStocks] = useState([]);
  const [chartData, setChartData] = useState(null);
  const [volumeData, setVolumeData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [nextUpdate, setNextUpdate] = useState(3600);

  // Format time remaining for display
  const formatTimeRemaining = (seconds) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const remainingSeconds = seconds % 60;
    
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    if (minutes > 0) {
      return `${minutes}m ${remainingSeconds}s`;
    }
    return `${remainingSeconds}s`;
  };

  // Update countdown timer effect
  useEffect(() => {
    let timer;
    if (!isLoading && !error) {
      timer = setInterval(() => {
        setNextUpdate((prev) => {
          if (prev <= 1) {
            return 3600; // Reset to 1 hour when timer reaches 0
          }
          return prev - 1;
        });
      }, 1000);
    }
    return () => clearInterval(timer);
  }, [isLoading, error]);

  // Add error reporting
  const reportError = (error, context) => {
    console.error(`Error in ${context}:`, error);
    setError(`Failed to fetch ${context}. ${error.message}`);
  };

  // Modified fetch market data function with better error handling
  useEffect(() => {
    const fetchAllMarketData = async () => {
      setIsLoading(true);
      setError(null);
      
      try {
        // Fetch market indices data
        const indicesResults = [];
        for (const [symbol, details] of Object.entries(MARKET_INDICES)) {
          const result = await fetchStockData(symbol);
          if (result) {
            indicesResults.push({
              ...details,
              ...result,
              volatility: ((result.high - result.low) / result.price * 100).toFixed(2),
              trend: result.changePercent >= 0 ? 'uptrend' : 'downtrend'
            });
          }
          // Reduced delay to 500ms to speed up loading
          await new Promise(resolve => setTimeout(resolve, 500));
        }
        setMarketIndices(indicesResults);

        // Fetch trending stocks data
        const stockResults = [];
        for (const symbol of TRENDING_STOCKS.slice(0, 10)) { // Limit to 10 stocks for faster loading
          const result = await fetchStockData(symbol);
          if (result) {
            stockResults.push({
              ...result,
              volatility: ((result.high - result.low) / result.price * 100).toFixed(2)
            });
          }
          await new Promise(resolve => setTimeout(resolve, 500));
        }
        
        stockResults.sort((a, b) => Math.abs(b.changePercent) - Math.abs(a.changePercent));
        setTrendingStocks(stockResults);

        // Fetch historical data
        const historicalData = await fetchHistoricalData('SPY');
        if (historicalData) {
          setChartData({
            labels: historicalData.labels,
            datasets: [{
              label: 'S&P 500 Price',
              data: historicalData.prices,
              borderColor: theme.palette.primary.main,
              backgroundColor: `${theme.palette.primary.main}20`,
              fill: true
            }]
          });

          setVolumeData({
            labels: historicalData.labels,
            datasets: [{
              label: 'Trading Volume',
              data: historicalData.volumes,
              backgroundColor: theme.palette.secondary.main,
              type: 'bar'
            }]
          });
        }

        setLastUpdated(new Date().toLocaleTimeString());
        setNextUpdate(3600); // Reset to 1 hour after successful fetch
      } catch (error) {
        reportError(error, 'market data');
      } finally {
        setIsLoading(false);
      }
    };

    fetchAllMarketData();
    // Update interval to 1 hour (3600000 milliseconds)
    const interval = setInterval(fetchAllMarketData, 3600000);
    return () => clearInterval(interval);
  }, [theme.palette.primary.main, theme.palette.secondary.main]);

  return (
    <Box sx={{
      flexGrow: 1,
      background: theme.palette.mode === 'dark' 
        ? 'linear-gradient(45deg, #1a237e 30%, #311b92 90%)'
        : 'linear-gradient(45deg, #e3f2fd 30%, #bbdefb 90%)',
      minHeight: '100vh',
      py: 4
    }}>
      <Container maxWidth="xl">
        <Grid container spacing={3}>
          {/* Header */}
          <Grid item xs={12}>
            <Paper
              elevation={3}
              sx={{
                p: 3,
                background: theme.palette.mode === 'dark'
                  ? 'linear-gradient(45deg, #303f9f 30%, #512da8 90%)'
                  : 'linear-gradient(45deg, #42a5f5 30%, #1976d2 90%)',
                color: 'white',
                borderRadius: 2
              }}
            >
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Box>
                  <Typography variant="h4" gutterBottom sx={{ fontWeight: 600 }}>
                    Market Insights Dashboard
                  </Typography>
                  <Typography variant="body1">
                    Real-time market data and analysis
                  </Typography>
                  {lastUpdated && (
                    <Typography variant="caption" sx={{ display: 'block', mt: 1 }}>
                      Last updated: {lastUpdated} (Next update in {formatTimeRemaining(nextUpdate)})
                    </Typography>
                  )}
                </Box>
                <Button 
                  variant="contained"
                  onClick={() => {
                    window.location.reload();
                    setNextUpdate(3600);
                  }}
                  startIcon={<RefreshIcon />}
                  sx={{ 
                    bgcolor: 'white',
                    color: theme.palette.primary.main,
                    '&:hover': {
                      bgcolor: 'rgba(255, 255, 255, 0.9)'
                    }
                  }}
                >
                  Refresh
                </Button>
              </Box>
            </Paper>
          </Grid>

          {/* Market Indices */}
          <Grid item xs={12}>
            <Card elevation={3}>
              <CardHeader 
                title="Global Market Indices"
                avatar={<ShowChartIcon />}
                action={
                  <Chip 
                    label={isLoading ? 'Updating...' : 'Live'}
                    color={isLoading ? 'warning' : 'success'}
                    size="small"
                  />
                }
              />
              <Divider />
              <CardContent>
                {isLoading ? (
                  <Box sx={{ width: '100%' }}>
                    <LinearProgress />
                  </Box>
                ) : error ? (
                  <Typography color="error" align="center">{error}</Typography>
                ) : (
                  <Grid container spacing={2}>
                    {marketIndices.map((index) => (
                      <Grid item xs={12} sm={6} md={4} lg={2} key={index.id}>
                        <Paper 
                          elevation={2}
                          sx={{ 
                            p: 2,
                            background: index.changePercent >= 0 
                              ? 'linear-gradient(45deg, #4caf50 30%, #81c784 90%)'
                              : 'linear-gradient(45deg, #f44336 30%, #e57373 90%)',
                            color: 'white'
                          }}
                        >
                          <Tooltip title={index.description} arrow>
                            <Typography variant="subtitle2" sx={{ mb: 1 }}>
                              {index.name}
                            </Typography>
                          </Tooltip>
                          <Typography variant="h5" sx={{ fontWeight: 'medium', mb: 1 }}>
                            ${index.price.toLocaleString()}
                          </Typography>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <Typography variant="body2" sx={{ display: 'flex', alignItems: 'center' }}>
                              {index.changePercent >= 0 ? 
                                <ArrowUpwardIcon fontSize="small" sx={{ mr: 0.5 }} /> : 
                                <ArrowDownwardIcon fontSize="small" sx={{ mr: 0.5 }} />
                              }
                              {index.changePercent.toFixed(2)}%
                            </Typography>
                            <Chip 
                              label={`Vol: ${index.volatility}%`}
                              size="small"
                              sx={{ 
                                bgcolor: 'rgba(255, 255, 255, 0.2)',
                                color: 'white',
                                fontSize: '0.75rem'
                              }}
                            />
                          </Box>
                        </Paper>
                      </Grid>
                    ))}
                  </Grid>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Charts */}
          <Grid item xs={12} md={8}>
            <Card elevation={3}>
              <CardHeader 
                title="S&P 500 30-Day Performance"
                avatar={<TimelineIcon />}
              />
              <Divider />
              <CardContent sx={{ height: 400 }}>
                {isLoading ? (
                  <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                    <CircularProgress />
                  </Box>
                ) : error ? (
                  <Typography color="error" align="center">{error}</Typography>
                ) : chartData ? (
                  <Line data={chartData} options={chartOptions} />
                ) : (
                  <Typography align="center">No data available</Typography>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Trading Volume */}
          <Grid item xs={12} md={4}>
            <Card elevation={3}>
              <CardHeader 
                title="Trading Volume Analysis"
                avatar={<BarChartIcon />}
              />
              <Divider />
              <CardContent sx={{ height: 400 }}>
                {isLoading ? (
                  <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                    <CircularProgress />
                  </Box>
                ) : error ? (
                  <Typography color="error" align="center">{error}</Typography>
                ) : volumeData ? (
                  <Bar data={volumeData} options={volumeChartOptions} />
                ) : (
                  <Typography align="center">No data available</Typography>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Top Movers */}
          <Grid item xs={12}>
            <Card elevation={3}>
              <CardHeader 
                title="Top Market Movers"
                avatar={<SpeedIcon />}
                subheader="Stocks with highest price movement"
              />
              <Divider />
              <CardContent>
                {isLoading ? (
                  <Box sx={{ width: '100%' }}>
                    <LinearProgress />
                  </Box>
                ) : error ? (
                  <Typography color="error" align="center">{error}</Typography>
                ) : (
                  <Grid container spacing={2}>
                    {trendingStocks.map((stock) => (
                      <Grid item xs={12} sm={6} md={4} lg={3} key={stock.symbol}>
                        <Paper 
                          elevation={2}
                          sx={{ 
                            p: 2,
                            bgcolor: 'background.paper',
                            border: 1,
                            borderColor: stock.changePercent >= 0 ? 'success.main' : 'error.main',
                            borderRadius: 1
                          }}
                        >
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                            <Typography variant="h6">{stock.symbol}</Typography>
                            <Chip 
                              icon={stock.changePercent >= 0 ? <TrendingUpIcon /> : <TrendingDownIcon />}
                              label={`${stock.changePercent >= 0 ? '+' : ''}${stock.changePercent.toFixed(2)}%`}
                              color={stock.changePercent >= 0 ? 'success' : 'error'}
                              size="small"
                            />
                          </Box>
                          <Typography variant="h5" sx={{ mb: 1 }}>
                            ${stock.price.toFixed(2)}
                          </Typography>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <Typography variant="body2" color="text.secondary">
                              Volume: {(stock.volume / 1000000).toFixed(1)}M
                            </Typography>
                            <Tooltip title="Daily Volatility" arrow>
                              <Chip 
                                label={`Vol: ${stock.volatility}%`}
                                size="small"
                                color="primary"
                                variant="outlined"
                              />
                            </Tooltip>
                          </Box>
                        </Paper>
                      </Grid>
                    ))}
                  </Grid>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
}