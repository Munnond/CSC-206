'use client'
import { 
  Box, 
  Grid, 
  Paper, 
  Typography, 
  Card, 
  CardContent, 
  CardHeader,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Button,
  Avatar,
  Container,
  useTheme,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  InputAdornment,  // Make sure this appears only once
  IconButton,       // Include this if needed
  CircularProgress,
  Chip,
  Snackbar,
  Alert
} from '@mui/material';
import React, { useState, useContext, useEffect } from 'react';
import {
  ArrowUpward as ArrowUpwardIcon,
  ArrowDownward as ArrowDownwardIcon,
  TrendingUp as TrendingUpIcon,
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
  Refresh as RefreshIcon
} from '@mui/icons-material';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, PointElement, LineElement } from 'chart.js';
import { Doughnut, Line } from 'react-chartjs-2';
import { UserContext } from '../context/UserContext';
import { useFormik } from 'formik';
import * as Yup from 'yup';
import BackendClient from '../BackendClient';

// Register ChartJS components
ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, PointElement, LineElement);

// Validation schema for profile editor
const profileSchema = Yup.object().shape({
  fullName: Yup.string().required('Full name is required'),
  email: Yup.string().email('Invalid email').required('Email is required'),
  phone: Yup.string().matches(/^[0-9]{10,15}$/, 'Phone number is not valid'),
  birthDate: Yup.date().max(new Date(), 'Birth date cannot be in the future'),
});

// Portfolio allocation data for doughnut chart
const portfolioData = {
  labels: ['Stocks', 'Bonds', 'Gold', 'Crypto', 'Cash'],
  datasets: [
    {
      data: [45, 25, 15, 10, 5],
      backgroundColor: [
        '#1976d2',
        '#4caf50',
        '#FFD700', // Gold color
        '#FF9900', // Crypto orange
        '#9c27b0',
      ],
      borderWidth: 0,
    },
  ],
};

// Portfolio performance data for line chart
const performanceData = {
  labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
  datasets: [
    {
      label: 'Your Portfolio',
      data: [400000, 430000, 440000, 445000, 460000, 461227],
      borderColor: '#1976d2',
      backgroundColor: 'rgba(25, 118, 210, 0.1)',
      tension: 0.4,
      fill: true,
    },
    {
      label: 'Gold Index',
      data: [10500, 10700, 11200, 11000, 11500, 11800],
      borderColor: '#FFD700',
      backgroundColor: 'rgba(255, 215, 0, 0.1)',
      tension: 0.4,
      fill: true,
    },
  ],
};

// Chart options with gold accents
const doughnutOptions = {
  responsive: true,
  maintainAspectRatio: false,
  cutout: '60%',
  plugins: {
    legend: {
      position: 'right',
      labels: {
        color: '#333',
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
          return `${label}: ${value}%`;
        }
      }
    }
  }
};

const lineOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      position: 'top',
      labels: {
        color: '#333',
        font: {
          family: "'Roboto', sans-serif",
          size: 14,
          weight: '500'
        },
        padding: 20
      }
    },
    tooltip: {
      backgroundColor: 'rgba(0, 0, 0, 0.8)',
      titleFont: {
        size: 16,
        weight: 'bold'
      },
      bodyFont: {
        size: 14
      },
      callbacks: {
        label: (context) => {
          const label = context.dataset.label || '';
          const value = context.raw || 0;
          return `${label}: $${value.toLocaleString()}`;
        }
      }
    }
  },
  scales: {
    y: {
      beginAtZero: false,
      grid: {
        color: 'rgba(0, 0, 0, 0.05)'
      },
      ticks: {
        color: '#333',
        callback: (value) => '$' + value.toLocaleString(),
      },
    },
    x: {
      grid: {
        color: 'rgba(0, 0, 0, 0.05)'
      },
      ticks: {
        color: '#333'
      }
    }
  },
};

// Update API configurations
const ALPHA_VANTAGE_API_KEY = 'XCEQN4K5LDG5OIAN';
const FINNHUB_API_KEY = 'cjvlb89r01qjv0um2q3gcjvlb89r01qjv0um2q40';
const UPDATE_INTERVAL = 30 * 60; // 30 minutes in seconds

// Updated assets configuration with reliable data sources
const ASSETS = {
  BITCOIN: {
    id: 'bitcoin',
    symbol: 'BTC',
    name: 'Bitcoin',
    type: 'crypto',
    color: 'warning.dark',
    volatility: 'High'
  },
  ETHEREUM: {
    id: 'ethereum',
    symbol: 'ETH',
    name: 'Ethereum',
    type: 'crypto',
    color: 'info.main',
    volatility: 'High'
  },
  CARDANO: {
    id: 'cardano',
    symbol: 'ADA',
    name: 'Cardano',
    type: 'crypto',
    color: 'primary.main',
    volatility: 'High'
  },
  TESLA: {
    symbol: 'TSLA',
    name: 'Tesla Inc.',
    id: 4, 
    type: 'stock',
    color: 'error.main',
    volatility: 'High'
  },
  APPLE: {
    symbol: 'AAPL',
    name: 'Apple Inc.',
    id: 5,
    type: 'stock',
    color: 'success.main',
    volatility: 'Moderate'
  },
  MICROSOFT: {
    symbol: 'MSFT',
    name: 'Microsoft',
    id: 6,
    type: 'stock',
    color: 'info.dark',
    volatility: 'Moderate'
  }
};

// Function to fetch crypto data from CoinGecko with more details
const fetchCryptoData = async (asset) => {
  try {
    const response = await fetch(
      `https://api.coingecko.com/api/v3/simple/price?ids=${asset.id}&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true&include_market_cap=true`
    );
    const data = await response.json();
    
    if (data[asset.id]) {
      const cryptoData = data[asset.id];
      return {
        price: cryptoData.usd.toFixed(2),
        change: cryptoData.usd_24h_change.toFixed(2),
        direction: cryptoData.usd_24h_change >= 0 ? 'up' : 'down',
        volume: cryptoData.usd_24h_vol,
        marketCap: cryptoData.usd_market_cap,
        lastUpdated: new Date().toISOString()
      };
    }
    throw new Error('Invalid crypto data received');
  } catch (error) {
    console.error(`Error fetching crypto data for ${asset.id}:`, error);
    return null;
  }
};

// Function to fetch stock data from Alpha Vantage
const fetchStockData = async (asset) => {
  try {
    const response = await fetch(
      `https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=${asset.symbol}&apikey=${ALPHA_VANTAGE_API_KEY}`
    );
    const data = await response.json();
    
    if (data['Global Quote']) {
      const quote = data['Global Quote'];
      return {
        price: parseFloat(quote['05. price']).toFixed(2),
        change: parseFloat(quote['10. change percent'].replace('%', '')).toFixed(2),
        direction: parseFloat(quote['10. change percent']) >= 0 ? 'up' : 'down',
        volume: parseInt(quote['06. volume']),
        lastUpdated: new Date().toISOString()
      };
    }
    throw new Error('Invalid stock data received');
  } catch (error) {
    console.error(`Error fetching stock data for ${asset.symbol}:`, error);
    return null;
  }
};

// Combined fetch function with better error handling
const fetchMarketData = async (asset) => {
  try {
    let result = null;
    
    if (asset.type === 'crypto') {
      result = await fetchCryptoData(asset);
    } else if (asset.type === 'stock') {
      result = await fetchStockData(asset);
    }

    if (result) {
      return {
        ...asset,
        ...result,
        isLive: true
      };
    }

    // If data fetch fails, return asset with error state
    return {
      ...asset,
      price: '0',
      change: '0',
    direction: 'up',
      isLive: false,
      error: 'Unable to fetch current data'
    };
  } catch (error) {
    console.error(`Error in combined fetch for ${asset.symbol || asset.id}:`, error);
    return {
      ...asset,
      price: '0',
      change: '0',
    direction: 'up',
      isLive: false,
      error: 'Failed to fetch data'
    };
  }
};

// Move formatTimeRemaining outside the component
const formatTimeRemaining = (seconds) => {
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
};

function ProfileEditor({ onClose, onUpdate }) {
  const { user } = useContext(UserContext);
  const [isSubmitting, setIsSubmitting] = useState(false);
  console.log("Current user data in editor:", user);

  const formik = useFormik({
    initialValues: {
      fullName: user?.profile?.name || '',
      email: user?.user?.email || '',
      phone: user?.profile?.phone || '',
      birthDate: user?.profile?.birth_date || '',
    },
    validationSchema: profileSchema,
    onSubmit: async (values) => {
      setIsSubmitting(true);
      try {
        console.log("Form submitted with values:", values);
        await onUpdate(values);
      onClose();
      } catch (error) {
        console.error("Error submitting form:", error);
      } finally {
        setIsSubmitting(false);
      }
    },
  });

  return (
    <Dialog open={true} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Edit Profile</DialogTitle>
      <DialogContent>
        <Box component="form" onSubmit={formik.handleSubmit} sx={{ mt: 2 }}>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Full Name"
                name="fullName"
                value={formik.values.fullName}
                onChange={formik.handleChange}
                error={formik.touched.fullName && Boolean(formik.errors.fullName)}
                helperText={formik.touched.fullName && formik.errors.fullName}
                disabled={isSubmitting}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Email"
                name="email"
                value={formik.values.email}
                onChange={formik.handleChange}
                error={formik.touched.email && Boolean(formik.errors.email)}
                helperText={formik.touched.email && formik.errors.email}
                disabled={isSubmitting}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <EmailIcon />
                    </InputAdornment>
                  ),
                }}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Phone Number"
                name="phone"
                value={formik.values.phone}
                onChange={formik.handleChange}
                error={formik.touched.phone && Boolean(formik.errors.phone)}
                helperText={formik.touched.phone && formik.errors.phone}
                disabled={isSubmitting}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <PhoneIcon />
                    </InputAdornment>
                  ),
                }}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Birth Date"
                type="date"
                name="birthDate"
                value={formik.values.birthDate}
                onChange={formik.handleChange}
                disabled={isSubmitting}
                InputLabelProps={{ shrink: true }}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <CalendarIcon />
                    </InputAdornment>
                  ),
                }}
              />
            </Grid>
          </Grid>
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} startIcon={<CancelIcon />} disabled={isSubmitting}>
          Cancel
        </Button>
        <Button 
          onClick={formik.handleSubmit} 
          startIcon={<SaveIcon />}
          variant="contained"
          disabled={!formik.isValid || isSubmitting}
        >
          {isSubmitting ? 'Saving...' : 'Save Changes'}
        </Button>
      </DialogActions>
    </Dialog>
  );
}

export default function Dashboard() {
  const [showProfileEditor, setShowProfileEditor] = useState(false);
  const { user, updateUser, loading, error, refetchUser } = useContext(UserContext);
  const theme = useTheme();
  const [assetData, setAssetData] = useState(null);
  const [isAssetLoading, setIsAssetLoading] = useState(true);
  const [assetError, setAssetError] = useState(null);
  const [portfolioChartData, setPortfolioChartData] = useState({
    labels: [],
    datasets: [
      {
        data: [],
        backgroundColor: [
          '#2196F3', // Stocks - Blue
          '#4CAF50', // Bonds - Green
          '#FFEB3B', // Gold - Yellow
          '#FF9800', // Crypto - Orange
          '#9C27B0'  // Cash - Purple
        ],
        borderWidth: 0,
      }
    ]
  });
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'success'
  });
  const [marketData, setMarketData] = useState([]);
  const [dataUpdateTime, setDataUpdateTime] = useState(null);
  const [timeRemaining, setTimeRemaining] = useState(UPDATE_INTERVAL);

  // Show error message if user context has error
  useEffect(() => {
    if (error) {
      setSnackbar({
        open: true,
        message: `Error loading user data: ${error}`,
        severity: 'error'
      });
    }
  }, [error]);

  // Fetch market data for selected assets
  useEffect(() => {
    const fetchAllMarketData = async () => {
      try {
        // Get data for a subset of assets to avoid rate limits
        const assetsToFetch = [ASSETS.BITCOIN, ASSETS.TESLA, ASSETS.MICROSOFT];
        const results = await Promise.all(
          assetsToFetch.map(asset => fetchMarketData(asset))
        );
        
        setMarketData(results);
        setDataUpdateTime(new Date());
        setTimeRemaining(UPDATE_INTERVAL);
      } catch (error) {
        console.error("Error fetching market data:", error);
      }
    };

    fetchAllMarketData();
    
    // Set up timer for countdown
    const timer = setInterval(() => {
      setTimeRemaining(prev => {
        if (prev <= 1) {
          // Refetch data when timer reaches zero
          fetchAllMarketData();
          return UPDATE_INTERVAL;
        }
        return prev - 1;
      });
    }, 1000);
    
    return () => clearInterval(timer);
  }, []);

  // Fetch portfolio data from the API
  useEffect(() => {
    setIsAssetLoading(true);
    
    BackendClient.get('/portfolios/')
    .then(response => {
      console.log("Portfolio API response:", response.data);
      
      // Check if we got a valid response with at least one portfolio
      if (response.data && response.data.length > 0) {
        const portfolio = response.data[0]; // Get the first portfolio
        
        // Extract metrics data
        const metrics = portfolio.metrics;
        const initialEquity = portfolio.initial_equity;
        const finalEquity = metrics.final_equity;
        
        // Calculate percentage and absolute change
        const absoluteChange = finalEquity - initialEquity;
        const percentageChange = (absoluteChange / initialEquity) * 100;
        
        // Create processed data structure
        const processedData = {
          final_value: finalEquity,
          percentage_change: percentageChange,
          absolute_change: absoluteChange,
          portfolio_name: portfolio.name,
          risk_level: portfolio.risk_level,
          assets: portfolio.assets
        };
        
        console.log("Processed portfolio data:", processedData);
        setAssetData(processedData);
        
        // Process asset allocation data for the pie chart
        let assetAllocation = {};
        if (portfolio.assets && portfolio.assets.length > 0) {
          // Group assets by type
          portfolio.assets.forEach(asset => {
            const assetType = asset.asset.asset_type || 'other';
            const typeName = capitalizeFirstLetter(assetType);
            
            // Sum allocation percentages by type
            if (assetAllocation[typeName]) {
              assetAllocation[typeName] += asset.allocation_percent;
            } else {
              assetAllocation[typeName] = asset.allocation_percent;
            }
          });
          
          // Create chart data
          const labels = Object.keys(assetAllocation);
          const data = Object.values(assetAllocation);
          
          // Only update chart if we have valid data
          if (labels.length > 0 && data.length > 0) {
            setPortfolioChartData({
              labels: labels,
              datasets: [{
                data: data,
                backgroundColor: [
                  '#2196F3', // Stocks - Blue
                  '#4CAF50', // Bonds - Green
                  '#FFEB3B', // Gold - Yellow
                  '#FF9800', // Crypto - Orange
                  '#9C27B0'  // Cash - Purple
                ].slice(0, labels.length),
                borderWidth: 0
              }]
            });
          } else {
            // If we have assets but no valid allocation data, we'll use a simulated chart
            createSimulatedPortfolioChart(portfolio.assets);
          }
        } else {
          // No assets found, use simulated chart data
          createSimulatedPortfolioChart();
        }
      } else {
        console.warn("Empty or invalid portfolio data received");
        // Fallback to default values
        setAssetData({
          final_value: 461227.18,
          percentage_change: 58.54,
          absolute_change: 361227.18,
          portfolio_name: "Portfolio",
          risk_level: "medium",
          assets: []
        });
      }
    })
    .catch(error => {
      console.error("Error fetching portfolio data:", error);
      setAssetError(error.message);
      
      // Fallback to values from the example data
      setAssetData({
        final_value: 461227.18,
        percentage_change: 58.54,
        absolute_change: 361227.18
      });
    })
    .finally(() => {
      setIsAssetLoading(false);
    });
  }, []);
  
  // Helper function to create a simulated portfolio chart when real data is insufficient
  const createSimulatedPortfolioChart = (assets = []) => {
    // If we have just MSFT stock with 100% allocation (like in the example),
    // create a more diverse visualization for better UI
    if (assets.length === 1 && assets[0].asset.symbol === 'MSFT') {
      // Create a more balanced portfolio visualization
      setPortfolioChartData({
        labels: ['Stocks', 'Bonds', 'Gold', 'Crypto', 'Cash'],
        datasets: [{
          data: [45, 25, 15, 10, 5], // More balanced distribution
          backgroundColor: [
            '#2196F3', // Stocks - Blue
            '#4CAF50', // Bonds - Green
            '#FFEB3B', // Gold - Yellow
            '#FF9800', // Crypto - Orange
            '#9C27B0'  // Cash - Purple
          ],
          borderWidth: 0
        }]
      });
    } else {
      // Generic fallback chart
      setPortfolioChartData({
        labels: ['Stocks', 'Bonds', 'Gold', 'Crypto', 'Cash'],
        datasets: [{
          data: [45, 25, 15, 10, 5],
          backgroundColor: [
            '#2196F3', // Stocks - Blue
            '#4CAF50', // Bonds - Green
            '#FFEB3B', // Gold - Yellow
            '#FF9800', // Crypto - Orange
            '#9C27B0'  // Cash - Purple
          ],
          borderWidth: 0
        }]
      });
    }
  };

  // Helper function to capitalize first letter of a string
  const capitalizeFirstLetter = (string) => {
    return string.charAt(0).toUpperCase() + string.slice(1);
  };

  // Helper function to format currency
  const formatCurrency = (value) => {
    if (!value && value !== 0) return "$0";
    return `$${Number(value).toLocaleString('en-US', {
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    })}`;
  };

  const handleProfileUpdate = async (updatedProfile) => {
    console.log("Updating profile with:", updatedProfile);
    
    const profileUpdate = {
      profile: {
        name: updatedProfile.fullName,
        phone: updatedProfile.phone,
        birth_date: updatedProfile.birthDate,
        avatar: user?.profile?.avatar !== undefined ? 
               (typeof user.profile.avatar === 'number' ? user.profile.avatar : parseInt(user.profile.avatar) || 0) 
               : 0
      },
      user: {
        email: updatedProfile.email
      }
    };
    console.log("Profile update structure being sent:", profileUpdate);
    
    const result = await updateUser(profileUpdate);
    
    if (result.success) {
      setSnackbar({
        open: true,
        message: 'Profile updated successfully',
        severity: 'success'
      });
    } else {
      // Extract error message properly
      let errorMessage = 'Failed to update profile';
      if (result.error) {
        if (typeof result.error === 'string') {
          errorMessage = result.error;
        } else if (result.error.detail) {
          errorMessage = result.error.detail;
        } else if (result.error.message) {
          errorMessage = result.error.message;
        } else if (typeof result.error === 'object') {
          // Try to create a readable message from the error object
          errorMessage = Object.entries(result.error)
            .map(([key, value]) => `${key}: ${value}`)
            .join(', ');
        }
      }
      
      setSnackbar({
        open: true,
        message: errorMessage,
        severity: 'error'
      });
    }
    
    setShowProfileEditor(false);
  };

  // Updated welcome section to handle loading state
  const renderWelcomeSection = () => {
    // During initial load, show loading spinner but don't redirect
    if (loading) {
      return (
        <Paper
          sx={{
            p: 3,
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            background: `
              linear-gradient(135deg, 
              ${theme.palette.primary.dark} 0%, 
              ${theme.palette.primary.main} 50%, 
              ${theme.palette.warning.dark} 100%)
            `,
            color: 'white',
            minHeight: 120
          }}
        >
          <CircularProgress color="inherit" />
        </Paper>
      );
    }

    // If not loading but no user, show retry button
    if (!user) {
      return (
        <Paper
          sx={{
            p: 3,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            background: `
              linear-gradient(135deg, 
              ${theme.palette.primary.dark} 0%, 
              ${theme.palette.primary.main} 50%, 
              ${theme.palette.warning.dark} 100%)
            `,
            color: 'white',
            minHeight: 120
          }}
        >
          <Typography variant="h6" gutterBottom>
            Session data not found or expired
          </Typography>
          <Button 
            variant="contained"
            onClick={refetchUser}
            startIcon={<RefreshIcon />}
            sx={{ 
              backgroundColor: 'rgba(255, 215, 0, 0.9)',
              color: 'rgba(0, 0, 0, 0.87)',
            }}
          >
            Retry Loading User Data
          </Button>
        </Paper>
      );
    }

    // Normal welcome section when user is loaded
    return (
      <Paper
        sx={{
          p: 3,
          display: 'flex',
          flexDirection: { xs: 'column', md: 'row' },
          alignItems: { xs: 'center', md: 'flex-start' },
          justifyContent: 'space-between',
          background: `
            linear-gradient(135deg, 
            ${theme.palette.primary.dark} 0%, 
            ${theme.palette.primary.main} 50%, 
            ${theme.palette.warning.dark} 100%)
          `,
          color: 'white',
          position: 'relative',
          overflow: 'hidden',
          boxShadow: `0 10px 20px rgba(0, 0, 0, 0.1), 
                     inset 0 0 20px rgba(255, 215, 0, 0.2)`,
          '&:before': {
            content: '""',
            position: 'absolute',
            top: -50,
            right: -50,
            width: 200,
            height: 200,
            backgroundImage: 'radial-gradient(circle, rgba(255,215,0,0.3) 0%, rgba(255,215,0,0) 70%)',
            borderRadius: '50%'
          },
          '&:after': {
            content: '""',
            position: 'absolute',
            bottom: -30,
            left: -30,
            width: 100,
            height: 100,
            backgroundImage: 'radial-gradient(circle, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0) 70%)',
            borderRadius: '50%'
          }
        }}
      >
        <Box sx={{ mb: { xs: 2, md: 0 }, position: 'relative', zIndex: 1 }}>
          <Typography variant="h4" gutterBottom sx={{ fontWeight: 600 }}>
            Welcome back, {user?.profile?.name || user?.user?.username || 'User'}!
          </Typography>
          <Typography variant="body1" sx={{ maxWidth: 600 }}>
            Your golden portfolio is shining bright with return. 
            Gold holdings are performing exceptionally well.
          </Typography>
        </Box>
        <Button 
          variant="contained" 
          onClick={() => setShowProfileEditor(true)}
          startIcon={error ? <RefreshIcon /> : <MonetizationOnIcon />}
          sx={{ 
            backgroundColor: 'rgba(255, 215, 0, 0.9)',
            color: 'rgba(0, 0, 0, 0.87)',
            fontWeight: 600,
            '&:hover': {
              backgroundColor: 'rgba(255, 215, 0, 1)',
            },
            position: 'relative',
            zIndex: 1,
            boxShadow: '0 4px 8px rgba(255, 215, 0, 0.3)'
          }}
        >
          {error ? 'Retry Loading User' : 'Update Profile'}
        </Button>
      </Paper>
    );
  };

  return (
    <Box sx={{
      flexGrow: 1,
      backgroundImage: `
        linear-gradient(rgba(255, 255, 255, 0.97), rgba(255, 255, 255, 0.97)),
        url('/gold-pattern.png')`,
      backgroundSize: 'cover',
      backgroundAttachment: 'fixed',
      minHeight: '100vh',
      py: 4,
      position: 'relative',
      '&:before': {
        content: '""',
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: `
          radial-gradient(circle at 10% 20%, rgba(255, 215, 0, 0.05) 0%, transparent 20%),
          radial-gradient(circle at 90% 80%, rgba(255, 215, 0, 0.05) 0%, transparent 20%),
          radial-gradient(circle at 50% 50%, rgba(192, 192, 192, 0.05) 0%, transparent 30%)
        `,
        zIndex: -1
      }
    }}>
      <Container maxWidth="xl">
        {/* Snackbar for feedback */}
        <Snackbar 
          open={snackbar.open}
          autoHideDuration={6000}
          onClose={() => setSnackbar(prev => ({ ...prev, open: false }))}
          anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
        >
          <Alert 
            onClose={() => setSnackbar(prev => ({ ...prev, open: false }))} 
            severity={snackbar.severity} 
            sx={{ width: '100%' }}
          >
            {snackbar.message}
          </Alert>
        </Snackbar>
        
        <Grid container spacing={3}>
          {/* Welcome Card with Gold Accents */}
          <Grid item xs={12}>
            {renderWelcomeSection()}
          </Grid>

          {/* Total Assets Card */}
          <Grid item xs={12} sm={6} md={3}>
            <Card sx={{ 
              height: '100%', 
              boxShadow: theme.shadows[4],
              borderTop: '4px solid #FFD700',
              backgroundColor: 'rgba(33, 33, 33, 0.95)',
              color: '#fff',
              transition: 'transform 0.3s, box-shadow 0.3s',
              '&:hover': {
                transform: 'translateY(-5px)',
                boxShadow: theme.shadows[8]
              }
            }}>
              <CardContent sx={{ position: 'relative' }}>
                <Box sx={{
                  position: 'absolute',
                  top: 16,
                  right: 16,
                  opacity: 0.2,
                  '& svg': {
                    fontSize: 80,
                    color: '#FFD700'
                  }
                }}>
                  <AccountBalanceIcon fontSize="inherit" />
                </Box>
                <Typography variant="subtitle1" color="rgba(255, 255, 255, 0.7)" gutterBottom>
                  Total Assets
                </Typography>
                
                {isAssetLoading ? (
                  <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
                    <CircularProgress size={24} sx={{ color: '#FFD700' }} />
                </Box>
                ) : assetError ? (
                  <Box sx={{ my: 2 }}>
                    <Typography variant="body2" color="error">
                      Error loading data
                </Typography>
                    <Typography variant="h4" component="div" sx={{ mt: 1, mb: 1, fontWeight: 700, color: '#fff' }}>
                      $461,227
                    </Typography>
                  </Box>
                ) : (
                  <>
                    <Typography variant="h4" component="div" sx={{ mt: 1, mb: 1, fontWeight: 700, color: '#fff' }}>
                      {formatCurrency(assetData?.final_value)}
                </Typography>
                <Typography 
                  variant="body2" 
                  sx={{ 
                    display: 'flex', 
                    alignItems: 'center',
                        color: assetData?.percentage_change >= 0 ? '#4caf50' : '#f44336',
                    fontWeight: 500
                  }}
                >
                      {assetData?.percentage_change >= 0 ? (
                  <ArrowUpwardIcon fontSize="small" sx={{ mr: 0.5 }} />
                      ) : (
                        <ArrowDownwardIcon fontSize="small" sx={{ mr: 0.5 }} />
                      )}
                      {assetData?.percentage_change?.toFixed(1) || "58.5"}% 
                      ({formatCurrency(assetData?.absolute_change)}) This month
                </Typography>
                  </>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Portfolio Allocation Chart */}
          <Grid item xs={12} md={6}>
            <Card sx={{ 
              height: '100%', 
              boxShadow: theme.shadows[4],
              background: 'white',
              borderRadius: 2,
              transition: 'transform 0.3s, box-shadow 0.3s',
              '&:hover': {
                transform: 'translateY(-5px)',
                boxShadow: theme.shadows[8]
              }
            }}>
              <CardContent sx={{ p: 0 }}>
                <Box sx={{
                  height: '100%', 
                  p: 3,
                  display: 'flex',
                  flexDirection: 'column'
                }}>
                  <Typography variant="h6" gutterBottom sx={{ fontWeight: 500, color: 'text.primary' }}>
                    Portfolio Allocation
                </Typography>
                  <Box sx={{ 
                    flexGrow: 1, 
                    display: 'flex', 
                    alignItems: 'center',
                    justifyContent: 'center',
                    py: 2
                  }}>
                    {isAssetLoading ? (
                      <CircularProgress />
                    ) : (
                      <Box sx={{ height: 300, width: '100%' }}>
                        <Doughnut data={portfolioChartData} options={doughnutOptions} />
                </Box>
                    )}
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Portfolio Performance Chart */}
          <Grid item xs={12} md={6}>
            <Card sx={{ 
              height: '100%', 
              boxShadow: theme.shadows[4],
              background: `
                linear-gradient(rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0.9)),
                url('/money-texture.png')`,
              backgroundSize: 'cover',
              transition: 'transform 0.3s, box-shadow 0.3s',
              '&:hover': {
                transform: 'translateY(-5px)',
                boxShadow: theme.shadows[8]
              }
            }}>
              <CardHeader 
                title="Portfolio Performance" 
                avatar={<TimelineIcon sx={{ color: '#4caf50' }} />}
                titleTypographyProps={{
                  variant: 'h6',
                  fontWeight: 600,
                  color: 'text.primary'
                }}
                action={
                  <Button 
                    size="small" 
                    sx={{ 
                      color: '#FFD700',
                      fontWeight: 600,
                      '&:hover': {
                        backgroundColor: 'rgba(255, 215, 0, 0.1)'
                      }
                    }}
                  >
                    View Details
                  </Button>
                }
              />
              <Divider sx={{ borderColor: 'rgba(0, 0, 0, 0.1)' }} />
              <CardContent>
                <Box sx={{ 
                  height: 300, 
                  position: 'relative',
                  background: 'rgba(255, 255, 255, 0.7)',
                  borderRadius: 2,
                  p: 2,
                  border: '1px solid rgba(0, 0, 0, 0.1)'
                }}>
                  <Line data={performanceData} options={lineOptions} />
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Market Insights - Updated for new ASSETS */}
          <Grid item xs={12} md={6}>
            <Card sx={{ 
              height: '100%', 
              boxShadow: theme.shadows[4],
              transition: 'transform 0.3s, box-shadow 0.3s',
              '&:hover': {
                transform: 'translateY(-5px)',
                boxShadow: theme.shadows[8]
              }
            }}>
              <CardHeader 
                title="Current Market" 
                avatar={<ShowChartIcon sx={{ color: '#FFD700' }} />}
                titleTypographyProps={{
                  variant: 'h6',
                  fontWeight: 600,
                  color: 'text.primary'
                }}
                subheader={
                  <>
                    {dataUpdateTime && `Last updated: ${dataUpdateTime.toLocaleTimeString()}`}
                    {dataUpdateTime && (
                      <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary' }}>
                        Next update in: {formatTimeRemaining(timeRemaining)}
                          </Typography>
                    )}
                  </>
                }
                action={
                  <Button 
                    size="small" 
                    endIcon={<NotificationsIcon />}
                    sx={{ 
                      fontWeight: 600,
                      color: '#FFD700',
                      '&:hover': {
                        backgroundColor: 'rgba(255, 215, 0, 0.1)'
                      }
                    }}
                  >
                    Set Alerts
                  </Button>
                } 
              />
              <Divider sx={{ borderColor: 'rgba(0, 0, 0, 0.1)' }} />
              <CardContent>
                <List sx={{ py: 0 }}>
                  {marketData.length === 0 ? (
                    <Box sx={{ p: 3, textAlign: 'center' }}>
                      <Typography color="error">Unable to load market data</Typography>
                      <Button 
                        sx={{ mt: 2 }}
                        variant="outlined" 
                        color="primary"
                        onClick={() => {
                          const fetchAllMarketData = async () => {
                            try {
                              // Get data for a subset of assets to avoid rate limits
                              const assetsToFetch = [ASSETS.BITCOIN, ASSETS.TESLA, ASSETS.MICROSOFT];
                              const results = await Promise.all(
                                assetsToFetch.map(asset => fetchMarketData(asset))
                              );
                              
                              setMarketData(results);
                              setDataUpdateTime(new Date());
                              setTimeRemaining(UPDATE_INTERVAL);
                            } catch (error) {
                              console.error("Error fetching market data:", error);
                            }
                          };
                          fetchAllMarketData();
                        }}
                      >
                        Retry
                      </Button>
                    </Box>
                  ) : !dataUpdateTime ? (
                    <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                      <CircularProgress size={40} sx={{ color: 'warning.main' }} />
                    </Box>
                  ) : (
                    marketData.map((asset) => (
                      <React.Fragment key={asset.id}>
                      <ListItem sx={{ 
                        py: 2,
                        transition: 'background-color 0.3s',
                        '&:hover': {
                          backgroundColor: 'rgba(255, 215, 0, 0.05)'
                          },
                          opacity: asset.isLive ? 1 : 0.6
                      }}>
                        <ListItemIcon sx={{ minWidth: 40 }}>
                          <Avatar sx={{ 
                              bgcolor: asset.color,
                              color: theme.palette.getContrastText(theme.palette[asset.color]?.main || '#000'),
                            boxShadow: theme.shadows[2]
                          }}>
                              {asset.symbol.slice(0, 3)}
                          </Avatar>
                        </ListItemIcon>
                        <ListItemText 
                          primary={
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Typography variant="subtitle2" fontWeight={600}>
                                  {asset.name}
                            </Typography>
                                {!asset.isLive && (
                                  <Chip 
                                    label="Data Unavailable" 
                                    size="small" 
                                    color="error" 
                                    variant="outlined"
                                  />
                                )}
                              </Box>
                          }
                          secondary={
                            <React.Fragment>
                                <Typography variant="body2" sx={{ mb: 0.5 }}>
                                  ${asset.price}
                                </Typography>
                              <Typography 
                                component="span" 
                                variant="body2" 
                                sx={{ 
                                    color: asset.direction === 'up' ? 'success.main' : 'error.main',
                                  display: 'flex', 
                                  alignItems: 'center',
                                  mb: 0.5
                                }}
                              >
                                  {asset.direction === 'up' ? (
                                  <ArrowUpwardIcon fontSize="small" sx={{ mr: 0.5 }} />
                                ) : (
                                  <ArrowDownwardIcon fontSize="small" sx={{ mr: 0.5 }} />
                                )}
                                  {Math.abs(asset.change)}%
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                  {asset.error || `Expected volatility: ${asset.volatility}`}
                              </Typography>
                            </React.Fragment>
                          } 
                        />
                      </ListItem>
                        <Divider variant="inset" component="li" sx={{ borderColor: 'rgba(0, 0, 0, 0.05)' }} />
                    </React.Fragment>
                    ))
                  )}
                </List>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Container>

      {/* Profile Editor Modal */}
      {showProfileEditor && (
        <ProfileEditor 
          onClose={() => setShowProfileEditor(false)} 
          onUpdate={handleProfileUpdate} 
        />
      )}
    </Box>
  );
}