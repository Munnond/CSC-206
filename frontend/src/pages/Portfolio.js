import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  CardHeader,
  Divider,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  IconButton,
  ButtonGroup,
  Button,
  Menu,
  MenuItem,
  ListItemIcon,
  Tabs,
  Tab,
  Alert
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  MoreVert as MoreVertIcon,
  Archive as ArchiveIcon,
  Delete as DeleteIcon,
  Add as AddIcon,
  RemoveRedEye as ViewIcon,
  Lightbulb as LightbulbIcon,
  Warning as WarningIcon,
  InsertChart as InsertChartIcon
} from '@mui/icons-material';
import { Line } from 'react-chartjs-2';

// Sample portfolio data
const portfolioHoldings = [
  {
    id: 1,
    symbol: 'AAPL',
    name: 'Apple Inc.',
    shares: 25,
    price: 178.72,
    value: 4468.00,
    costBasis: 3875.25,
    return: 15.3,
    allocation: 14.8,
    sector: 'Technology'
  },
  {
    id: 2,
    symbol: 'MSFT',
    name: 'Microsoft Corporation',
    shares: 15,
    price: 325.42,
    value: 4881.30,
    costBasis: 4200.00,
    return: 16.2,
    allocation: 16.1,
    sector: 'Technology'
  },
  {
    id: 3,
    symbol: 'AMZN',
    name: 'Amazon.com Inc.',
    shares: 12,
    price: 177.23,
    value: 2126.76,
    costBasis: 2340.24,
    return: -9.1,
    allocation: 7.0,
    sector: 'Consumer Discretionary'
  },
  {
    id: 4,
    symbol: 'JNJ',
    name: 'Johnson & Johnson',
    shares: 18,
    price: 152.50,
    value: 2745.00,
    costBasis: 2970.54,
    return: -7.6,
    allocation: 9.1,
    sector: 'Healthcare'
  },
  {
    id: 5,
    symbol: 'V',
    name: 'Visa Inc.',
    shares: 22,
    price: 272.36,
    value: 5991.92,
    costBasis: 5280.32,
    return: 13.5,
    allocation: 19.8,
    sector: 'Financial Services'
  },
  {
    id: 6,
    symbol: 'VTI',
    name: 'Vanguard Total Stock Market ETF',
    shares: 35,
    price: 240.12,
    value: 8404.20,
    costBasis: 7752.50,
    return: 8.4,
    allocation: 27.8,
    sector: 'ETF'
  },
  {
    id: 7,
    symbol: 'VXUS',
    name: 'Vanguard Total International Stock ETF',
    shares: 40,
    price: 58.78,
    value: 2351.20,
    costBasis: 2480.00,
    return: -5.2,
    allocation: 7.8,
    sector: 'ETF'
  }
];

// Portfolio performance data
const portfolioPerformanceData = {
  labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'],
  datasets: [
    {
      label: 'Portfolio Value',
      data: [28000, 28500, 29200, 29800, 30100, 29500, 30200, 30800, 30968.38],
      fill: true,
      backgroundColor: 'rgba(25, 118, 210, 0.1)',
      borderColor: 'rgba(25, 118, 210, 1)',
      tension: 0.4,
    },
    {
      label: 'S&P 500',
      data: [28000, 28200, 28600, 29000, 29400, 29100, 29800, 30200, 30500],
      fill: false,
      borderColor: 'rgba(244, 67, 54, 1)',
      borderDash: [5, 5],
      tension: 0.4,
    }
  ],
};

// Chart options
const chartOptions = {
  responsive: true,
  plugins: {
    legend: {
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
        callback: (value) => '$' + value.toLocaleString(),
      },
    },
  },
  interaction: {
    mode: 'nearest',
    axis: 'x',
    intersect: false
  }
};

// AI Insights
const aiInsights = [
  {
    id: 1,
    type: 'opportunity',
    title: 'Increase Diversification',
    description: 'Your portfolio is heavily weighted in Technology. Consider adding exposure to Utilities and Real Estate sectors.',
    icon: <LightbulbIcon color="primary" />,
  },
  {
    id: 2,
    type: 'risk',
    title: 'Concentration Risk',
    description: 'Five stocks make up over 65% of your portfolio. Consider rebalancing to reduce individual stock risk.',
    icon: <WarningIcon color="warning" />,
  }
];

export default function Portfolio() {
  const [anchorEl, setAnchorEl] = useState(null);
  const [tabValue, setTabValue] = useState(0);
  
  const handleMenuOpen = (event, holding) => {
    setAnchorEl(event.currentTarget);
  };
  
  const handleMenuClose = () => {
    setAnchorEl(null);
  };
  
  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };
  
  // Calculate totals
  const totalValue = portfolioHoldings.reduce((sum, holding) => sum + holding.value, 0);
  const totalCostBasis = portfolioHoldings.reduce((sum, holding) => sum + holding.costBasis, 0);
  const totalReturn = ((totalValue - totalCostBasis) / totalCostBasis) * 100;
  
  // Time periods for performance
  const periods = ['1M', '3M', '6M', 'YTD', '1Y', '3Y', '5Y', 'All'];
  
  return (
    <Box sx={{ flexGrow: 1 }}>
      <Grid container spacing={3}>
        {/* Portfolio Summary */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h4" gutterBottom>
              Your Portfolio
            </Typography>
            <Grid container spacing={3}>
              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Typography variant="subtitle1" color="text.secondary">
                      Total Value
                    </Typography>
                    <Typography variant="h4" component="div" sx={{ mt: 1, mb: 1 }}>
                      ${totalValue.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                    </Typography>
                    <Typography 
                      variant="body2" 
                      sx={{ 
                        display: 'flex', 
                        alignItems: 'center',
                        color: totalReturn >= 0 ? 'success.main' : 'error.main'
                      }}
                    >
                      {totalReturn >= 0 ? 
                        <TrendingUpIcon fontSize="small" /> : 
                        <TrendingDownIcon fontSize="small" />
                      }
                      {Math.abs(totalReturn).toFixed(2)}% Overall
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Typography variant="subtitle1" color="text.secondary">
                      Daily Change
                    </Typography>
                    <Typography variant="h4" component="div" sx={{ mt: 1, mb: 1 }}>
                      +$152.25
                    </Typography>
                    <Typography 
                      variant="body2" 
                      sx={{ 
                        display: 'flex', 
                        alignItems: 'center',
                        color: 'success.main'
                      }}
                    >
                      <TrendingUpIcon fontSize="small" />
                      0.49% Today
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Typography variant="subtitle1" color="text.secondary">
                      Holdings
                    </Typography>
                    <Typography variant="h4" component="div" sx={{ mt: 1, mb: 1 }}>
                      {portfolioHoldings.length}
                    </Typography>
                    <Typography 
                      variant="body2" 
                      color="text.secondary"
                    >
                      Across {new Set(portfolioHoldings.map(h => h.sector)).size} sectors
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Typography variant="subtitle1" color="text.secondary">
                      Cash
                    </Typography>
                    <Typography variant="h4" component="div" sx={{ mt: 1, mb: 1 }}>
                      $2,450.62
                    </Typography>
                    <Typography 
                      variant="body2" 
                      color="text.secondary"
                    >
                      7.3% of portfolio
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
        
        {/* Performance Chart */}
        <Grid item xs={12}>
          <Card>
            <CardHeader 
              title="Portfolio Performance" 
              action={
                <ButtonGroup variant="outlined" size="small">
                  {periods.map((period) => (
                    <Button 
                      key={period}
                      sx={{ 
                        fontSize: '0.75rem',
                        px: 1, 
                        py: 0.5,
                        ...(period === '6M' ? { 
                          backgroundColor: 'primary.main', 
                          color: 'white',
                          '&:hover': {
                            backgroundColor: 'primary.dark',
                          }
                        } : {})
                      }}
                    >
                      {period}
                    </Button>
                  ))}
                </ButtonGroup>
              }
            />
            <Divider />
            <CardContent>
              <Box sx={{ height: 300, position: 'relative' }}>
                <Line data={portfolioPerformanceData} options={chartOptions} />
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        {/* Holdings Table and Insights */}
        <Grid item xs={12} md={8}>
          <Card>
            <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
              <Tabs 
                value={tabValue} 
                onChange={handleTabChange}
                aria-label="portfolio tabs"
              >
                <Tab label="All Holdings" id="tab-0" />
                <Tab label="Stocks" id="tab-1" />
                <Tab label="ETFs" id="tab-2" />
                <Tab label="Mutual Funds" id="tab-3" />
              </Tabs>
            </Box>
            <TableContainer component={Box}>
              <Table sx={{ minWidth: 650 }}>
                <TableHead>
                  <TableRow>
                    <TableCell>Symbol</TableCell>
                    <TableCell>Name</TableCell>
                    <TableCell align="right">Shares</TableCell>
                    <TableCell align="right">Price</TableCell>
                    <TableCell align="right">Value</TableCell>
                    <TableCell align="right">Return</TableCell>
                    <TableCell align="right">Allocation</TableCell>
                    <TableCell align="right">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {portfolioHoldings.map((holding) => (
                    <TableRow
                      key={holding.id}
                      sx={{ '&:last-child td, &:last-child th': { border: 0 } }}
                    >
                      <TableCell component="th" scope="row">
                        <Typography variant="body2" fontWeight="bold">{holding.symbol}</Typography>
                      </TableCell>
                      <TableCell>
                        <Box>
                          <Typography variant="body2">{holding.name}</Typography>
                          <Chip 
                            label={holding.sector} 
                            size="small" 
                            sx={{ 
                              height: 20, 
                              fontSize: '0.65rem', 
                              mt: 0.5, 
                              backgroundColor: 'rgba(0, 0, 0, 0.08)'
                            }} 
                          />
                        </Box>
                      </TableCell>
                      <TableCell align="right">
                        {holding.shares}
                      </TableCell>
                      <TableCell align="right">
                        ${holding.price.toFixed(2)}
                      </TableCell>
                      <TableCell align="right">
                        ${holding.value.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                      </TableCell>
                      <TableCell align="right">
                        <Box sx={{ 
                          display: 'flex', 
                          alignItems: 'center', 
                          justifyContent: 'flex-end',
                          color: holding.return >= 0 ? 'success.main' : 'error.main'
                        }}>
                          {holding.return >= 0 ? 
                            <TrendingUpIcon fontSize="small" sx={{ mr: 0.5 }} /> : 
                            <TrendingDownIcon fontSize="small" sx={{ mr: 0.5 }} />
                          }
                          {Math.abs(holding.return).toFixed(2)}%
                        </Box>
                      </TableCell>
                      <TableCell align="right">
                        {holding.allocation.toFixed(1)}%
                      </TableCell>
                      <TableCell align="right">
                        <IconButton
                          size="small"
                          onClick={(event) => handleMenuOpen(event, holding)}
                        >
                          <MoreVertIcon fontSize="small" />
                        </IconButton>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
            <Box sx={{ p: 2, display: 'flex', justifyContent: 'flex-end' }}>
              <Button 
                variant="contained" 
                startIcon={<AddIcon />}
                sx={{ ml: 1 }}
              >
                Add Investment
              </Button>
            </Box>
          </Card>
        </Grid>
        
        {/* AI Insights */}
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardHeader 
              title="AI Insights" 
              subheader="Personalized investment recommendations"
              action={
                <IconButton aria-label="settings">
                  <InsertChartIcon />
                </IconButton>
              }
            />
            <Divider />
            <CardContent>
              <Alert severity="info" sx={{ mb: 3 }}>
                Based on your portfolio analysis, we've identified 2 key insights for optimization.
              </Alert>
              
              {aiInsights.map((insight) => (
                <Card key={insight.id} variant="outlined" sx={{ mb: 2 }}>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'flex-start', mb: 1 }}>
                      <Box sx={{ mr: 1 }}>{insight.icon}</Box>
                      <Typography variant="h6">{insight.title}</Typography>
                    </Box>
                    <Typography variant="body2" color="text.secondary">
                      {insight.description}
                    </Typography>
                  </CardContent>
                </Card>
              ))}
              
              <Divider sx={{ my: 2 }} />
              
              <Typography variant="subtitle1" gutterBottom>Sector Allocation Analysis</Typography>
              <Typography variant="body2" paragraph>
                Your portfolio has high concentration in Technology (30.9%) and Financial Services (19.8%). Consider diversifying into underrepresented sectors.
              </Typography>
              
              <Button 
                variant="outlined" 
                fullWidth 
                sx={{ mt: 1 }}
                startIcon={<ViewIcon />}
              >
                View Detailed Analysis
              </Button>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      
      {/* Action Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={handleMenuClose}>
          <ListItemIcon>
            <AddIcon fontSize="small" />
          </ListItemIcon>
          Buy More
        </MenuItem>
        <MenuItem onClick={handleMenuClose}>
          <ListItemIcon>
            <ArchiveIcon fontSize="small" />
          </ListItemIcon>
          Sell
        </MenuItem>
        <Divider />
        <MenuItem onClick={handleMenuClose}>
          <ListItemIcon>
            <ViewIcon fontSize="small" />
          </ListItemIcon>
          View Details
        </MenuItem>
        <MenuItem onClick={handleMenuClose}>
          <ListItemIcon>
            <DeleteIcon fontSize="small" />
          </ListItemIcon>
          Remove from Watchlist
        </MenuItem>
      </Menu>
    </Box>
  );
}

// ... Do NOT add any other function Portfolio or export default Portfolio below this line ...