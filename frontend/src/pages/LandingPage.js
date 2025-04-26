import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Button,
  Container,
  Grid,
  Typography,
  useTheme,
  Paper,
  Stack,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Avatar,
  Tabs,
  Tab,
  Chip,
  IconButton,
  Tooltip,
  Fade,
  Grow,
  Slide,
  Zoom,
  useMediaQuery,
  InputAdornment,
  TextField,
  Skeleton
} from '@mui/material';
import {
  ArrowForward,
  CheckCircle,
  Security,
  Timeline,
  ShowChart,
  AutoGraph,
  BarChart,
  PieChart,
  AccountBalance,
  MonetizationOn,
  AttachMoney,
  TrendingUp,
  VerifiedUser,
  People,
  Star,
  StarBorder,
  ArrowRightAlt,
  Search,
  Notifications,
  Email,
  Phone,
  LocationOn,
  Twitter,
  LinkedIn,
  Facebook,
  Instagram,
  GitHub,
  YouTube,
  KeyboardArrowDown,
  KeyboardArrowUp
} from '@mui/icons-material';
import { styled, keyframes } from '@mui/material/styles';
import { Line, Bar, Pie } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip as ChartTooltip,
  Legend,
  Filler
} from 'chart.js';
import Logo from '../components/Logo/Logo';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  ChartTooltip,
  Legend,
  Filler
);

// Animation for the hero section
const floatAnimation = keyframes`
  0% { transform: translateY(0px); }
  50% { transform: translateY(-20px); }
  100% { transform: translateY(0px); }
`;

// Styled components
const FloatingCard = styled(Paper)(({ theme }) => ({
  animation: `${floatAnimation} 6s ease-in-out infinite`,
  transition: 'all 0.3s ease',
  '&:hover': {
    transform: 'translateY(-10px)',
    boxShadow: `0 30px 60px rgba(0, 0, 0, 0.4)`
  }
}));

const GlowButton = styled(Button)(({ theme }) => ({
  boxShadow: `0 0 20px ${theme.palette.primary.main}`,
  '&:hover': {
    boxShadow: `0 0 30px ${theme.palette.primary.light}`
  }
}));

const FeatureCard = styled(Paper)(({ theme }) => ({
  position: 'relative',
  overflow: 'hidden',
  '&::before': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: '4px',
    background: `linear-gradient(90deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`
  }
}));

// Data for charts
const portfolioData = {
  labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'],
  datasets: [
    {
      label: 'Portfolio Value',
      data: [50000, 62000, 58000, 75000, 82000, 90000, 105000],
      borderColor: '#3EEFBF',
      backgroundColor: 'rgba(62, 237, 191, 0.1)',
      borderWidth: 3,
      tension: 0.4,
      fill: true,
      pointBackgroundColor: '#3EEFBF',
      pointRadius: 5,
      pointHoverRadius: 8
    }
  ]
};

const assetAllocationData = {
  labels: ['Stocks', 'Bonds', 'Crypto', 'Real Estate', 'Commodities'],
  datasets: [
    {
      data: [45, 20, 15, 10, 10],
      backgroundColor: [
        '#3EEFBF',
        '#FF4081',
        '#536DFE',
        '#FFC107',
        '#9C27B0'
      ],
      borderWidth: 0
    }
  ]
};

const performanceData = {
  labels: ['Your Portfolio', 'S&P 500', 'NASDAQ', 'Gold'],
  datasets: [
    {
      label: 'YTD Return (%)',
      data: [12.5, 8.2, 10.1, 2.3],
      backgroundColor: [
        'rgba(62, 237, 191, 0.8)',
        'rgba(255, 255, 255, 0.2)',
        'rgba(255, 255, 255, 0.2)',
        'rgba(255, 255, 255, 0.2)'
      ],
      borderColor: [
        '#3EEFBF',
        'rgba(255, 255, 255, 0.5)',
        'rgba(255, 255, 255, 0.5)',
        'rgba(255, 255, 255, 0.5)'
      ],
      borderWidth: 1
    }
  ]
};

const features = [
  {
    title: 'AI-Powered Portfolio Management',
    description: 'Our proprietary AI algorithms continuously analyze market conditions and automatically rebalance your portfolio to maximize returns while minimizing risk.',
    icon: <ShowChart fontSize="large" />,
    stats: 'Avg. 18% higher returns than manual strategies'
  },
  {
    title: 'Smart Asset Allocation',
    description: 'Dynamic asset allocation that adapts to market conditions and your personal risk tolerance, optimized using machine learning models trained on decades of market data.',
    icon: <PieChart fontSize="large" />,
    stats: 'Reduces risk by 32% compared to static allocation'
  }
];

const testimonials = [
  {
    name: 'Dr. Sarah Johnson',
    position: 'Chief Investment Officer, Vertex Capital',
    quote: 'IntelliWealth has transformed how we manage our clients\' portfolios. The AI-driven insights have consistently outperformed our traditional models by 15-20% annually.',
    initials: 'SJ',
    rating: 5
  },
  {
    name: 'Mark Chen',
    position: 'Private Investor',
    quote: 'As someone who manages my own investments, IntelliWealth gives me the sophisticated tools previously only available to institutional investors. My portfolio has grown 28% since switching.',
    initials: 'MC',
    rating: 5
  },
  {
    name: 'Jessica Williams',
    position: 'Financial Advisor',
    quote: 'I recommend IntelliWealth to all my clients. The platform combines institutional-grade analytics with an interface that anyone can understand. Client satisfaction has never been higher.',
    initials: 'JW',
    rating: 4
  },
  {
    name: 'Robert Kiyosaki',
    position: 'Founder, Rich Dad Company',
    quote: 'In my 30 years in finance, I\'ve never seen a platform that democratizes wealth management like IntelliWealth. This is the future of investing.',
    initials: 'RK',
    rating: 5
  }
];

const assetClasses = [
  { name: 'US Equities', allocation: '35%', change: '+2.1%', color: '#3EEFBF' },
  { name: 'International Stocks', allocation: '25%', change: '-0.8%', color: '#536DFE' },
  { name: 'Bonds', allocation: '20%', change: '+0.5%', color: '#FF4081' },
  { name: 'Real Estate', allocation: '10%', change: '+1.2%', color: '#FFC107' },
  { name: 'Commodities', allocation: '5%', change: '-1.5%', color: '#9C27B0' },
  { name: 'Cryptocurrency', allocation: '5%', change: '+8.3%', color: '#00BCD4' }
];

const performanceMetrics = [
  { metric: 'YTD Return', value: '+12.5%', benchmark: '+8.2%' },
  { metric: '1-Year Return', value: '+18.3%', benchmark: '+12.1%' },
  { metric: '3-Year CAGR', value: '+14.7%', benchmark: '+9.8%' },
  { metric: 'Sharpe Ratio', value: '1.8', benchmark: '1.2' },
  { metric: 'Max Drawdown', value: '-12.4%', benchmark: '-18.7%' },
  { metric: 'Volatility', value: '8.2%', benchmark: '10.5%' }
];

const newsUpdates = [
  {
    title: 'Fed Signals Potential Rate Cut in Q3',
    source: 'Financial Times',
    time: '2 hours ago',
    impact: 'High'
  },
  {
    title: 'Tech Stocks Rally on AI Earnings Boom',
    source: 'Wall Street Journal',
    time: '5 hours ago',
    impact: 'Medium'
  },
  {
    title: 'Bitcoin Surges Past $60,000 as ETF Approvals Loom',
    source: 'CoinDesk',
    time: '1 day ago',
    impact: 'High'
  },
  {
    title: 'Global Manufacturing PMI Shows Signs of Recovery',
    source: 'Bloomberg',
    time: '1 day ago',
    impact: 'Medium'
  }
];

const partners = [
  { name: 'BlackRock', logo: '/logos/blackrock.png' },
  { name: 'Vanguard', logo: '/logos/vanguard.png' },
  { name: 'Fidelity', logo: '/logos/fidelity.png' },
  { name: 'Charles Schwab', logo: '/logos/schwab.png' },
  { name: 'Coinbase', logo: '/logos/coinbase.png' },
  { name: 'Bloomberg', logo: '/logos/bloomberg.png' }
];

const AssetClassItem = ({ asset }) => {
  return (
    <Box sx={{ mb: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
        <Typography variant="body2" sx={{ fontWeight: 600 }}>
          {asset.name}
        </Typography>
        <Typography variant="body2" sx={{ 
          fontWeight: 600,
          color: asset.change.startsWith('+') ? 'success.main' : 'error.main'
        }}>
          {asset.change}
        </Typography>
      </Box>
      <Box sx={{ 
        height: 8, 
        bgcolor: 'rgba(255, 255, 255, 0.1)', 
        borderRadius: 4,
        overflow: 'hidden'
      }}>
        <Box sx={{ 
          width: asset.allocation, 
          height: '100%', 
          bgcolor: asset.color,
          borderRadius: 4
        }} />
      </Box>
    </Box>
  );
};

const PerformanceMetricItem = ({ metric }) => {
  return (
    <Paper
      elevation={0}
      sx={{
        p: 2,
        mb: 2,
        background: 'rgba(30, 30, 30, 0.6)',
        border: '1px solid rgba(255, 255, 255, 0.1)'
      }}
    >
      <Typography variant="body2" sx={{ mb: 1, color: 'text.secondary' }}>
        {metric.metric}
      </Typography>
      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
        <Typography variant="h6" sx={{ fontWeight: 700 }}>
          {metric.value}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Benchmark: {metric.benchmark}
        </Typography>
      </Box>
    </Paper>
  );
};

const NewsUpdateItem = ({ news }) => {
  return (
    <Paper
      elevation={0}
      sx={{
        p: 2,
        mb: 2,
        background: 'rgba(30, 30, 30, 0.6)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        '&:hover': {
          borderColor: 'rgba(62, 237, 191, 0.3)'
        }
      }}
    >
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
        <Chip
          label={news.impact}
          size="small"
          sx={{
            bgcolor: news.impact === 'High' ? 'error.dark' : 'warning.dark',
            color: 'white',
            fontWeight: 600
          }}
        />
        <Typography variant="caption" color="text.secondary">
          {news.time}
        </Typography>
      </Box>
      <Typography variant="body1" sx={{ mb: 1, fontWeight: 500 }}>
        {news.title}
      </Typography>
      <Typography variant="caption" color="text.secondary">
        {news.source}
      </Typography>
    </Paper>
  );
};

const RatingStars = ({ rating }) => {
  return (
    <Box sx={{ display: 'flex' }}>
      {[1, 2, 3, 4, 5].map((star) => (
        star <= rating ? 
          <Star key={star} sx={{ color: 'warning.main', fontSize: '1rem' }} /> : 
          <StarBorder key={star} sx={{ color: 'text.secondary', fontSize: '1rem' }} />
      ))}
    </Box>
  );
};

export default function LandingPage() {
  const theme = useTheme();
  const navigate = useNavigate();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [activeTab, setActiveTab] = useState(0);
  const [showFullFeatures, setShowFullFeatures] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Simulate loading
    const timer = setTimeout(() => setIsLoading(false), 1500);
    return () => clearTimeout(timer);
  }, []);

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const toggleFeatures = () => {
    setShowFullFeatures(!showFullFeatures);
  };

  const displayedFeatures = showFullFeatures ? features : features.slice(0, 4);

  return (
    <Box sx={{ 
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #0F0F0F 0%, #121212 100%)',
      color: 'white',
      overflow: 'hidden',
      position: 'relative',
    }}>
      {/* Background gradient effects */}
      <Box
        sx={{
          position: 'absolute',
          top: -100,
          left: -100,
          width: 300,
          height: 300,
          background: 'radial-gradient(circle, rgba(62, 237, 191, 0.2) 0%, rgba(0, 0, 0, 0) 70%)',
          borderRadius: '50%',
          filter: 'blur(60px)',
          zIndex: 0,
        }}
      />
      <Box
        sx={{
          position: 'absolute',
          top: '30%',
          right: -100,
          width: 250,
          height: 250,
          background: 'radial-gradient(circle, rgba(255, 64, 129, 0.15) 0%, rgba(0, 0, 0, 0) 70%)',
          borderRadius: '50%',
          filter: 'blur(60px)',
          zIndex: 0,
        }}
      />
      <Box
        sx={{
          position: 'absolute',
          bottom: '10%',
          left: '20%',
          width: 400,
          height: 400,
          background: 'radial-gradient(circle, rgba(83, 109, 254, 0.1) 0%, rgba(0, 0, 0, 0) 70%)',
          borderRadius: '50%',
          filter: 'blur(60px)',
          zIndex: 0,
        }}
      />

      {/* Navigation */}
      <Container maxWidth="xl" sx={{ position: 'relative', zIndex: 1 }}>
        <Box sx={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center', 
          py: 3,
          mb: { xs: 4, md: 10 }
        }}>
          <Logo />
          
          <Box sx={{ display: { xs: 'none', md: 'flex' }, gap: 2 }}>
            <Button 
              variant="text"
              onClick={() => navigate('/features')}
              sx={{ 
                color: 'white',
                fontWeight: 600,
              }}
            >
              Features
            </Button>
            <Button 
              variant="text"
              onClick={() => navigate('/solutions')}
              sx={{ 
                color: 'white',
                fontWeight: 600,
              }}
            >
              Solutions
            </Button>
            <Button 
              variant="text"
              onClick={() => navigate('/pricing')}
              sx={{ 
                color: 'white',
                fontWeight: 600,
              }}
            >
              Pricing
            </Button>
            <Button 
              variant="text"
              onClick={() => navigate('/about')}
              sx={{ 
                color: 'white',
                fontWeight: 600,
              }}
            >
              About
            </Button>
          </Box>
          
          <Box>
            <Button 
              variant="outlined" 
              onClick={() => navigate('/login')}
              sx={{ 
                mr: 2, 
                color: 'white',
                borderColor: 'rgba(255, 255, 255, 0.2)',
                borderRadius: 8,
                px: 3,
                py: 1,
                fontWeight: 600
              }}
            >
              Login
            </Button>
            <GlowButton
              variant="contained"
              disableElevation
              onClick={() => navigate('/register')}
              sx={{ 
                bgcolor: 'primary.main',
                color: 'black',
                borderRadius: 8,
                px: 3,
                py: 1,
                fontWeight: 600,
              }}
            >
              Get Started
            </GlowButton>
          </Box>
        </Box>

        {/* Hero Section */}
        <Grid container spacing={6} sx={{ mt: 6, mb: 10, alignItems: 'center' }}>
          <Grid item xs={12} md={6}>
            <Typography 
              variant="h1" 
              sx={{ 
                fontWeight: 800, 
                fontSize: { xs: '2.5rem', md: '3.5rem', lg: '4rem' },
                mb: 2,
                lineHeight: 1.2,
                background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.secondary.light} 100%)`,
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              AI-Powered Wealth Management for the Modern Investor
            </Typography>
            <Typography 
              variant="h6" 
              sx={{ 
                color: 'text.secondary', 
                mb: 4,
                maxWidth: '90%',
                fontWeight: 400
              }}
            >
              Harness the power of artificial intelligence to optimize your investment portfolio with institutional-grade tools previously only available to hedge funds and wealth managers.
            </Typography>
            <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', mb: 4 }}>
              <GlowButton
                variant="contained"
                size="large"
                disableElevation
                endIcon={<ArrowForward />}
                onClick={() => navigate('/register')}
                sx={{ 
                  bgcolor: 'primary.main',
                  color: 'black',
                  px: 4,
                  py: 1.5,
                  fontWeight: 600,
                  borderRadius: 8,
                }}
              >
                Get Started
              </GlowButton>
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 3 }}>
              <Box sx={{ display: 'flex' }}>
                {[1, 2, 3, 4, 5].map((item) => (
                  <Avatar 
                    key={item} 
                    src={`/avatars/user-${item}.jpg`} 
                    sx={{ 
                      width: 40, 
                      height: 40, 
                      border: '2px solid #121212',
                      ml: item > 1 ? -1 : 0 
                    }} 
                  />
                ))}
              </Box>
              <Box>
                <Typography variant="body2" sx={{ fontWeight: 600 }}>
                  Trusted by 25,000+ investors
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  {[1, 2, 3, 4, 5].map((star) => (
                    <Star key={star} sx={{ color: 'warning.main', fontSize: '1rem' }} />
                  ))}
                  <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>
                    4.9/5 from 1,200+ reviews
                  </Typography>
                </Box>
              </Box>
            </Box>
          </Grid>
          <Grid item xs={12} md={6}>
            <FloatingCard
              elevation={0}
              sx={{
                background: 'rgba(18, 18, 18, 0.8)',
                backdropFilter: 'blur(10px)',
                borderRadius: '24px',
                border: '1px solid rgba(50, 50, 50, 0.3)',
                p: 4,
                width: '100%',
                boxShadow: '0 20px 40px rgba(0, 0, 0, 0.3)',
                position: 'relative',
                overflow: 'hidden',
              }}
            >
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                  Portfolio Overview
                </Typography>
                <Chip 
                  label="Live" 
                  size="small" 
                  sx={{ 
                    bgcolor: 'error.main', 
                    color: 'black',
                    fontWeight: 600
                  }} 
                />
              </Box>
              
              {isLoading ? (
                <Skeleton variant="rectangular" width="100%" height={300} sx={{ mb: 3 }} />
              ) : (
                <Box sx={{ height: 300, mb: 3 }}>
                  <Line 
                    data={portfolioData}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: { display: false },
                        tooltip: {
                          mode: 'index',
                          intersect: false,
                          backgroundColor: 'rgba(30, 30, 30, 0.9)',
                          titleColor: 'white',
                          bodyColor: 'white',
                          borderColor: 'rgba(255, 255, 255, 0.1)',
                          borderWidth: 1,
                          padding: 12,
                          callbacks: {
                            label: (context) => {
                              return `$${context.parsed.y.toLocaleString()}`;
                            }
                          }
                        }
                      },
                      scales: {
                        x: {
                          grid: { color: 'rgba(255, 255, 255, 0.05)' },
                          ticks: { color: 'rgba(255, 255, 255, 0.7)' }
                        },
                        y: {
                          grid: { color: 'rgba(255, 255, 255, 0.05)' },
                          ticks: { 
                            color: 'rgba(255, 255, 255, 0.7)',
                            callback: (value) => `$${value / 1000}k`
                          }
                        }
                      },
                      elements: {
                        point: {
                          hoverRadius: 8,
                          hoverBorderWidth: 2
                        }
                      },
                      interaction: {
                        mode: 'nearest',
                        axis: 'x',
                        intersect: false
                      }
                    }}
                  />
                </Box>
              )}
              
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Paper
                    elevation={0}
                    sx={{
                      p: 2,
                      background: 'rgba(30, 30, 30, 0.6)',
                      border: '1px solid rgba(255, 255, 255, 0.1)',
                      borderRadius: '12px'
                    }}
                  >
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                      Total Portfolio Value
                    </Typography>
                    <Typography variant="h5" sx={{ fontWeight: 700 }}>
                      $142,568
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                      <TrendingUp sx={{ color: 'success.main', fontSize: '1rem', mr: 0.5 }} />
                      <Typography variant="body2" sx={{ color: 'success.main', fontWeight: 600 }}>
                        +5.2% this month
                      </Typography>
                    </Box>
                  </Paper>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Paper
                    elevation={0}
                    sx={{
                      p: 2,
                      background: 'rgba(30, 30, 30, 0.6)',
                      border: '1px solid rgba(255, 255, 255, 0.1)',
                      borderRadius: '12px'
                    }}
                  >
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                      Asset Allocation
                    </Typography>
                    {isLoading ? (
                      <Skeleton variant="rectangular" width="100%" height={60} />
                    ) : (
                      <Box sx={{ height: 60 }}>
                        <Pie 
                          data={assetAllocationData}
                          options={{
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {
                              legend: { display: false },
                              tooltip: {
                                backgroundColor: 'rgba(30, 30, 30, 0.9)',
                                titleColor: 'white',
                                bodyColor: 'white',
                                borderColor: 'rgba(255, 255, 255, 0.1)',
                                borderWidth: 1,
                                padding: 12
                              }
                            }
                          }}
                        />
                      </Box>
                    )}
                  </Paper>
                </Grid>
              </Grid>
            </FloatingCard>
          </Grid>
        </Grid>

        {/* Partners Section */}
        <Box sx={{ py: 6, textAlign: 'center' }}>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 4 }}>
            TRUSTED BY LEADING FINANCIAL INSTITUTIONS AND INVESTORS WORLDWIDE
          </Typography>
          <Grid container spacing={4} alignItems="center" justifyContent="center">
            {partners.map((partner, index) => (
              <Grid item xs={6} sm={4} md={2} key={index}>
                <Box
                  component="img"
                  src={partner.logo}
                  alt={partner.name}
                  sx={{ 
                    height: 30, 
                    filter: 'brightness(0) invert(1)',
                    opacity: 0.7,
                    transition: 'opacity 0.3s ease',
                    '&:hover': {
                      opacity: 1
                    }
                  }}
                />
              </Grid>
            ))}
          </Grid>
        </Box>

        {/* Features Section */}
        <Box sx={{ py: 10 }}>
          <Typography 
            variant="h2" 
            align="center" 
            sx={{ 
              mb: 2,
              fontWeight: 800,
              background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.secondary.light} 100%)`,
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            Institutional-Grade Tools at Your Fingertips
          </Typography>
          <Typography 
            variant="h6" 
            align="center" 
            sx={{ 
              mb: 8,
              color: 'text.secondary',
              maxWidth: 700,
              mx: 'auto',
              fontWeight: 400
            }}
          >
            Our platform combines cutting-edge AI with intuitive design to give you the tools you need to build and manage a sophisticated investment portfolio.
          </Typography>

          <Grid container spacing={4}>
            {displayedFeatures.map((feature, index) => (
              <Grid item xs={12} md={6} key={index}>
                <Grow in={true} timeout={index * 200}>
                  <FeatureCard
                    elevation={0}
                    sx={{
                      p: 4,
                      height: '100%',
                      display: 'flex',
                      flexDirection: 'column',
                      background: 'rgba(30, 30, 30, 0.6)',
                      border: '1px solid rgba(255, 255, 255, 0.1)',
                      transition: 'all 0.3s ease',
                    }}
                  >
                    <Box sx={{ color: 'primary.main', mb: 2 }}>
                      {feature.icon}
                    </Box>
                    <Typography variant="h5" sx={{ mb: 2, fontWeight: 700 }}>
                      {feature.title}
                    </Typography>
                    <Typography variant="body1" color="text.secondary" sx={{ mb: 2 }}>
                      {feature.description}
                    </Typography>
                    <Typography variant="body2" sx={{ 
                      mt: 'auto',
                      color: 'primary.main',
                      fontWeight: 600,
                      display: 'flex',
                      alignItems: 'center'
                    }}>
                      <VerifiedUser sx={{ fontSize: '1rem', mr: 1 }} />
                      {feature.stats}
                    </Typography>
                  </FeatureCard>
                </Grow>
              </Grid>
            ))}
          </Grid>
        </Box>
                {/* Performance Section */}
                <Box sx={{ py: 10, background: 'rgba(20, 20, 20, 0.5)', borderRadius: '24px', mb: 10 }}>
          <Container maxWidth="lg">
            <Grid container spacing={6} alignItems="center">
              <Grid item xs={12} md={6}>
                <Typography 
                  variant="h2" 
                  sx={{ 
                    mb: 3,
                    fontWeight: 800,
                    background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.secondary.light} 100%)`,
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                  }}
                >
                  Outperform the Market Consistently
                </Typography>
                <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
                  Our AI-driven strategies have consistently outperformed major benchmarks across all market conditions since our inception.
                </Typography>
                
                <Box sx={{ mb: 4 }}>
                  {performanceMetrics.slice(0, 3).map((metric, index) => (
                    <PerformanceMetricItem key={index} metric={metric} />
                  ))}
                </Box>
                
                <Button 
                  variant="outlined" 
                  endIcon={<ArrowForward />}
                  onClick={() => navigate('/performance')}
                  sx={{ 
                    borderColor: 'rgba(255, 255, 255, 0.2)',
                    color: 'white',
                    px: 4,
                    py: 1.5,
                    fontWeight: 600,
                    borderRadius: 8,
                  }}
                >
                  View Full Performance
                </Button>
              </Grid>
              <Grid item xs={12} md={6}>
                <Paper
                  elevation={0}
                  sx={{
                    p: 3,
                    background: 'rgba(30, 30, 30, 0.6)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '16px'
                  }}
                >
                  {isLoading ? (
                    <Skeleton variant="rectangular" width="100%" height={400} />
                  ) : (
                    <Box sx={{ height: 400 }}>
                      <Bar 
                        data={performanceData}
                        options={{
                          responsive: true,
                          maintainAspectRatio: false,
                          plugins: {
                            legend: { display: false },
                            tooltip: {
                              backgroundColor: 'rgba(30, 30, 30, 0.9)',
                              titleColor: 'white',
                              bodyColor: 'white',
                              borderColor: 'rgba(255, 255, 255, 0.1)',
                              borderWidth: 1,
                              padding: 12
                            }
                          },
                          scales: {
                            x: {
                              grid: { color: 'rgba(255, 255, 255, 0.05)' },
                              ticks: { color: 'rgba(255, 255, 255, 0.7)' }
                            },
                            y: {
                              grid: { color: 'rgba(255, 255, 255, 0.05)' },
                              ticks: { 
                                color: 'rgba(255, 255, 255, 0.7)',
                                callback: (value) => `${value}%`
                              }
                            }
                          }
                        }}
                      />
                    </Box>
                  )}
                </Paper>
              </Grid>
            </Grid>
          </Container>
        </Box>

        {/* Asset Allocation & News */}
        <Box sx={{ py: 10, mb: 10 }}>
          <Grid container spacing={6}>
            <Grid item xs={12} md={6}>
              <Typography 
                variant="h3" 
                sx={{ 
                  mb: 4,
                  fontWeight: 700,
                }}
              >
                Smart Asset Allocation
              </Typography>
              
              <Paper
                elevation={0}
                sx={{
                  p: 3,
                  mb: 4,
                  background: 'rgba(30, 30, 30, 0.6)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  borderRadius: '16px'
                }}
              >
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    Current Allocation
                  </Typography>
                  <Chip 
                    label="AI-Optimized" 
                    size="small" 
                    sx={{ 
                      bgcolor: 'primary.dark', 
                      color: 'black',
                      fontWeight: 600
                    }} 
                  />
                </Box>
                
                {assetClasses.map((asset, index) => (
                  <AssetClassItem key={index} asset={asset} />
                ))}
              </Paper>
              
              <Button 
                variant="contained"
                endIcon={<ArrowForward />}
                onClick={() => navigate('/register')}
                sx={{ 
                  bgcolor: 'primary.main',
                  color: 'black',
                  px: 4,
                  py: 1.5,
                  fontWeight: 600,
                  borderRadius: 8,
                }}
              >
                Get Started
              </Button>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Typography 
                variant="h3" 
                sx={{ 
                  mb: 4,
                  fontWeight: 700,
                }}
              >
                Market News & Insights
              </Typography>
              
              <Paper
                elevation={0}
                sx={{
                  p: 3,
                  background: 'rgba(30, 30, 30, 0.6)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  borderRadius: '16px'
                }}
              >
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    Latest Updates
                  </Typography>
                  <TextField
                    size="small"
                    placeholder="Search news..."
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <Search sx={{ color: 'text.secondary' }} />
                        </InputAdornment>
                      ),
                      sx: {
                        color: 'white',
                        borderRadius: 8,
                        bgcolor: 'rgba(255, 255, 255, 0.1)',
                        '& fieldset': {
                          border: 'none'
                        }
                      }
                    }}
                  />
                </Box>
                
                {newsUpdates.map((news, index) => (
                  <NewsUpdateItem key={index} news={news} />
                ))}
                
                <Button 
                  fullWidth
                  endIcon={<ArrowForward />}
                  sx={{ 
                    mt: 2,
                    color: 'primary.main',
                    fontWeight: 600,
                  }}
                >
                  View All Market Updates
                </Button>
              </Paper>
            </Grid>
          </Grid>
        </Box>

        {/* Testimonials */}
        <Box sx={{ py: 10, mb: 10 }}>
          <Typography 
            variant="h2" 
            align="center" 
            sx={{ 
              mb: 8,
              fontWeight: 800,
              background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.secondary.light} 100%)`,
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            What Our Clients Say
          </Typography>
          
          <Grid container spacing={4}>
            {testimonials.map((testimonial, index) => (
              <Grid item xs={12} md={6} lg={3} key={index}>
                <Zoom in={true} timeout={index * 200}>
                  <Paper
                    elevation={0}
                    sx={{
                      p: 4,
                      height: '100%',
                      display: 'flex',
                      flexDirection: 'column',
                      background: 'rgba(30, 30, 30, 0.6)',
                      border: '1px solid rgba(255, 255, 255, 0.1)',
                      borderRadius: '16px',
                      transition: 'all 0.3s ease',
                      '&:hover': {
                        borderColor: 'rgba(62, 237, 191, 0.3)'
                      }
                    }}
                  >
                    <Box sx={{ display: 'flex', mb: 3 }}>
                      <Avatar sx={{ bgcolor: 'primary.dark', color: 'black', mr: 2, fontWeight: 700 }}>
                        {testimonial.initials}
                      </Avatar>
                      <Box>
                        <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                          {testimonial.name}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {testimonial.position}
                        </Typography>
                      </Box>
                    </Box>
                    
                    <Typography variant="body1" sx={{ mb: 3, flexGrow: 1, fontStyle: 'italic' }}>
                      "{testimonial.quote}"
                    </Typography>
                    
                    <RatingStars rating={testimonial.rating} />
                  </Paper>
                </Zoom>
              </Grid>
            ))}
          </Grid>
        </Box>

        {/* CTA Section */}
        <Box 
          sx={{ 
            py: 10,
            mb: 10,
            background: 'linear-gradient(135deg, rgba(62, 237, 191, 0.1) 0%, rgba(83, 109, 254, 0.1) 100%)',
            borderRadius: '24px',
            border: '1px solid rgba(62, 237, 191, 0.1)',
            textAlign: 'center',
            position: 'relative',
            overflow: 'hidden'
          }}
        >
          <Box
            sx={{
              position: 'absolute',
              top: -100,
              right: -100,
              width: 300,
              height: 300,
              background: 'radial-gradient(circle, rgba(62, 237, 191, 0.2) 0%, rgba(0, 0, 0, 0) 70%)',
              borderRadius: '50%',
              filter: 'blur(60px)',
              zIndex: 0,
            }}
          />
          <Box
            sx={{
              position: 'absolute',
              bottom: -100,
              left: -100,
              width: 300,
              height: 300,
              background: 'radial-gradient(circle, rgba(83, 109, 254, 0.2) 0%, rgba(0, 0, 0, 0) 70%)',
              borderRadius: '50%',
              filter: 'blur(60px)',
              zIndex: 0,
            }}
          />
          
          <Container maxWidth="md" sx={{ position: 'relative', zIndex: 1 }}>
            <Typography 
              variant="h2" 
              sx={{ 
                mb: 3,
                fontWeight: 800,
                background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.secondary.light} 100%)`,
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              Ready to Transform Your Wealth?
            </Typography>
            <Typography variant="h6" color="text.secondary" sx={{ mb: 6, maxWidth: 600, mx: 'auto' }}>
              Join thousands of investors who are already using IntelliWealth to grow their portfolios smarter and faster.
            </Typography>
            
            <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', flexWrap: 'wrap' }}>
              <GlowButton
                variant="contained"
                size="large"
                disableElevation
                endIcon={<ArrowForward />}
                onClick={() => navigate('/register')}
                sx={{ 
                  bgcolor: 'primary.main',
                  color: 'black',
                  px: 6,
                  py: 1.5,
                  fontWeight: 600,
                  borderRadius: 8,
                }}
              >
                Get Started
              </GlowButton>
            </Box>
          </Container>
        </Box>

        {/* Footer */}
        <Box sx={{ py: 8, borderTop: '1px solid rgba(255, 255, 255, 0.1)' }}>
          <Container maxWidth="xl">
            <Grid container spacing={6}>
              <Grid item xs={12} md={3}>
                <Box sx={{ mb: 2 }}>
                  <Logo />
                </Box>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                  AI-powered wealth management for the modern investor.
                </Typography>
                <Box sx={{ display: 'flex', gap: 2 }}>
                  <IconButton sx={{ color: 'text.secondary' }}>
                    <Twitter />
                  </IconButton>
                  <IconButton sx={{ color: 'text.secondary' }}>
                    <LinkedIn />
                  </IconButton>
                  <IconButton sx={{ color: 'text.secondary' }}>
                    <Facebook />
                  </IconButton>
                  <IconButton 
                    sx={{ color: 'text.secondary' }}
                    onClick={() => window.open('https://www.instagram.com/intelliwealth.top/', '_blank')}
                  >
                    <Instagram />
                  </IconButton>
                  <IconButton sx={{ color: 'text.secondary' }}>
                    <GitHub />
                  </IconButton>
                  <IconButton sx={{ color: 'text.secondary' }}>
                    <YouTube />
                  </IconButton>
                </Box>
              </Grid>
              
              <Grid item xs={6} md={2}>
                <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 600 }}>
                  Product
                </Typography>
                <List dense>
                  {['Features', 'Pricing', 'Case Studies', 'Updates'].map((text) => (
                    <ListItem key={text} disableGutters>
                      <ListItemText 
                        primary={text} 
                        primaryTypographyProps={{ 
                          variant: 'body2',
                          color: 'text.secondary',
                          sx: { '&:hover': { color: 'white', cursor: 'pointer' } }
                        }} 
                      />
                    </ListItem>
                  ))}
                </List>
              </Grid>
              
              <Grid item xs={6} md={2}>
                <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 600 }}>
                  Company
                </Typography>
                <List dense>
                  {[
                    { text: 'Careers', path: '/careers' },
                    { text: 'News', path: '/news' },
                    { text: 'Contact', path: '/contact' }
                  ].map((item) => (
                    <ListItem key={item.text} disableGutters>
                      <ListItemText 
                        primary={item.text} 
                        onClick={() => navigate(item.path)}
                        primaryTypographyProps={{ 
                          variant: 'body2',
                          color: 'text.secondary',
                          sx: { 
                            '&:hover': { 
                              color: 'white', 
                              cursor: 'pointer' 
                            } 
                          }
                        }} 
                      />
                    </ListItem>
                  ))}
                </List>
              </Grid>
              
              <Grid item xs={6} md={2}>
                <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 600 }}>
                  Resources
                </Typography>
                <List dense>
                  {['Blog', 'Help Center', 'Tutorials', 'API Docs'].map((text) => (
                    <ListItem key={text} disableGutters>
                      <ListItemText 
                        primary={text} 
                        primaryTypographyProps={{ 
                          variant: 'body2',
                          color: 'text.secondary',
                          sx: { '&:hover': { color: 'white', cursor: 'pointer' } }
                        }} 
                      />
                    </ListItem>
                  ))}
                </List>
              </Grid>
              
              <Grid item xs={6} md={3}>
                <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 600 }}>
                  Get in Touch
                </Typography>
                <List dense>
                  <ListItem disableGutters>
                    <ListItemIcon sx={{ minWidth: 32, color: 'text.secondary' }}>
                      <Email fontSize="small" />
                    </ListItemIcon>
                    <ListItemText 
                      primary="info@intelliwealth.com" 
                      primaryTypographyProps={{ variant: 'body2', color: 'text.secondary' }} 
                    />
                  </ListItem>
                  <ListItem disableGutters>
                    <ListItemIcon sx={{ minWidth: 32, color: 'text.secondary' }}>
                      <Phone fontSize="small" />
                    </ListItemIcon>
                    <ListItemText 
                      primary="+91 7988653936" 
                      primaryTypographyProps={{ variant: 'body2', color: 'text.secondary' }} 
                    />
                  </ListItem>
                  <ListItem disableGutters>
                    <ListItemIcon sx={{ minWidth: 32, color: 'text.secondary' }}>
                      <LocationOn fontSize="small" />
                    </ListItemIcon>
                    <ListItemText 
                      primary="IIT Roorkee, Haridwar, Uttrakhand, INDIA" 
                      primaryTypographyProps={{ variant: 'body2', color: 'text.secondary' }} 
                    />
                  </ListItem>
                </List>
              </Grid>
            </Grid>
            
            <Divider sx={{ my: 6, borderColor: 'rgba(255, 255, 255, 0.05)' }} />
            
            <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, justifyContent: 'space-between' }}>
              <Typography variant="body2" color="text.secondary">
                © 2023 IntelliWealth. All rights reserved.
              </Typography>
              <Box sx={{ display: 'flex', gap: 3, mt: { xs: 2, md: 0 } }}>
                <Typography variant="body2" color="text.secondary" sx={{ '&:hover': { color: 'white', cursor: 'pointer' } }}>
                  Privacy Policy
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ '&:hover': { color: 'white', cursor: 'pointer' } }}>
                  Terms of Service
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ '&:hover': { color: 'white', cursor: 'pointer' } }}>
                  Cookies
                </Typography>
              </Box>
            </Box>
          </Container>
        </Box>
      </Container>
    </Box>
  );
}