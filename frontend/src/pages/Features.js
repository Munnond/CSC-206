import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Typography,
  Grid,
  Paper,
  useTheme,
  Button,
  Chip,
  Avatar,
} from '@mui/material';
import {
  ShowChart,
  PieChart,
  Security,
  AutoGraph,
  VerifiedUser,
} from '@mui/icons-material';
import Logo from '../components/Logo/Logo';

// Features data
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
  },
  {
    title: 'Advanced Risk Assessment',
    description: 'Comprehensive risk analysis using Monte Carlo simulations and stress testing to ensure your portfolio can withstand market volatility.',
    icon: <Security fontSize="large" />,
    stats: 'Identifies 95% of potential risk factors'
  },
  {
    title: 'Real-Time Market Insights',
    description: 'Actionable intelligence powered by natural language processing analyzing thousands of news sources, earnings reports, and economic indicators.',
    icon: <AutoGraph fontSize="large" />,
    stats: 'Processes 10,000+ data points per second'
  }
];

export default function Features() {
  const theme = useTheme();
  const navigate = useNavigate();

  const handleStartInvesting = () => {
    navigate('/register');
  };

  return (
    <Box sx={{ 
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #0F0F0F 0%, #121212 100%)',
      color: 'white',
      pt: 4,
      pb: 12
    }}>
      {/* Logo Section */}
      <Container maxWidth="lg" sx={{ mb: 6 }}>
        <Logo />
      </Container>

      {/* Header Section */}
      <Container maxWidth="lg">
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
          Powerful Features for Smart Investing
        </Typography>
        <Typography 
          variant="h5" 
          align="center" 
          sx={{ 
            mb: 8,
            color: 'text.secondary',
            maxWidth: 800,
            mx: 'auto'
          }}
        >
          Discover how our AI-powered platform revolutionizes investment management with cutting-edge features
        </Typography>

        {/* Main Features Grid */}
        <Grid container spacing={4}>
          {features.map((feature, index) => (
            <Grid item xs={12} md={6} key={index}>
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
                    transform: 'translateY(-5px)',
                    borderColor: 'rgba(62, 237, 191, 0.3)',
                    boxShadow: '0 8px 24px rgba(62, 237, 191, 0.1)'
                  }
                }}
              >
                <Box sx={{ color: 'primary.main', mb: 2 }}>
                  {feature.icon}
                </Box>
                <Typography variant="h5" sx={{ mb: 2, fontWeight: 700 }}>
                  {feature.title}
                </Typography>
                <Typography variant="body1" color="text.secondary" sx={{ mb: 2, flexGrow: 1 }}>
                  {feature.description}
                </Typography>
                <Chip
                  icon={<VerifiedUser sx={{ fontSize: '1rem' }} />}
                  label={feature.stats}
                  sx={{
                    bgcolor: 'primary.dark',
                    color: 'black',
                    fontWeight: 600,
                    alignSelf: 'flex-start'
                  }}
                />
              </Paper>
            </Grid>
          ))}
        </Grid>

        {/* CTA Section */}
        <Box sx={{ mt: 12, textAlign: 'center' }}>
          <Button
            variant="contained"
            size="large"
            onClick={handleStartInvesting}
            sx={{
              bgcolor: 'primary.main',
              color: 'black',
              px: 6,
              py: 2,
              fontSize: '1.1rem',
              fontWeight: 600,
              borderRadius: 8,
              '&:hover': {
                bgcolor: 'primary.dark'
              }
            }}
          >
            Start Investing Now
          </Button>
        </Box>
      </Container>
    </Box>
  );
}
