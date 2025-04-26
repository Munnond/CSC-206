import React from 'react';
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
  Business,
  Person,
  Groups,
  AccountBalance,
  TrendingUp,
  VerifiedUser,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import Logo from '../components/Logo/Logo';

const solutions = [
  {
    title: 'Personal Investing',
    description: 'Perfect for individual investors looking to grow their wealth through smart, AI-driven portfolio management.',
    icon: <Person fontSize="large" />,
    features: [
      'Personalized portfolio management',
      'Automated rebalancing',
      'Tax optimization',
      'Mobile app access'
    ],
    minInvestment: 'USD 5,000',
    targetAudience: 'Individual Investors'
  },
  {
    title: 'Professional Trading',
    description: 'Advanced tools and features for professional traders who need sophisticated analysis and execution capabilities.',
    icon: <TrendingUp fontSize="large" />,
    features: [
      'Advanced market analytics',
      'API access',
      'Real-time trading signals',
      'Custom strategy development'
    ],
    minInvestment: 'USD 25,000',
    targetAudience: 'Professional Traders'
  },
  {
    title: 'Wealth Management',
    description: 'Comprehensive wealth management solutions for high-net-worth individuals and family offices.',
    icon: <AccountBalance fontSize="large" />,
    features: [
      'Dedicated portfolio manager',
      'Estate planning',
      'Tax strategy',
      'Private banking services'
    ],
    minInvestment: 'USD 100,000',
    targetAudience: 'High-Net-Worth Individuals'
  },
  {
    title: 'Institutional',
    description: 'Enterprise-grade solutions for financial institutions, hedge funds, and large organizations.',
    icon: <Business fontSize="large" />,
    features: [
      'Custom API integration',
      'White-label solutions',
      'Regulatory compliance',
      'Institutional-grade security'
    ],
    minInvestment: 'Custom',
    targetAudience: 'Financial Institutions'
  }
];

export default function Solutions() {
  const theme = useTheme();
  const navigate = useNavigate();

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

      <Container maxWidth="lg">
        {/* Header Section */}
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
          Investment Solutions
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
          Tailored investment solutions to meet your specific needs and goals
        </Typography>

        {/* Solutions Grid */}
        <Grid container spacing={4}>
          {solutions.map((solution, index) => (
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
                  {solution.icon}
                </Box>
                <Typography variant="h5" sx={{ mb: 2, fontWeight: 700 }}>
                  {solution.title}
                </Typography>
                <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
                  {solution.description}
                </Typography>
                
                {/* Features List */}
                <Box sx={{ mb: 3, flexGrow: 1 }}>
                  {solution.features.map((feature, idx) => (
                    <Box key={idx} sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                      <VerifiedUser sx={{ fontSize: '0.8rem', mr: 1, color: 'primary.main' }} />
                      <Typography variant="body2" color="text.secondary">
                        {feature}
                      </Typography>
                    </Box>
                  ))}
                </Box>

                {/* Solution Details */}
                <Box sx={{ mt: 'auto' }}>
                  <Chip
                    label={`Min. Investment: ${solution.minInvestment}`}
                    sx={{
                      bgcolor: 'primary.dark',
                      color: 'black',
                      fontWeight: 600,
                      mb: 2
                    }}
                  />
                  <Button
                    variant="outlined"
                    fullWidth
                    sx={{
                      borderColor: 'rgba(62, 237, 191, 0.3)',
                      color: 'white',
                      '&:hover': {
                        borderColor: 'primary.main',
                        bgcolor: 'rgba(62, 237, 191, 0.1)'
                      }
                    }}
                  >
                    Learn More
                  </Button>
                </Box>
              </Paper>
            </Grid>
          ))}
        </Grid>

        {/* CTA Section */}
        <Box sx={{ mt: 12, textAlign: 'center' }}>
          <Typography variant="h4" sx={{ mb: 4, fontWeight: 700 }}>
            Ready to Start Investing?
          </Typography>
          <Button
            variant="contained"
            size="large"
            onClick={() => navigate('/register')}
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
            Get Started Now
          </Button>
        </Box>
      </Container>
    </Box>
  );
}
