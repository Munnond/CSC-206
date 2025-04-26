import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Typography,
  Grid,
  Paper,
  Button,
  useTheme,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
} from '@mui/material';
import { Check, Star } from '@mui/icons-material';
import Logo from '../components/Logo/Logo';

const pricingPlans = [
  {
    title: 'Starter',
    price: 'USD 29',
    period: '/month',
    description: 'Perfect for beginners starting their investment journey',
    features: [
      'Basic portfolio management',
      'Market analysis tools',
      'Up to 5 investment strategies',
      'Email support',
      'Mobile app access'
    ],
    buttonText: 'Start Free Trial',
    isPopular: false,
    action: 'register'
  },
  {
    title: 'Professional',
    price: 'USD 99',
    period: '/month',
    description: 'Advanced features for serious investors',
    features: [
      'Advanced portfolio management',
      'Real-time market analytics',
      'Unlimited investment strategies',
      'Priority support',
      'API access',
      'Custom reports',
      'Tax optimization'
    ],
    buttonText: 'Get Started',
    isPopular: true,
    action: 'register'
  },
  {
    title: 'Enterprise',
    price: 'Custom',
    period: '',
    description: 'Tailored solutions for institutions and large organizations',
    features: [
      'Custom portfolio solutions',
      'Dedicated account manager',
      'White-label options',
      '24/7 premium support',
      'Custom API integration',
      'Advanced security features',
      'Regulatory compliance',
      'Team collaboration tools'
    ],
    buttonText: 'Contact Sales',
    isPopular: false,
    action: 'about'
  }
];

export default function Pricing() {
  const theme = useTheme();
  const navigate = useNavigate();

  const handlePlanClick = (action) => {
    if (action === 'register') {
      navigate('/register');
    } else if (action === 'about') {
      navigate('/about');
    }
  };

  return (
    <Box sx={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #0F0F0F 0%, #121212 100%)',
      color: 'white',
      pt: 4,
      pb: 12
    }}>
      <Container maxWidth="lg" sx={{ mb: 6 }}>
        <Logo />
      </Container>

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
          Transparent Pricing
        </Typography>
        <Typography
          variant="h6"
          align="center"
          color="text.secondary"
          sx={{ mb: 8, maxWidth: 800, mx: 'auto' }}
        >
          Choose the plan that best fits your investment needs
        </Typography>

        <Grid container spacing={4} alignItems="stretch">
          {pricingPlans.map((plan, index) => (
            <Grid item xs={12} md={4} key={index}>
              <Paper
                elevation={0}
                sx={{
                  p: 4,
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  background: plan.isPopular ? 'rgba(62, 237, 191, 0.1)' : 'rgba(30, 30, 30, 0.6)',
                  border: `1px solid ${plan.isPopular ? 'rgba(62, 237, 191, 0.3)' : 'rgba(255, 255, 255, 0.1)'}`,
                  borderRadius: '16px',
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    transform: 'translateY(-5px)',
                    borderColor: 'rgba(62, 237, 191, 0.3)'
                  }
                }}
              >
                {plan.isPopular && (
                  <Chip
                    icon={<Star sx={{ color: 'black !important' }} />}
                    label="Most Popular"
                    sx={{
                      bgcolor: 'primary.main',
                      color: 'black',
                      fontWeight: 600,
                      mb: 2
                    }}
                  />
                )}
                
                <Typography variant="h5" sx={{ mb: 2, fontWeight: 700 }}>
                  {plan.title}
                </Typography>
                
                <Box sx={{ mb: 3 }}>
                  <Typography variant="h3" component="span" sx={{ fontWeight: 800 }}>
                    {plan.price}
                  </Typography>
                  <Typography variant="body1" component="span" color="text.secondary">
                    {plan.period}
                  </Typography>
                </Box>

                <Typography color="text.secondary" sx={{ mb: 3 }}>
                  {plan.description}
                </Typography>

                <List sx={{ mb: 4, flexGrow: 1 }}>
                  {plan.features.map((feature, idx) => (
                    <ListItem key={idx} disableGutters>
                      <ListItemIcon>
                        <Check sx={{ color: 'primary.main' }} />
                      </ListItemIcon>
                      <ListItemText
                        primary={feature}
                        primaryTypographyProps={{
                          color: 'text.secondary'
                        }}
                      />
                    </ListItem>
                  ))}
                </List>

                <Button
                  variant={plan.isPopular ? 'contained' : 'outlined'}
                  fullWidth
                  onClick={() => handlePlanClick(plan.action)}
                  sx={{
                    py: 1.5,
                    fontWeight: 600,
                    ...(plan.isPopular ? {
                      bgcolor: 'primary.main',
                      color: 'black',
                      '&:hover': {
                        bgcolor: 'primary.dark'
                      }
                    } : {
                      borderColor: 'rgba(255, 255, 255, 0.2)',
                      color: 'white',
                      '&:hover': {
                        borderColor: 'rgba(62, 237, 191, 0.3)',
                        backgroundColor: 'rgba(62, 237, 191, 0.1)'
                      }
                    })
                  }}
                >
                  {plan.buttonText}
                </Button>
              </Paper>
            </Grid>
          ))}
        </Grid>
      </Container>
    </Box>
  );
}
