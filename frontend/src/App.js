// src/App.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { UserProvider } from './context/UserContext';

// Pages
import Dashboard from './pages/Dashboard';
import Login from './pages/Login';
import Register from './pages/Register';
import Sentiment from './pages/Sentiment';
import Portfolio from './pages/Portfolio';
import AssetAllocation from './pages/AssetAllocation';
import MarketInsights from './pages/MarketInsights';
import LandingPage from './pages/LandingPage';
import PortfolioOptimizer from './pages/PortfolioOptimizer';
import FinancialReportGenerator from './pages/FinancialReportGenerator';
import StockNews from './pages/StockNews';
import ProfilePage from './pages/ProfilePage';
import SettingsPage from './pages/SettingsPage';
import About from './pages/About';
import Features from './pages/Features';
import Solutions from './pages/Solutions';
import Pricing from './pages/Pricing';

// Components
import Layout from './components/Layout/Layout';

// Create a theme
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#3EEDBF',
      light: '#64FFDA',
      dark: '#00BFA5',
    },
    secondary: {
      main: '#FF4081',
      light: '#FF80AB',
      dark: '#F50057',
    },
    background: {
      default: '#0A0A0A',
      paper: 'rgba(25, 25, 25, 0.8)',
    },
    text: {
      primary: '#FFFFFF',
      secondary: 'rgba(255, 255, 255, 0.7)',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 600,
      letterSpacing: '-0.025em',
    },
    h2: {
      fontWeight: 600,
      letterSpacing: '-0.025em',
    },
    h3: {
      fontWeight: 500,
      letterSpacing: '-0.015em',
    },
    h4: {
      fontWeight: 500,
      letterSpacing: '-0.01em',
    },
    button: {
      textTransform: 'none',
      fontWeight: 500,
    },
  },
  shape: {
    borderRadius: 16,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          padding: '10px 24px',
          backdropFilter: 'blur(10px)',
          transition: 'all 0.2s ease-in-out',
        },
        containedPrimary: {
          backgroundColor: '#3EEDBF',
          color: '#0A0A0A',
          '&:hover': {
            backgroundColor: '#64FFDA',
            boxShadow: '0 0 15px rgba(62, 237, 191, 0.3)',
          },
        },
        outlinedPrimary: {
          borderColor: 'rgba(255, 255, 255, 0.2)',
          color: 'white',
          '&:hover': {
            borderColor: 'rgba(255, 255, 255, 0.5)',
            backgroundColor: 'rgba(255, 255, 255, 0.05)',
          },
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 12,
            backdropFilter: 'blur(10px)',
            background: 'rgba(35, 35, 35, 0.5)',
            transition: 'all 0.2s ease-in-out',
            '&:hover': {
              background: 'rgba(45, 45, 45, 0.5)',
            },
            '&.Mui-focused': {
              boxShadow: '0 0 10px rgba(62, 237, 191, 0.2)',
            },
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backdropFilter: 'blur(10px)',
          background: 'rgba(25, 25, 25, 0.8)',
          borderRadius: 24,
        },
      },
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <UserProvider>
        <Router>
          <Routes>
            <Route path="/landing" element={<LandingPage />} />
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
            <Route path="/dashboard" element={<Layout />}>
              <Route index element={<Dashboard />} />
              <Route path="Sentiment" element={<Sentiment />} />
              <Route path="portfolio" element={<Portfolio />} />
              <Route path="asset-allocation" element={<AssetAllocation />} />
              <Route path="market-insights" element={<MarketInsights />} />
              <Route path="portfolio-optimizer" element={<PortfolioOptimizer/>} />
              <Route path="financial-report" element={<FinancialReportGenerator />} />
              <Route path="stock-news" element={<StockNews />} />
              <Route path="profile" element={<ProfilePage />} />
              <Route path="settings" element={<SettingsPage />} />
            </Route>
            <Route path="/about" element={<About />} />
            <Route path="/features" element={<Features />} />
            <Route path="/solutions" element={<Solutions />} />
            <Route path="/pricing" element={<Pricing />} />
            <Route path="/" element={<Navigate replace to="/landing" />} />
          </Routes>
        </Router>
      </UserProvider>
    </ThemeProvider>
  );
}

export default App;