'use client'

import React, { useEffect, useState, useContext } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Box, Typography, CircularProgress, Container, Paper, Alert } from '@mui/material';
import { UserContext } from '../context/UserContext';

export default function GoogleCallback() {
  const location = useLocation();
  const navigate = useNavigate();
  const { loginWithGoogle } = useContext(UserContext);
  const [status, setStatus] = useState({ loading: true, error: null });

  useEffect(() => {
    const processAuth = async () => {
      try {
        // Extract token from URL
        const params = new URLSearchParams(location.hash.substring(1));
        const token = params.get('access_token');

        if (!token) {
          throw new Error('No access token found in the URL');
        }

        // Send the token to our backend
        const result = await loginWithGoogle(token);
        
        if (result.success) {
          // Redirect to dashboard after successful login
          setTimeout(() => {
            navigate('/dashboard');
          }, 1500);
        } else {
          throw new Error(typeof result.error === 'string' ? result.error : 'Failed to authenticate with Google');
        }
      } catch (error) {
        console.error('Google auth callback error:', error);
        setStatus({
          loading: false,
          error: error.message || 'Authentication failed'
        });
      }
    };

    processAuth();
  }, [location, loginWithGoogle, navigate]);

  return (
    <Container maxWidth="sm" sx={{ mt: 8 }}>
      <Paper elevation={3} sx={{ p: 4, textAlign: 'center' }}>
        <Typography variant="h5" gutterBottom>
          Google Authentication
        </Typography>

        {status.loading ? (
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', mt: 4 }}>
            <CircularProgress />
            <Typography variant="body1" sx={{ mt: 2 }}>
              Completing authentication...
            </Typography>
          </Box>
        ) : status.error ? (
          <Alert severity="error" sx={{ mt: 2 }}>
            {status.error}
          </Alert>
        ) : (
          <Alert severity="success" sx={{ mt: 2 }}>
            Authentication successful! Redirecting...
          </Alert>
        )}
      </Paper>
    </Container>
  );
} 