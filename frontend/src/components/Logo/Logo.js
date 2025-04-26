import React from 'react';
import { Box, Typography, useTheme } from '@mui/material';
import { Link } from 'react-router-dom';
import { Timeline } from '@mui/icons-material';

export default function Logo({ sx = {} }) {
  const theme = useTheme();
  
  return (
    <Link to="/" style={{ textDecoration: 'none' }}>
      <Box sx={{ 
        display: 'flex', 
        alignItems: 'center',
        ...sx
      }}>
        <Timeline sx={{ color: 'primary.main', mr: 1, fontSize: '2rem' }} />
        <Typography 
          variant="h5" 
          component="div"
          sx={{ 
            fontWeight: 700,
            background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.secondary.light} 100%)`,
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}
        >
          IntelliWealth
        </Typography>
      </Box>
    </Link>
  );
}