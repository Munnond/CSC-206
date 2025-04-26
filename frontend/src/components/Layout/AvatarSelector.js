// src/components/Layout/AvatarSelector.js
'use client'

import React from 'react';
import { Avatar, Box } from '@mui/material';

const AvatarSelector = ({ name }) => {
  return (
    <Box 
      sx={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        mb: 2
      }}
    >
      <Avatar
        sx={{
          width: 80,
          height: 80,
          bgcolor: '#3EEFBF', // Mint green color from your design
          color: '#000000', // Black text
          fontSize: '2rem',
          fontWeight: 600,
          border: '2px solid #3EEFBF',
          boxShadow: '0 4px 8px rgba(0,0,0,0.2)'
        }}
      >
        {name.charAt(0).toUpperCase()}
      </Avatar>
    </Box>
  );
};

export default AvatarSelector;