// frontend/src/pages/About.js
import React from 'react';
import {
  Box,
  Container,
  Typography,
  Grid,
  Paper,
  Divider,
  useTheme,
  Avatar
} from '@mui/material';
import { Timeline } from '@mui/icons-material';
import AvatarSelector from '../components/Layout/AvatarSelector';
import Logo from '../components/Logo/Logo';

const teamMembers = [
  {
    name: "Shiv Shakti Kumar",
    role: "B.Tech CSE Student at IIT Roorkee",
    enrollmentNumber: "23114091",
    mobileNumber: "8290957549",
    gender: "male",
    avatar: "/avatars/male1.png"
  },
  {
    name: "Nisarg Prajapati",
    role: "B.Tech CSE Student at IIT Roorkee",
    enrollmentNumber: "23114073",
    mobileNumber: "7041882433",
    gender: "male",
    avatar: "/avatars/male1.png"
  },
  {
    name: "Samrat Middha",
    role: "B.Tech Student at IIT Roorkee",
    enrollmentNumber: "21117109",
    mobileNumber: "8146812840",
    gender: "male",
    avatar: "/avatars/male2.png"
  },
  {
    name: "Kaustubh Dwivedi",
    role: "B.Tech Student at IIT Roorkee",
    enrollmentNumber: "22117066",
    mobileNumber: "7080255488",
    gender: "male",
    avatar: "/avatars/male2.png"
  },
  {
    name: "Vanjale Pranjal Jeetendra",
    role: "B.Tech Student at IIT Roorkee",
    enrollmentNumber: "22115158",
    mobileNumber: "9860974349",
    gender: "female",
    avatar: "/avatars/female1.png"
  },
  {
    name: "Shreyash Sinha",
    role: "B.Tech Student at IIT Roorkee",
    enrollmentNumber: "21111037",
    mobileNumber: "9711019497",
    gender: "male",
    avatar: "/avatars/male3.png"
  }
];

export default function About() {
  const theme = useTheme();

  return (
    <Box sx={{ 
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #0F0F0F 0%, #121212 100%)',
      color: 'white',
      pt: 4,
      pb: 8
    }}>
      {/* Logo Section */}
      <Container maxWidth="lg" sx={{ mb: 6 }}>
        <Logo />
      </Container>

      <Container maxWidth="lg">
        {/* Header Section */}
        <Box sx={{ textAlign: 'center', mb: 8 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 3 }}>
            <Timeline sx={{ color: 'primary.main', mr: 1, fontSize: '2.5rem' }} />
            <Typography 
              variant="h3" 
              component="h1"
              sx={{ 
                fontWeight: 700,
                background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.secondary.light} 100%)`,
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              About IntelliWealth
            </Typography>
          </Box>
          <Typography variant="h6" color="text.secondary" sx={{ maxWidth: 800, mx: 'auto', mb: 4 }}>
            IntelliWealth is a cutting-edge AI-powered wealth management platform developed by a team of talented students from IIT Roorkee.
          </Typography>
        </Box>

        {/* Team Section */}
        <Box sx={{ mb: 8 }}>
          <Typography 
            variant="h4" 
            sx={{ 
              mb: 4, 
              textAlign: 'center',
              fontWeight: 700,
              background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.secondary.light} 100%)`,
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            Meet Our Team
          </Typography>

          {/* Group 22 Heading */}
          <Typography 
            variant="h2" 
            sx={{ 
              mb: 6, 
              textAlign: 'center',
              fontWeight: 900,
              fontSize: '3.5rem',
              textTransform: 'uppercase',
              letterSpacing: '4px',
              background: `linear-gradient(135deg, #FFD700 0%, #FFA500 100%)`,
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              textShadow: '2px 2px 4px rgba(0,0,0,0.3)',
              position: 'relative',
              '&::after': {
                content: '""',
                position: 'absolute',
                bottom: '-10px',
                left: '50%',
                transform: 'translateX(-50%)',
                width: '100px',
                height: '4px',
                background: `linear-gradient(90deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                borderRadius: '2px'
              }
            }}
          >
            Group 22
          </Typography>

          <Grid container spacing={4} justifyContent="center" sx={{ px: 2 }}>
            {teamMembers.map((member, index) => (
              <Grid item xs={12} sm={6} lg={4} key={index} sx={{ display: 'flex' }}>
                <Paper
                  elevation={0}
                  sx={{
                    p: 4,
                    width: '100%',
                    height: '100%',
                    background: 'rgba(30, 30, 30, 0.8)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '16px',
                    transition: 'all 0.3s ease',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    '&:hover': {
                      transform: 'translateY(-5px)',
                      borderColor: 'rgba(62, 237, 191, 0.3)',
                      boxShadow: '0 10px 20px rgba(0,0,0,0.2)',
                      background: 'rgba(35, 35, 35, 0.9)'
                    }
                  }}
                >
                  <Avatar
                    src={member.avatar}
                    sx={{
                      width: 100,
                      height: 100,
                      bgcolor: '#3EEFBF',
                      border: '3px solid #3EEFBF',
                      boxShadow: '0 4px 8px rgba(0,0,0,0.2)',
                      mb: 3,
                      '&:hover': {
                        transform: 'scale(1.05)',
                        transition: 'transform 0.3s ease'
                      }
                    }}
                  >
                    {member.name.charAt(0)}
                  </Avatar>
                  <Box sx={{ 
                    textAlign: 'center',
                    width: '100%'
                  }}>
                    <Typography variant="h6" sx={{ 
                      fontWeight: 600,
                      mb: 1,
                      color: '#FFFFFF',
                      fontSize: '1.2rem'
                    }}>
                      {member.name}
                    </Typography>
                    <Typography 
                      variant="body2" 
                      sx={{ 
                        color: 'rgba(255,255,255,0.7)',
                        mb: 2,
                        fontSize: '0.9rem'
                      }}
                    >
                      {member.role}
                    </Typography>
                    <Divider sx={{ 
                      my: 2, 
                      borderColor: 'rgba(255,255,255,0.1)',
                      width: '80%',
                      mx: 'auto'
                    }} />
                    <Box sx={{
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 1,
                      alignItems: 'center'
                    }}>
                      <Typography 
                        variant="body2" 
                        sx={{ 
                          color: '#3EEFBF',
                          fontSize: '0.9rem'
                        }}
                      >
                        Enrollment: {member.enrollmentNumber}
                      </Typography>
                      <Typography 
                        variant="body2" 
                        sx={{ 
                          color: '#3EEFBF',
                          fontSize: '0.9rem'
                        }}
                      >
                        Mobile: {member.mobileNumber}
                      </Typography>
                    </Box>
                  </Box>
                </Paper>
              </Grid>
            ))}
          </Grid>
        </Box>

        {/* About Website Section */}
        <Box sx={{ mb: 8 }}>
          <Typography 
            variant="h4" 
            sx={{ 
              mb: 4, 
              textAlign: 'center',
              fontWeight: 700,
              background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.secondary.light} 100%)`,
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            About IntelliWealth Platform
          </Typography>
          <Paper
            elevation={0}
            sx={{
              p: 4,
              background: 'rgba(30, 30, 30, 0.6)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '16px'
            }}
          >
            <Typography variant="body1" paragraph color="text.secondary">
              IntelliWealth is a revolutionary wealth management platform that combines cutting-edge artificial intelligence with intuitive design to democratize sophisticated investment strategies. Our platform provides institutional-grade tools and insights previously only available to hedge funds and professional wealth managers.
            </Typography>
            <Typography variant="body1" paragraph color="text.secondary">
              Developed as part of our academic project at IIT Roorkee, IntelliWealth represents our vision for the future of personal finance - where advanced technology meets user-friendly design to help individuals make better investment decisions.
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Our platform features AI-powered portfolio management, real-time market insights, smart asset allocation, and comprehensive risk assessment tools, all designed to help users optimize their investments and achieve their financial goals.
            </Typography>
          </Paper>
        </Box>
      </Container>
    </Box>
  );
}
