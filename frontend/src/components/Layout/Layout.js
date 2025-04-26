'use client'
import React, { useContext, useState, useEffect } from 'react';
import { Outlet, useLocation, useNavigate } from 'react-router-dom';
import { 
  AppBar, 
  Avatar, 
  Box, 
  CssBaseline, 
  Divider, 
  Drawer, 
  IconButton, 
  List, 
  ListItem, 
  ListItemButton, 
  ListItemIcon, 
  ListItemText, 
  Menu, 
  MenuItem, 
  Toolbar, 
  Tooltip, 
  Typography,
  styled,
  Badge,
  Popover,
  ListItemAvatar,
  Button,
  CircularProgress
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  Assessment as AssessmentIcon,
  PieChart as PieChartIcon,
  TrendingUp as TrendingUpIcon,
  Timeline as TimelineIcon,
  AccountBalance as AccountBalanceIcon,
  NotificationsNone as NotificationsIcon,
  Person as PersonIcon,
  Settings as SettingsIcon,
  ExitToApp as LogoutIcon,
  Close as CloseIcon
} from '@mui/icons-material';
import { UserContext } from '../../context/UserContext';
import { getFinancialNews } from '../../services/newsService';

const drawerWidth = 240;

const StyledListItemButton = styled(ListItemButton)(({ theme }) => ({
  borderRadius: 12,
  margin: '4px 8px',
  '&.Mui-selected': {
    backgroundColor: `${theme.palette.primary.main}20`,
    color: theme.palette.primary.light,
    '& .MuiListItemIcon-root': {
      color: theme.palette.primary.light,
    },
  },
  '&:hover': {
    backgroundColor: `${theme.palette.primary.main}10`,
  },
}));

function Layout() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [anchorEl, setAnchorEl] = useState(null);
  const [notifications, setNotifications] = useState([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [notificationAnchor, setNotificationAnchor] = useState(null);
  const [loadingNotifications, setLoadingNotifications] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const { user, updateUser, loading } = useContext(UserContext);

  useEffect(() => {
    if (!loading && !user) {
      navigate('/login');
    }
  }, [loading, user, navigate]);

  useEffect(() => {
    setMobileOpen(false);
  }, [location.pathname]);

  useEffect(() => {
    const fetchNotifications = async () => {
      if (!user) return;
      
      setLoadingNotifications(true);
      try {
        const news = await getFinancialNews();
        const formattedNotifications = news.map((item, index) => ({
          id: `news-${index}`,
          title: item.title || 'Financial News Update',
          summary: item.description ? `${item.description.substring(0, 100)}...` : 'Click to read more...',
          date: item.publishedAt ? new Date(item.publishedAt).toLocaleDateString() : 'Today',
          read: false,
          isNews: true,
          url: item.url || '#'
        }));
        setNotifications(formattedNotifications);
        setUnreadCount(formattedNotifications.filter(n => !n.read).length);
      } catch (error) {
        console.error('Error fetching notifications:', error);
        // Fallback to empty array if API fails
        setNotifications([]);
      } finally {
        setLoadingNotifications(false);
      }
    };
    
    fetchNotifications();
    
    // Refresh notifications every 5 minutes
    const interval = setInterval(fetchNotifications, 300000);
    return () => clearInterval(interval);
  }, [user]);

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handleProfileMenuOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleNavigation = (path) => {
    navigate(path);
  };

  const handleLogout = () => {
    localStorage.removeItem('user');
    navigate('/login');
  };

  const handleNotificationClick = (event) => {
    setNotificationAnchor(event.currentTarget);
  };

  const handleNotificationClose = () => {
    setNotificationAnchor(null);
  };

  const markAsRead = (id) => {
    setNotifications(prev => 
      prev.map(item => 
        item.id === id ? { ...item, read: true } : item
      )
    );
    setUnreadCount(prev => prev - 1);
  };

  const markAllAsRead = () => {
    setNotifications(prev => 
      prev.map(item => ({
        ...item,
        read: true
      }))
    );
    setUnreadCount(0);
  };

  const handleNotificationItemClick = (item) => {
    markAsRead(item.id);
    if (item.url && item.url !== '#') {
      window.open(item.url, '_blank');
    }
    handleNotificationClose();
  };

  const menuItems = [
    { text: 'Dashboard', icon: <DashboardIcon />, path: '/dashboard' },
    { text: 'Sentiment Analysis', icon: <AssessmentIcon />, path: '/dashboard/Sentiment' },
    // { text: 'Portfolio', icon: <AccountBalanceIcon />, path: '/dashboard/portfolio' },
    // { text: 'Asset Allocation', icon: <PieChartIcon />, path: '/dashboard/asset-allocation' },
    { text: 'Market Insights', icon: <TrendingUpIcon />, path: '/dashboard/market-insights' },
    { text: 'Portfolio Optimizer', icon: <TrendingUpIcon />, path: '/dashboard/portfolio-optimizer' },
    { text: 'Financial Report', icon: <TrendingUpIcon />, path: '/dashboard/financial-report' },
    { text: 'Stock News', icon: <TrendingUpIcon />, path: '/dashboard/stock-news' },
  ];

  const drawer = (
    <div>
      <Toolbar sx={{ justifyContent: 'center', py: 2 }}>
        <Box 
          sx={{ 
            display: 'flex', 
            alignItems: 'center',
            cursor: 'pointer',
            '&:hover': {
              opacity: 0.8
            }
          }}
          onClick={() => navigate('/dashboard')}
        >
          <TimelineIcon 
            sx={{ 
              color: 'primary.main', 
              fontSize: '2rem',
              mr: 1 
            }} 
          />
          <Typography
            variant="h5"
            component="div"
            sx={{
              fontWeight: 700,
              background: 'linear-gradient(135deg, #3EEDBF 0%, #FF80AB 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent'
            }}
          >
            IntelliWealth
          </Typography>
        </Box>
      </Toolbar>
      <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.12)' }} />
      <List>
        {menuItems.map((item) => (
          <ListItem key={item.text} disablePadding>
            <StyledListItemButton
              onClick={() => handleNavigation(item.path)}
              selected={location.pathname === item.path}
            >
              <ListItemIcon sx={{ color: 'inherit' }}>
                {item.icon}
              </ListItemIcon>
              <ListItemText primary={item.text} />
            </StyledListItemButton>
          </ListItem>
        ))}
      </List>
    </div>
  );

  const isMenuOpen = Boolean(anchorEl);
  const isNotificationOpen = Boolean(notificationAnchor);

  // If we're loading the user or user is null, show a simplified layout
  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <CircularProgress />
      </Box>
    );
  }
  
  // If user is still null after loading, show a message instead of redirecting in a hook
  if (!user) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <Typography variant="body1">Session expired. Redirecting to login...</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ display: 'flex' }}>
      <CssBaseline />
      <AppBar
        position="fixed"
        sx={{
          width: '100%',
          backgroundColor: 'background.paper',
          color: 'text.primary',
          boxShadow: 'none',
          borderBottom: '1px solid rgba(255, 255, 255, 0.12)'
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2 }}
          >
            <MenuIcon />
          </IconButton>

          <Box 
            sx={{ 
              display: 'flex', 
              alignItems: 'center',
              mr: 2,
              cursor: 'pointer',
              '&:hover': {
                opacity: 0.8
              }
            }}
            onClick={() => navigate('/dashboard')}
          >
            <TimelineIcon 
              sx={{ 
                color: 'primary.main', 
                fontSize: '2rem',
                mr: 1 
              }} 
            />
            <Typography
              variant="h5"
              component="div"
              sx={{
                fontWeight: 700,
                background: 'linear-gradient(135deg, #3EEDBF 0%, #FF80AB 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent'
              }}
            >
              IntelliWealth
            </Typography>
          </Box>

          <Box sx={{ flexGrow: 1 }} />
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Tooltip title="Notifications">
              <IconButton
                size="large"
                color="inherit"
                onClick={handleNotificationClick}
              >
                <Badge badgeContent={unreadCount} color="error">
                  <NotificationsIcon />
                </Badge>
              </IconButton>
            </Tooltip>
            
            <Popover
              open={isNotificationOpen}
              anchorEl={notificationAnchor}
              onClose={handleNotificationClose}
              anchorOrigin={{
                vertical: 'bottom',
                horizontal: 'right',
              }}
              transformOrigin={{
                vertical: 'top',
                horizontal: 'right',
              }}
              PaperProps={{
                sx: {
                  width: 350,
                  maxHeight: 500,
                  p: 1
                }
              }}
            >
              <Box sx={{ 
                display: 'flex', 
                justifyContent: 'space-between', 
                alignItems: 'center',
                p: 1
              }}>
                <Typography variant="h6" fontWeight="bold">
                  Notifications
                </Typography>
                <Box>
                  <Button 
                    size="small" 
                    onClick={markAllAsRead}
                    disabled={unreadCount === 0}
                  >
                    Mark all as read
                  </Button>
                  <IconButton size="small" onClick={handleNotificationClose}>
                    <CloseIcon fontSize="small" />
                  </IconButton>
                </Box>
              </Box>
              
              <Divider />
              
              <List sx={{ overflow: 'auto' }}>
                {loadingNotifications ? (
                  <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
                    <CircularProgress size={24} />
                  </Box>
                ) : notifications.length > 0 ? (
                  notifications.map((item) => (
                    <ListItem 
                      key={item.id} 
                      sx={{ 
                        backgroundColor: item.read ? 'inherit' : 'action.hover',
                        borderRadius: 1,
                        mb: 0.5
                      }}
                    >
                      <ListItemButton
                        onClick={() => handleNotificationItemClick(item)}
                      >
                        <ListItemAvatar>
                          <Avatar sx={{ bgcolor: item.isNews ? 'secondary.main' : 'primary.main' }}>
                            {item.isNews ? <TrendingUpIcon fontSize="small" /> : <NotificationsIcon fontSize="small" />}
                          </Avatar>
                        </ListItemAvatar>
                        <ListItemText
                          primary={
                            <Typography fontWeight={item.read ? 'normal' : 'bold'}>
                              {item.title}
                            </Typography>
                          }
                          secondary={
                            <>
                              <Typography
                                component="span"
                                variant="body2"
                                color="text.primary"
                                display="block"
                              >
                                {item.summary}
                              </Typography>
                              <Typography variant="caption">
                                {item.date}
                              </Typography>
                            </>
                          }
                        />
                      </ListItemButton>
                    </ListItem>
                  ))
                ) : (
                  <ListItem>
                    <ListItemText 
                      primary="No new notifications" 
                      sx={{ textAlign: 'center', py: 2 }} 
                    />
                  </ListItem>
                )}
              </List>
              
              <Divider />
              
              <Box sx={{ p: 1, textAlign: 'center' }}>
                <Button 
                  size="small" 
                  onClick={() => navigate('/dashboard/notifications')}
                >
                  View all notifications
                </Button>
              </Box>
            </Popover>

            <Tooltip title="Account">
              <IconButton
                size="large"
                edge="end"
                aria-label="account of current user"
                aria-haspopup="true"
                onClick={handleProfileMenuOpen}
                color="inherit"
              >
                {loading ? (
                  <CircularProgress size={24} />
                ) : user && user.profile && user.profile.avatar ? (
                  <Avatar 
                    src={user.profile.avatar} 
                    sx={{ width: 32, height: 32 }}
                    alt={user.profile?.name || 'User'}
                  />
                ) : (
                  <Avatar sx={{ width: 32, height: 32, bgcolor: 'primary.main' }}>
                    {user && user.profile && user.profile.name ? user.profile.name.split(' ').map(n => n[0]).join('') : 'U'}
                  </Avatar>
                )}
              </IconButton>
            </Tooltip>
          </Box>
        </Toolbar>
      </AppBar>

      <Menu
        anchorEl={anchorEl}
        open={isMenuOpen}
        onClose={handleMenuClose}
        MenuListProps={{ sx: { py: 0 } }}
      >
        <MenuItem onClick={() => navigate('/dashboard/profile')}>
          <ListItemIcon><PersonIcon fontSize="small" /></ListItemIcon>
          Profile
        </MenuItem>
        <MenuItem onClick={() => navigate('/dashboard/settings')}>
          <ListItemIcon><SettingsIcon fontSize="small" /></ListItemIcon>
          Settings
        </MenuItem>
        <Divider />
        <MenuItem onClick={handleLogout}>
          <ListItemIcon><LogoutIcon fontSize="small" /></ListItemIcon>
          Logout
        </MenuItem>
      </Menu>

      <Drawer
        variant="temporary"
        open={mobileOpen}
        onClose={handleDrawerToggle}
        ModalProps={{ keepMounted: true }}
        sx={{
          '& .MuiDrawer-paper': {
            width: drawerWidth,
            backgroundColor: 'background.paper',
          },
        }}
      >
        {drawer}
      </Drawer>

      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          width: '100%',
          minHeight: '100vh',
          backgroundColor: 'background.default'
        }}
      >
        <Toolbar />
        <Outlet />
      </Box>
    </Box>
  );
}

export default Layout;