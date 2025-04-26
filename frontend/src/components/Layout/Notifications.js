import React, { useState } from 'react';
import {
  Badge,
  IconButton,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Typography,
  Box
} from '@mui/material';
import { Notifications as NotificationsIcon, Close as CloseIcon } from '@mui/icons-material';
import { getFinancialNews } from '../../services/newsService';

const Notifications = () => {
  const [anchorEl, setAnchorEl] = useState(null);
  const [notifications, setNotifications] = useState([]);

  const handleMenuOpen = async (event) => {
    setAnchorEl(event.currentTarget);
    try {
      const news = await getFinancialNews();
      setNotifications(news);
    } catch (error) {
      console.error('Error loading notifications:', error);
    }
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  return (
    <>
      <IconButton
        size="large"
        color="inherit"
        onClick={handleMenuOpen}
      >
        <Badge badgeContent={notifications.length} color="error">
          <NotificationsIcon />
        </Badge>
      </IconButton>

      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
        PaperProps={{
          sx: {
            width: 350,
            maxHeight: 400,
          },
        }}
      >
        <MenuItem disabled>
          <Typography variant="subtitle1" fontWeight="bold">
            Notifications
          </Typography>
          <Box sx={{ flexGrow: 1 }} />
          <IconButton size="small" onClick={handleMenuClose}>
            <CloseIcon fontSize="small" />
          </IconButton>
        </MenuItem>
        <Divider />

        {notifications.length > 0 ? (
          notifications.map((notification, index) => (
            <React.Fragment key={index}>
              <MenuItem onClick={handleMenuClose}>
                <ListItemText
                  primary={notification.title}
                  secondary={notification.description}
                  primaryTypographyProps={{ noWrap: true }}
                  secondaryTypographyProps={{ noWrap: true }}
                />
              </MenuItem>
              {index < notifications.length - 1 && <Divider />}
            </React.Fragment>
          ))
        ) : (
          <MenuItem disabled>
            <ListItemText primary="No new notifications" />
          </MenuItem>
        )}
      </Menu>
    </>
  );
};

export default Notifications;