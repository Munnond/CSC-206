// SettingsPage.js
import React from 'react';
import { 
  Box, 
  Typography, 
  Card, 
  CardContent, 
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Switch,
  Button,
  Select,
  MenuItem,
  FormControl,
  Alert,
  Snackbar
} from '@mui/material';
import { 
  Notifications as NotificationsIcon,
  Security as SecurityIcon,
  Language as LanguageIcon,
  CreditCard as CreditCardIcon
} from '@mui/icons-material';

export default function SettingsPage() {
  const [settings, setSettings] = React.useState({
    notifications: true,
    language: 'en',
    currency: 'USD',
    emailFrequency: 'weekly',
    securityAlerts: true
  });

  const [openSnackbar, setOpenSnackbar] = React.useState(false);

  const handleSettingChange = (setting) => (event) => {
    setSettings({ ...settings, [setting]: event.target.checked });
  };

  const handleSelectChange = (setting) => (event) => {
    setSettings({ ...settings, [setting]: event.target.value });
  };

  const handleSave = () => {
    // API call to save settings would go here
    console.log('Settings saved:', settings);
    setOpenSnackbar(true);
  };

  const handleCloseSnackbar = () => {
    setOpenSnackbar(false);
  };

  return (
    <Box sx={{ maxWidth: 800, mx: 'auto', my: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom sx={{ mb: 4 }}>
        Account Settings
      </Typography>

      <Snackbar
        open={openSnackbar}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
      >
        <Alert onClose={handleCloseSnackbar} severity="success">
          Settings saved successfully!
        </Alert>
      </Snackbar>

      {/* Application Preferences */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
            Preferences
          </Typography>
          
          <List>
            <ListItem>
              <ListItemIcon>
                <LanguageIcon />
              </ListItemIcon>
              <ListItemText primary="Language" secondary="Select your preferred language" />
              <FormControl sx={{ minWidth: 120 }}>
                <Select
                  value={settings.language}
                  onChange={handleSelectChange('language')}
                >
                  <MenuItem value="en">English</MenuItem>
                </Select>
              </FormControl>
            </ListItem>
            
            <Divider variant="inset" component="li" />
            
            <ListItem>
              <ListItemIcon>
                <CreditCardIcon />
              </ListItemIcon>
              <ListItemText primary="Currency" secondary="Select your preferred currency" />
              <FormControl sx={{ minWidth: 120 }}>
                <Select
                  value={settings.currency}
                  onChange={handleSelectChange('currency')}
                >
                  <MenuItem value="USD">USD ($)</MenuItem>
                  <MenuItem value="EUR">EUR (€)</MenuItem>
                  <MenuItem value="INR">INR (₹)</MenuItem>
                  <MenuItem value="JPY">JPY (¥)</MenuItem>
                </Select>
              </FormControl>
            </ListItem>
          </List>
        </CardContent>
      </Card>

      {/* Notification Settings */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
            <NotificationsIcon sx={{ mr: 1 }} /> Notifications
          </Typography>
          
          <List>
            <ListItem>
              <ListItemIcon>
                <NotificationsIcon />
              </ListItemIcon>
              <ListItemText 
                primary="Email Notifications" 
                secondary="Receive important account notifications" 
              />
              <Switch
                checked={settings.notifications}
                onChange={handleSettingChange('notifications')}
              />
            </ListItem>
            
            <Divider variant="inset" component="li" />
            
            <ListItem>
              <ListItemIcon>
                <SecurityIcon />
              </ListItemIcon>
              <ListItemText 
                primary="Security Alerts" 
                secondary="Get immediate alerts about security events" 
              />
              <Switch
                checked={settings.securityAlerts}
                onChange={handleSettingChange('securityAlerts')}
              />
            </ListItem>
          </List>
        </CardContent>
      </Card>

      {/* Security Settings */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
            <SecurityIcon sx={{ mr: 1 }} /> Security
          </Typography>
          
          <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
            <Button variant="outlined" color="error">
              Change Password
            </Button>
            <Button variant="outlined">
              Manage Active Sessions
            </Button>
          </Box>
        </CardContent>
      </Card>

      {/* Data & Privacy */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Data & Privacy
          </Typography>
          <Box sx={{ display: 'flex', gap: 2, mt: 2 }}>
            <Button variant="outlined">
              Download Your Data
            </Button>
            <Button variant="outlined" color="error">
              Delete Account
            </Button>
          </Box>
        </CardContent>
      </Card>

      <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
        <Button 
          variant="contained" 
          color="primary" 
          size="large"
          onClick={handleSave}
          sx={{ minWidth: 200 }}
        >
          Save Settings
        </Button>
      </Box>
    </Box>
  );
}