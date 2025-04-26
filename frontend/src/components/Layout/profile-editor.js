'use client'

import { useState, useContext } from 'react';
import { 
  Box, 
  Typography, 
  Card, 
  CardContent, 
  Avatar,
  Divider,
  Button,
  TextField,
  Grid,
  InputAdornment,
  IconButton,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Alert,
  Snackbar
} from '@mui/material';
import { 
  Person as PersonIcon, 
  Edit as EditIcon,
  Email as EmailIcon,
  Phone as PhoneIcon,
  LocationOn as LocationIcon,
  CalendarToday as CalendarIcon,
  Visibility,
  VisibilityOff,
  Save as SaveIcon,
  Cancel as CancelIcon,
  Add as AddIcon
} from '@mui/icons-material';
import { UserContext } from '../../context/UserContext';
import Chip from '@mui/material/Chip';
import Stack from '@mui/material/Stack';

const countries = [
  { code: 'US', name: 'United States' },
  { code: 'IN', name: 'India' },
  { code: 'GB', name: 'United Kingdom' },
  { code: 'CA', name: 'Canada' },
  { code: 'AU', name: 'Australia' },
  { code: 'DE', name: 'Germany' },
  { code: 'FR', name: 'France' },
  { code: 'JP', name: 'Japan' },
  { code: 'BR', name: 'Brazil' },
  { code: 'CN', name: 'China' }
];

export default function ProfileEditor({ onClose }) {
  const { user, updateUser } = useContext(UserContext);
  const [showPassword, setShowPassword] = useState(false);
  const [showCurrentPassword, setShowCurrentPassword] = useState(false);
  const [newInterest, setNewInterest] = useState('');
  const [openSnackbar, setOpenSnackbar] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  
  const [profile, setProfile] = useState({
    firstName: user?.firstName || '',
    lastName: user?.lastName || '',
    email: user?.email || '',
    phone: user?.phone || '',
    birthDate: user?.birthDate || '',
    bio: user?.bio || '',
    address: user?.address || {
      street: '',
      city: '',
      country: '',
      zipCode: ''
    },
    notifications: user?.notifications || true,
    twoFactorAuth: user?.twoFactorAuth || false,
    interests: user?.interests || [],
    currentPassword: '',
    newPassword: '',
    confirmPassword: ''
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    
    if (name.startsWith('address.')) {
      const addressField = name.split('.')[1];
      setProfile(prev => ({
        ...prev,
        address: {
          ...prev.address,
          [addressField]: value
        }
      }));
    } else {
      setProfile(prev => ({ ...prev, [name]: value }));
    }
  };

  const handleSwitchChange = (e) => {
    const { name, checked } = e.target;
    setProfile(prev => ({ ...prev, [name]: checked }));
  };

  const handleAddInterest = () => {
    if (newInterest && !profile.interests.includes(newInterest)) {
      setProfile(prev => ({
        ...prev,
        interests: [...prev.interests, newInterest]
      }));
      setNewInterest('');
    }
  };

  const handleRemoveInterest = (interest) => {
    setProfile(prev => ({
      ...prev,
      interests: prev.interests.filter(i => i !== interest)
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    // Prepare updated user data
    const updatedUser = {
      name: `${profile.firstName} ${profile.lastName}`,
      firstName: profile.firstName,
      lastName: profile.lastName,
      email: profile.email,
      phone: profile.phone,
      birthDate: profile.birthDate,
      bio: profile.bio,
      address: profile.address,
      notifications: profile.notifications,
      twoFactorAuth: profile.twoFactorAuth,
      interests: profile.interests
    };

    // Update user context
    updateUser(updatedUser);
    
    // Show success message
    setSnackbarMessage('Profile updated successfully');
    setOpenSnackbar(true);
    
    // Close the editor after 1 second
    setTimeout(() => {
      onClose();
    }, 1000);
  };

  return (
    <Box sx={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(0, 0, 0, 0.7)',
      zIndex: 1000,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      p: 2,
    }} onClick={onClose}>
      <Card sx={{ 
        maxWidth: 1200, 
        width: '100%', 
        maxHeight: '90vh',
        overflowY: 'auto',
      }} onClick={(e) => e.stopPropagation()}>
        <CardContent>
          <Snackbar
            open={openSnackbar}
            autoHideDuration={6000}
            onClose={() => setOpenSnackbar(false)}
          >
            <Alert severity="success">
              {snackbarMessage}
            </Alert>
          </Snackbar>

          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
            <Typography variant="h4" fontWeight="bold">
              Edit Profile
            </Typography>
            <Box>
              <Button
                variant="outlined"
                startIcon={<CancelIcon />}
                onClick={onClose}
                sx={{ mr: 2 }}
              >
                Cancel
              </Button>
              <Button
                variant="contained"
                startIcon={<SaveIcon />}
                onClick={handleSubmit}
              >
                Save Changes
              </Button>
            </Box>
          </Box>

          <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 4 }}>
            <Box sx={{ width: { xs: '100%', md: '30%' } }}>
              <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', mb: 4 }}>
                <Avatar sx={{ 
                  width: 150, 
                  height: 150, 
                  fontSize: 60, 
                  mb: 3,
                  bgcolor: 'primary.main'
                }}>
                  {profile.firstName?.[0]}{profile.lastName?.[0]}
                </Avatar>
                
                <TextField
                  fullWidth
                  label="First Name"
                  name="firstName"
                  value={profile.firstName}
                  onChange={handleChange}
                  sx={{ mb: 2 }}
                />
                <TextField
                  fullWidth
                  label="Last Name"
                  name="lastName"
                  value={profile.lastName}
                  onChange={handleChange}
                  sx={{ mb: 2 }}
                />
              </Box>

              <Divider sx={{ my: 3 }} />

              <Typography variant="h6" gutterBottom fontWeight="bold">
                Account Security
              </Typography>
              
              <TextField
                fullWidth
                label="Email"
                name="email"
                value={profile.email}
                onChange={handleChange}
                InputProps={{
                  startAdornment: <EmailIcon />,
                }}
                sx={{ mb: 2 }}
              />
              
              <TextField
                fullWidth
                label="Current Password"
                type={showCurrentPassword ? 'text' : 'password'}
                name="currentPassword"
                value={profile.currentPassword}
                onChange={handleChange}
                InputProps={{
                  startAdornment: <PersonIcon />,
                  endAdornment: (
                    <IconButton onClick={() => setShowCurrentPassword(!showCurrentPassword)}>
                      {showCurrentPassword ? <VisibilityOff /> : <Visibility />}
                    </IconButton>
                  ),
                }}
                sx={{ mb: 2 }}
              />
              
              <TextField
                fullWidth
                label="New Password"
                type={showPassword ? 'text' : 'password'}
                name="newPassword"
                value={profile.newPassword}
                onChange={handleChange}
                InputProps={{
                  endAdornment: (
                    <IconButton onClick={() => setShowPassword(!showPassword)}>
                      {showPassword ? <VisibilityOff /> : <Visibility />}
                    </IconButton>
                  ),
                }}
                sx={{ mb: 2 }}
              />
              
              <FormControlLabel
                control={
                  <Switch
                    checked={profile.twoFactorAuth}
                    onChange={handleSwitchChange}
                    name="twoFactorAuth"
                  />
                }
                label="Two-Factor Authentication"
              />
            </Box>

            <Box sx={{ width: { xs: '100%', md: '70%' } }}>
              <Typography variant="h6" gutterBottom fontWeight="bold">
                Contact Information
              </Typography>
              <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Phone Number"
                    name="phone"
                    value={profile.phone}
                    onChange={handleChange}
                    InputProps={{
                      startAdornment: <PhoneIcon />,
                    }}
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Birth Date"
                    type="date"
                    name="birthDate"
                    value={profile.birthDate}
                    onChange={handleChange}
                    InputLabelProps={{ shrink: true }}
                    InputProps={{
                      startAdornment: <CalendarIcon />,
                    }}
                  />
                </Grid>
              </Grid>

              <Typography variant="h6" gutterBottom fontWeight="bold" mt={4}>
                Address Information
              </Typography>
              <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Street Address"
                    name="address.street"
                    value={profile.address.street}
                    onChange={handleChange}
                    InputProps={{
                      startAdornment: <LocationIcon />,
                    }}
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="City"
                    name="address.city"
                    value={profile.address.city}
                    onChange={handleChange}
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth>
                    <InputLabel>Country</InputLabel>
                    <Select
                      label="Country"
                      name="address.country"
                      value={profile.address.country}
                      onChange={handleChange}
                    >
                      {countries.map((country) => (
                        <MenuItem key={country.code} value={country.name}>
                          {country.name}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="ZIP/Postal Code"
                    name="address.zipCode"
                    value={profile.address.zipCode}
                    onChange={handleChange}
                  />
                </Grid>
              </Grid>

              <Typography variant="h6" gutterBottom fontWeight="bold" mt={4}>
                Additional Information
              </Typography>
              
              <TextField
                fullWidth
                label="Bio"
                name="bio"
                value={profile.bio}
                onChange={handleChange}
                multiline
                rows={4}
                sx={{ mb: 2 }}
              />
              
              <FormControlLabel
                control={
                  <Switch
                    checked={profile.notifications}
                    onChange={handleSwitchChange}
                    name="notifications"
                  />
                }
                label="Email Notifications"
                sx={{ mb: 2 }}
              />

              <Typography variant="h6" gutterBottom fontWeight="bold" mt={4}>
                Investment Interests
              </Typography>
              <Box sx={{ mb: 2 }}>
                <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap sx={{ mb: 2 }}>
                  {profile.interests.map((interest) => (
                    <Chip
                      key={interest}
                      label={interest}
                      onDelete={() => handleRemoveInterest(interest)}
                      sx={{ m: 0.5 }}
                    />
                  ))}
                </Stack>
                <Grid container spacing={1}>
                  <Grid item xs>
                    <TextField
                      fullWidth
                      label="Add Interest"
                      value={newInterest}
                      onChange={(e) => setNewInterest(e.target.value)}
                    />
                  </Grid>
                  <Grid item>
                    <Button
                      variant="contained"
                      startIcon={<AddIcon />}
                      onClick={handleAddInterest}
                      disabled={!newInterest}
                      sx={{ height: '100%' }}
                    >
                      Add
                    </Button>
                  </Grid>
                </Grid>
              </Box>
            </Box>
          </Box>
        </CardContent>
      </Card>
    </Box>
  )
}