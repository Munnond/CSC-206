// ProfilePage.js
'use client'

import React, { useContext, useEffect, useState } from 'react';
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
  Snackbar,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText
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
  Check as CheckIcon
} from '@mui/icons-material';
import { UserContext } from '../context/UserContext';
import { useFormik } from 'formik';
import * as Yup from 'yup';

// Avatar options
const avatarOptions = [
  { id: 'male1', label: 'Male 1', path: '/avatars/male1.png' },
  { id: 'male2', label: 'Male 2', path: '/avatars/male2.png' },
  { id: 'male3', label: 'Male 3', path: '/avatars/male3.png' },
  { id: 'female1', label: 'Female 1', path: '/avatars/female1.png' },
  { id: 'female2', label: 'Female 2', path: '/avatars/female2.png' },
  { id: 'female3', label: 'Female 3', path: '/avatars/female3.png' },
];

// Temporary countries data
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

// Validation schema
const profileSchema = Yup.object().shape({
  firstName: Yup.string().required('First name is required'),
  lastName: Yup.string().required('Last name is required'),
  email: Yup.string().email('Invalid email').required('Email is required'),
  phone: Yup.string().matches(/^[0-9]{10,15}$/, 'Phone number is not valid'),
  birthDate: Yup.date().max(new Date(), 'Birth date cannot be in the future'),
  address: Yup.object().shape({
    street: Yup.string(),
    city: Yup.string(),
    country: Yup.string(),
    zipCode: Yup.string()
  }),
  notifications: Yup.boolean(),
  twoFactorAuth: Yup.boolean()
});

export default function ProfilePage() {
  const { user, updateUser } = useContext(UserContext);
  const [editMode, setEditMode] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [openSnackbar, setOpenSnackbar] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const [snackbarSeverity, setSnackbarSeverity] = useState('success');
  const [avatarDialogOpen, setAvatarDialogOpen] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const formik = useFormik({
    initialValues: {
      firstName: user?.firstName || '',
      lastName: user?.lastName || '',
      email: user?.email || '',
      phone: user?.phone || '',
      birthDate: user?.birthDate || '',
      bio: user?.bio || '',
      avatar: user?.avatar || '',
      address: user?.address || {
        street: '',
        city: '',
        country: '',
        zipCode: ''
      },
      notifications: user?.notifications || true,
      twoFactorAuth: user?.twoFactorAuth || false
    },
    validationSchema: profileSchema,
    onSubmit: async (values) => {
      setIsSubmitting(true);
      
      try {
        // Prepare data for API
        const profileUpdate = {
          user: {
            email: values.email
          },
          profile: {
            name: `${values.firstName} ${values.lastName}`,
            phone: values.phone,
            birth_date: values.birthDate,
            avatar: typeof values.avatar === 'number' ? values.avatar : 
                  (values.avatar && !isNaN(parseInt(values.avatar)) ? parseInt(values.avatar) : 0)
            // Add any other profile fields that need to be updated
          }
        };
        
        console.log("Sending profile update:", profileUpdate);
        const result = await updateUser(profileUpdate);
        
        if (result.success) {
          showSnackbar('Profile updated successfully', 'success');
          setEditMode(false);
        } else {
          // Extract error message properly
          let errorMessage = 'Failed to update profile';
          if (result.error) {
            if (typeof result.error === 'string') {
              errorMessage = result.error;
            } else if (result.error.detail) {
              errorMessage = result.error.detail;
            } else if (result.error.message) {
              errorMessage = result.error.message;
            } else if (typeof result.error === 'object') {
              // Try to create a readable message from the error object
              errorMessage = Object.entries(result.error)
                .map(([key, value]) => `${key}: ${value}`)
                .join(', ');
            }
          }
          showSnackbar(errorMessage, 'error');
        }
      } catch (error) {
        console.error('Error updating profile:', error);
        let errorMessage = 'An error occurred while updating profile';
        if (error.response?.data) {
          if (typeof error.response.data === 'string') {
            errorMessage = error.response.data;
          } else if (error.response.data.detail) {
            errorMessage = error.response.data.detail;
          } else if (typeof error.response.data === 'object') {
            errorMessage = Object.entries(error.response.data)
              .map(([key, value]) => `${key}: ${value}`)
              .join(', ');
          }
        }
        showSnackbar(errorMessage, 'error');
      } finally {
        setIsSubmitting(false);
      }
    }
  });

  useEffect(() => {
    if (user) {
      // Extract first and last name from the full name
      let firstName = '', lastName = '';
      if (user.profile?.name) {
        const nameParts = user.profile.name.split(' ');
        firstName = nameParts[0] || '';
        lastName = nameParts.slice(1).join(' ') || '';
      }
      
      formik.setValues({
        firstName: firstName,
        lastName: lastName,
        email: user.user?.email || '',
        phone: user.profile?.phone || '',
        birthDate: user.profile?.birth_date || '',
        bio: user.bio || '',
        avatar: user.profile?.avatar || '',
        address: user.address || {
          street: '',
          city: '',
          country: '',
          zipCode: ''
        },
        notifications: user.notifications || true,
        twoFactorAuth: user.twoFactorAuth || false
      });
    }
  }, [user]);

  const showSnackbar = (message, severity = 'success') => {
    setSnackbarMessage(message);
    setSnackbarSeverity(severity);
    setOpenSnackbar(true);
  };

  const handleCloseSnackbar = () => {
    setOpenSnackbar(false);
  };

  const handleCancelEdit = () => {
    formik.resetForm();
    setEditMode(false);
  };

  const handleEditProfile = () => {
    setEditMode(true);
  };

  const handlePasswordVisibility = () => {
    setShowPassword(!showPassword);
  };

  const handleAvatarSelect = (avatarPath) => {
    formik.setFieldValue('avatar', avatarPath);
    setAvatarDialogOpen(false);
  };

  return (
    <Box sx={{ maxWidth: 1200, mx: 'auto', p: 3 }}>
      <Snackbar
        open={openSnackbar}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
      >
        <Alert onClose={handleCloseSnackbar} severity={snackbarSeverity} sx={{ width: '100%' }}>
          {snackbarMessage}
        </Alert>
      </Snackbar>

      <Card elevation={3} sx={{ borderRadius: 4 }}>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
            <Typography variant="h4" component="h1" fontWeight="bold">
              My Profile
            </Typography>
            {editMode ? (
              <Box>
                <Button
                  variant="outlined"
                  startIcon={<CancelIcon />}
                  onClick={handleCancelEdit}
                  sx={{ mr: 2 }}
                  disabled={isSubmitting}
                >
                  Cancel
                </Button>
                <Button
                  variant="contained"
                  startIcon={<SaveIcon />}
                  onClick={formik.handleSubmit}
                  disabled={!formik.isValid || isSubmitting}
                >
                  {isSubmitting ? 'Saving...' : 'Save Changes'}
                </Button>
              </Box>
            ) : (
              <Button
                variant="contained"
                startIcon={<EditIcon />}
                onClick={handleEditProfile}
              >
                Edit Profile
              </Button>
            )}
          </Box>
          
          <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 4 }}>
            {/* Left Column - Profile Picture and Basic Info */}
            <Box sx={{ width: { xs: '100%', md: '30%' } }}>
              <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', mb: 4 }}>
                {editMode ? (
                  <>
                    <Avatar 
                      src={formik.values.avatar || `/avatars/${user.gender || 'male'}1.png`}
                      sx={{ 
                        width: 150, 
                        height: 150, 
                        fontSize: 60, 
                        mb: 3,
                        cursor: 'pointer'
                      }}
                      onClick={() => setAvatarDialogOpen(true)}
                    />
                    <Button 
                      variant="outlined" 
                      onClick={() => setAvatarDialogOpen(true)}
                      sx={{ mb: 2 }}
                    >
                      Change Avatar
                    </Button>
                  </>
                ) : (
                  <Avatar 
                  src={user.avatar || '/avatars/male1.png'}
                    sx={{ 
                      width: 150, 
                      height: 150, 
                      fontSize: 60, 
                      mb: 3
                    }}
                  />
                )}
                
                {editMode ? (
                  <>
                    <TextField
                      fullWidth
                      label="First Name"
                      name="firstName"
                      value={formik.values.firstName}
                      onChange={formik.handleChange}
                      error={formik.touched.firstName && Boolean(formik.errors.firstName)}
                      helperText={formik.touched.firstName && formik.errors.firstName}
                      sx={{ mb: 2 }}
                    />
                    <TextField
                      fullWidth
                      label="Last Name"
                      name="lastName"
                      value={formik.values.lastName}
                      onChange={formik.handleChange}
                      error={formik.touched.lastName && Boolean(formik.errors.lastName)}
                      helperText={formik.touched.lastName && formik.errors.lastName}
                      sx={{ mb: 2 }}
                    />
                  </>
                ) : (
                  <>
                    <Typography variant="h5" component="h2" fontWeight="medium">
                      {user?.firstName} {user?.lastName}
                    </Typography>
                    <Typography variant="body1" color="text.secondary">
                      {user?.email}
                    </Typography>
                  </>
                )}
              </Box>

              <Divider sx={{ my: 3 }} />

              {/* Account Security Section */}
              <Typography variant="h6" gutterBottom fontWeight="bold">
                Account Security
              </Typography>
              {editMode ? (
                <>
                  <TextField
                    fullWidth
                    label="Email"
                    name="email"
                    value={formik.values.email}
                    onChange={formik.handleChange}
                    error={formik.touched.email && Boolean(formik.errors.email)}
                    helperText={formik.touched.email && formik.errors.email}
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <EmailIcon />
                        </InputAdornment>
                      ),
                    }}
                    sx={{ mb: 2 }}
                  />
                  <TextField
                    fullWidth
                    label="Password"
                    type={showPassword ? 'text' : 'password'}
                    value="********"
                    disabled
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <PersonIcon />
                        </InputAdornment>
                      ),
                      endAdornment: (
                        <InputAdornment position="end">
                          <IconButton
                            onClick={handlePasswordVisibility}
                            edge="end"
                          >
                            {showPassword ? <VisibilityOff /> : <Visibility />}
                          </IconButton>
                        </InputAdornment>
                      ),
                    }}
                    sx={{ mb: 2 }}
                  />
                  <FormControlLabel
                    control={
                      <Switch
                        checked={formik.values.twoFactorAuth}
                        onChange={formik.handleChange}
                        name="twoFactorAuth"
                        color="primary"
                      />
                    }
                    label="Two-Factor Authentication"
                    sx={{ mb: 2 }}
                  />
                </>
              ) : (
                <>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <EmailIcon sx={{ mr: 2, color: 'text.secondary' }} />
                    <Typography>{user?.email}</Typography>
                  </Box>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <PersonIcon sx={{ mr: 2, color: 'text.secondary' }} />
                    <Typography>Password: ********</Typography>
                  </Box>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <Typography sx={{ mr: 2 }}>Two-Factor Auth:</Typography>
                    <Typography color={user?.twoFactorAuth ? 'success.main' : 'text.secondary'}>
                      {user?.twoFactorAuth ? 'Enabled' : 'Disabled'}
                    </Typography>
                  </Box>
                </>
              )}
            </Box>

            {/* Right Column - Detailed Information */}
            <Box sx={{ width: { xs: '100%', md: '70%' } }}>
              {/* Contact Information */}
              <Typography variant="h6" gutterBottom fontWeight="bold">
                Contact Information
              </Typography>
              <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={12} sm={6}>
                  {editMode ? (
                    <TextField
                      fullWidth
                      label="Phone Number"
                      name="phone"
                      value={formik.values.phone}
                      onChange={formik.handleChange}
                      error={formik.touched.phone && Boolean(formik.errors.phone)}
                      helperText={formik.touched.phone && formik.errors.phone}
                      InputProps={{
                        startAdornment: (
                          <InputAdornment position="start">
                            <PhoneIcon />
                          </InputAdornment>
                        ),
                      }}
                    />
                  ) : (
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <PhoneIcon sx={{ mr: 2, color: 'text.secondary' }} />
                      <Typography>{user?.phone || 'Not provided'}</Typography>
                    </Box>
                  )}
                </Grid>
                <Grid item xs={12} sm={6}>
                  {editMode ? (
                    <TextField
                      fullWidth
                      label="Birth Date"
                      type="date"
                      name="birthDate"
                      value={formik.values.birthDate}
                      onChange={formik.handleChange}
                      InputLabelProps={{ shrink: true }}
                      InputProps={{
                        startAdornment: (
                          <InputAdornment position="start">
                            <CalendarIcon />
                          </InputAdornment>
                        ),
                      }}
                    />
                  ) : (
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <CalendarIcon sx={{ mr: 2, color: 'text.secondary' }} />
                      <Typography>{user?.birthDate || 'Not provided'}</Typography>
                    </Box>
                  )}
                </Grid>
              </Grid>

              {/* Address Information */}
              <Typography variant="h6" gutterBottom fontWeight="bold" mt={4}>
                Address Information
              </Typography>
              {editMode ? (
                <Grid container spacing={2} sx={{ mb: 3 }}>
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="Street Address"
                      name="address.street"
                      value={formik.values.address.street}
                      onChange={formik.handleChange}
                      InputProps={{
                        startAdornment: (
                          <InputAdornment position="start">
                            <LocationIcon />
                          </InputAdornment>
                        ),
                      }}
                    />
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <TextField
                      fullWidth
                      label="City"
                      name="address.city"
                      value={formik.values.address.city}
                      onChange={formik.handleChange}
                    />
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <FormControl fullWidth>
                      <InputLabel>Country</InputLabel>
                      <Select
                        label="Country"
                        name="address.country"
                        value={formik.values.address.country}
                        onChange={formik.handleChange}
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
                      value={formik.values.address.zipCode}
                      onChange={formik.handleChange}
                    />
                  </Grid>
                </Grid>
              ) : (
                <Box sx={{ mb: 3 }}>
                  {user?.address?.street ? (
                    <>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        <LocationIcon sx={{ mr: 2, color: 'text.secondary' }} />
                        <Typography>{user.address.street}</Typography>
                      </Box>
                      <Box sx={{ display: 'flex', gap: 2, ml: 4 }}>
                        <Typography>{user.address.city}</Typography>
                        <Typography>{user.address.country}</Typography>
                        <Typography>{user.address.zipCode}</Typography>
                      </Box>
                    </>
                  ) : (
                    <Typography color="text.secondary">No address provided</Typography>
                  )}
                </Box>
              )}

              {/* Additional Information */}
              <Typography variant="h6" gutterBottom fontWeight="bold" mt={4}>
                Additional Information
              </Typography>
              {editMode ? (
                <>
                  <TextField
                    fullWidth
                    label="Bio"
                    name="bio"
                    value={formik.values.bio}
                    onChange={formik.handleChange}
                    multiline
                    rows={4}
                    sx={{ mb: 2 }}
                  />
                  <FormControlLabel
                    control={
                      <Switch
                        checked={formik.values.notifications}
                        onChange={formik.handleChange}
                        name="notifications"
                        color="primary"
                      />
                    }
                    label="Email Notifications"
                  />
                </>
              ) : (
                <>
                  <Typography paragraph sx={{ mb: 2 }}>
                    {user?.bio || 'No bio provided'}
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <Typography sx={{ mr: 2 }}>Email Notifications:</Typography>
                    <Typography color={user?.notifications ? 'success.main' : 'text.secondary'}>
                      {user?.notifications ? 'Enabled' : 'Disabled'}
                    </Typography>
                  </Box>
                </>
              )}
            </Box>
          </Box>
        </CardContent>
      </Card>

      {/* Avatar Selection Dialog */}
      {avatarDialogOpen && (
        <Dialog open={avatarDialogOpen} onClose={() => setAvatarDialogOpen(false)}>
          <DialogTitle>Select an Avatar</DialogTitle>
          <DialogContent>
            <List>
              {avatarOptions.map((avatar) => (
                <ListItem 
                  button 
                  key={avatar.id}
                  onClick={() => handleAvatarSelect(avatar.path)}
                >
                  <ListItemAvatar>
                    <Avatar src={avatar.path} sx={{ width: 56, height: 56 }} />
                  </ListItemAvatar>
                  <ListItemText primary={avatar.label} />
                  {formik.values.avatar === avatar.path && <CheckIcon color="primary" />}
                </ListItem>
              ))}
            </List>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setAvatarDialogOpen(false)}>Cancel</Button>
          </DialogActions>
        </Dialog>
      )}
    </Box>
  );
}