// src/context/UserContext.js
'use client'

import React, { createContext, useState, useContext, useEffect } from 'react';
import axios from 'axios';
import BackendClient from '../BackendClient';

// Configure axios defaults
// You might want to change this URL based on your configuration
axios.defaults.baseURL = 'http://localhost:8000/api/';
axios.defaults.headers.common['Content-Type'] = 'application/json';
axios.defaults.withCredentials = true; // Include cookies with requests

// Add request interceptor to include authentication token if available
axios.interceptors.request.use(
  config => {
    // Check for token in localStorage with different possible keys
    const token = localStorage.getItem('authToken') || 
                  localStorage.getItem('access_token') || 
                  localStorage.getItem('token');
    
    // Check for CSRF token (needed for Django)
    const csrfToken = document.cookie
      .split('; ')
      .find(cookie => cookie.startsWith('csrftoken='))
      ?.split('=')[1];
      
    if (csrfToken) {
      config.headers['X-CSRFToken'] = csrfToken;
    }
    
    if (token) {
      // Try both token formats that Django might expect
      config.headers.Authorization = `Bearer ${token}`;
      // Some Django setups use Token instead of Bearer
      // config.headers.Authorization = `Token ${token}`;
    }
    
    return config;
  },
  error => Promise.reject(error)
);

export const UserContext = createContext();

export const UserProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch user data when the provider loads
  useEffect(() => {
    fetchCurrentUser();
  }, []);

  // Function to fetch the current user's data
  const fetchCurrentUser = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Get token from localStorage if available
      const token = localStorage.getItem('authToken');
      
      // Set up request config with authorization if token exists
      const config = {};
      if (token) {
        config.headers = {
          'Authorization': `Bearer ${token}`
        };
        // Also update default headers for future requests
        BackendClient.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      } else {
        console.warn("No auth token found in localStorage.");
      }
      
      const response = await BackendClient.get('/users/whoami/');
      console.log('User data from API:', response.data);
      
      if (response.data) {
        setUser(response.data);
        // Update authenticated state
        setIsAuthenticated(true);
      } else {
        throw new Error('Empty response received');
      }
    } catch (err) {
      console.error('Error fetching current user:', err);
      setError(err.message || 'Failed to fetch user data');
      // Clear token if unauthorized
      if (err.response && (err.response.status === 401 || err.response.status === 403)) {
        localStorage.removeItem('authToken');
        setIsAuthenticated(false);
      }
    } finally {
      setLoading(false);
    }
  };

  // Function to update user information
  const updateUser = async (userData) => {
    try {
      // Assuming you have an endpoint to update the user
      const response = await BackendClient.put('http://127.0.0.1:8000/api/users/update/', userData);
      console.log('User update response:', response.data);
      
      // Update the local user state with the new data
      setUser(response.data);
      
      return { success: true };
    } catch (error) {
      console.error('Error updating user:', error);
      return { 
        success: false, 
        error: error.response?.data || error.message || 'Failed to update user'
      };
    }
  };

  // Check if user is authenticated
  const [isAuthenticated, setIsAuthenticated] = useState(() => {
    if (typeof window !== 'undefined') {
      return !!(localStorage.getItem('authToken') || 
                localStorage.getItem('access_token') || 
                localStorage.getItem('token'));
    }
    return false;
  });

  // Function to set authentication token
  const setAuthToken = (token) => {
    if (token) {
      localStorage.setItem('authToken', token);
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      setIsAuthenticated(true);
    } else {
      localStorage.removeItem('authToken');
      delete axios.defaults.headers.common['Authorization'];
      setIsAuthenticated(false);
    }
  };

  function getDefaultUser() {
    return {
      "user": {
          "id": 1,
          "email": "iammiddha@gmail.com",
          "username": "iammiddha",
          "is_client": false,
          "is_financial_advisor": false,
          "financial_advisor": null
      },
      "profile": {
        "id": 1,
        "user": {
            "id": 1,
            "email": "iammiddha@gmail.com",
            "username": "iammiddha",
            "is_client": false,
            "is_financial_advisor": false,
            "financial_advisor": null
        },
        "risk_tolerance": "high",
        "risk_tolerance_display": "High",
        "investment_horizon": 12,
        "investment_horizon_display": "Less than 1 year",
        "primary_investment_goal": "income_generation",
        "primary_investment_goal_display": "Income Generation",
        "familiarity_with_market": 1,
        "familiarity_with_market_display": "Somewhat familiar",
        "percentage_comforatable_savings": 3,
        "percentage_comforatable_savings_display": "0-5% of Income",
        "age": 50,
        "annual_income": 300000,
        "target_return": 30.0,
        "preferred_assets": [
            "AAPL",
            "MSFT"
        ],
        "profile_picture": null,
        "phone": "8146812840",
        "address": "H:58, Peepal Colony, New Court Road Mansa, Punjab\r\nPeepal Coloney",
        "birth_date": "2003-10-19",
        "avatar": 0,
        "name": "Samrat Middha"  
      }
    };
  }

  // Function to login a user with token authentication
  const login = async (credentials) => {
    try {
      // Example login request - adjust endpoint as needed
      const response = await axios.post('auth/login/', credentials);
      const token = response.data.key || response.data.token || response.data.access;
      
      if (token) {
        setAuthToken(token);
        
        // Fetch user data after login
        const userResponse = await axios.get('users/whoami/');
        setUser(userResponse.data);
        localStorage.setItem('user', JSON.stringify(userResponse.data));
        
        return { success: true };
      } else {
        throw new Error('No authentication token received');
      }
    } catch (error) {
      console.error("Login error:", error);
      return { 
        success: false, 
        error: error.response?.data || error.message || 'Login failed'
      };
    }
  };
  
  // Function for Google authentication
  const loginWithGoogle = async (googleToken) => {
    try {
      // Exchange Google token for our backend token
      const response = await axios.post('auth/google/', { 
        access_token: googleToken 
      });
      
      const token = response.data.key || response.data.token || response.data.access;
      
      if (token) {
        setAuthToken(token);
        
        // Fetch user data after login
        const userResponse = await axios.get('users/whoami/');
        setUser(userResponse.data);
        localStorage.setItem('user', JSON.stringify(userResponse.data));
        
        return { success: true };
      } else {
        throw new Error('No authentication token received from Google login');
      }
    } catch (error) {
      console.error("Google login error:", error);
      return {
        success: false,
        error: error.response?.data || error.message || 'Google login failed'
      };
    }
  };
  
  // Function to initiate Google login flow
  const initiateGoogleLogin = () => {
    // This implementation depends on whether you're using the Google OAuth2 directly or a library
    // Example with a redirect flow:
    const googleClientId = "YOUR_GOOGLE_CLIENT_ID"; // Replace with your actual client ID
    const redirectUri = encodeURIComponent(window.location.origin + "/auth/google/callback");
    const scope = encodeURIComponent("email profile");
    
    const googleAuthUrl = `https://accounts.google.com/o/oauth2/auth?client_id=${googleClientId}&redirect_uri=${redirectUri}&scope=${scope}&response_type=token`;
    
    // Open Google login in a new window or redirect
    window.location.href = googleAuthUrl;
    
    // Alternatively, you might use a Google authentication library
    // that handles the flow for you
  };
  
  // Function to logout
  const logout = async () => {
    try {
      // Call logout endpoint if available
      await axios.post('auth/logout/');
    } catch (error) {
      console.error("Logout error:", error);
    } finally {
      // Clear local storage and state regardless of API response
      setAuthToken(null);
      setUser(getDefaultUser());
      localStorage.removeItem('user');
    }
  };

  return (
    <UserContext.Provider 
      value={{ 
        user, 
        setUser, 
        updateUser, 
        loading, 
        error,
        refetchUser: fetchCurrentUser,
        isAuthenticated,
        login,
        loginWithGoogle,
        initiateGoogleLogin,
        logout,
        setAuthToken
      }}
    >
      {children}
    </UserContext.Provider>
  );
};

export function useUser() {
  return useContext(UserContext);
}