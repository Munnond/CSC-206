import React, {useState, useEffect} from "react"
import {
  Avatar,
  Button,
  TextField,
  Paper,
  Box,
  Typography,
  Alert,
  Container,
  useTheme,
  IconButton,
  Divider,
  InputAdornment,
  Link, // Added this import
} from "@mui/material"
import LockOutlinedIcon from "@mui/icons-material/LockOutlined"
import GoogleIcon from "@mui/icons-material/Google"
import TwitterIcon from "@mui/icons-material/Twitter"
import ArrowForwardIcon from "@mui/icons-material/ArrowForward"
import Visibility from "@mui/icons-material/Visibility"
import VisibilityOff from "@mui/icons-material/VisibilityOff"
import {useNavigate} from "react-router-dom"
import BackendClient from "../BackendClient"
import {useGoogleLogin} from "@react-oauth/google"
import axios from "axios"

// Financial chart component for background
const FinancialChartBackground = () => {
  const [bars, setBars] = useState([])
  const theme = useTheme()

  useEffect(() => {
    // Generate random bars for the financial chart
    const generateBars = () => {
      const newBars = []
      for (let i = 0; i < 20; i++) {
        newBars.push({
          id: i,
          height: Math.random() * 80 + 20,
          color:
            Math.random() > 0.5
              ? theme.palette.primary.main
              : theme.palette.secondary.main,
          delay: Math.random() * 5,
        })
      }
      setBars(newBars)
    }

    generateBars()
    const interval = setInterval(generateBars, 3000)
    return () => clearInterval(interval)
  }, [theme])

  return (
    <Box
      sx={{
        position: "absolute",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        display: "flex",
        alignItems: "flex-end",
        justifyContent: "center",
        zIndex: 0,
        opacity: 0.15,
        overflow: "hidden",
      }}
    >
      {bars.map((bar) => (
        <Box
          key={bar.id}
          sx={{
            width: "4%",
            height: `${bar.height}%`,
            backgroundColor: bar.color,
            margin: "0 1%",
            transition: "height 1.5s ease-in-out",
            transitionDelay: `${bar.delay}s`,
            borderRadius: "4px 4px 0 0",
          }}
        />
      ))}
    </Box>
  )
}

function Copyright(props) {
  return (
    <Typography
      variant="body2"
      color="text.secondary"
      align="center"
      {...props}
    >
      {"Copyright © "}
      <Link color="inherit" href="/">
        IntelliWealth
      </Link>{" "}
      {new Date().getFullYear()}
      {"."}
    </Typography>
  )
}

export default function Login() {
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [error, setError] = useState("")
  const [showPassword, setShowPassword] = useState(false)
  const navigate = useNavigate()
  const theme = useTheme()

  const handleSubmit = (event) => {
    event.preventDefault()
    if (!email) {
      setError("Please enter your email")
      return
    }
    if (!password) {
      setError("Please enter your password")
      return
    }
    BackendClient.post("/auth/login/", {email, password})
      .then((response) => {
        // Store token in localStorage
        if (response.data && (response.data.key || response.data.token || response.data.access)) {
          const token = response.data.key || response.data.token || response.data.access;
          localStorage.setItem('authToken', token);
          
          // Set authorization header for future requests
          BackendClient.defaults.headers.common['Authorization'] = `Bearer ${token}`;
          
          // For debugging
          console.log("Token stored:", token);
        } else {
          console.warn("No token received in response:", response.data);
        }
        
        // Handle successful login
        console.log("Login successful:", response.data)
        navigate("/dashboard")
      })
      .catch((error) => {
        console.error("Login error:", error)
        if (error.response) {
          // The request was made and the server responded with a status code
          // that falls out of the range of 2xx
          if (error.response.status === 403) {
            setError("Access forbidden. Check your credentials or permissions.")
          } else {
            setError(`Error: ${error.response.data.detail || "Invalid email or password"}`)
          }
        } else if (error.request) {
          // The request was made but no response was received
          setError("No response from server. Please check your internet connection.")
        } else {
          // Something happened in setting up the request that triggered an Error
          setError(`Error: ${error.message}`)
        }
      })
  }

  const handleArrowClick = () => {
    if (!email) {
      setError("Please enter your email")
      return
    }
    if (!password) {
      setError("Please enter your password")
      return
    }
    BackendClient.post("/auth/login/", {email, password})
      .then((response) => {
        // Handle successful login
        console.log("Login successful:", response.data)
        navigate("/dashboard")
      })
      .catch((error) => {
        console.error("Login error:", error)
        if (error.response) {
          // The request was made and the server responded with a status code
          // that falls out of the range of 2xx
          if (error.response.status === 403) {
            setError("Access forbidden. Check your credentials or permissions.")
          } else {
            setError(`Error: ${error.response.data.detail || "Invalid email or password"}`)
          }
        } else if (error.request) {
          // The request was made but no response was received
          setError("No response from server. Please check your internet connection.")
        } else {
          // Something happened in setting up the request that triggered an Error
          setError(`Error: ${error.message}`)
        }
      })
    // navigate('/dashboard');
  }
  const googleLogin = useGoogleLogin({
    onSuccess: async (tokenResponse) => {
      console.log("Google Login Success:", tokenResponse)
      console.log("Access Token:", tokenResponse.access_token)

      BackendClient.post("/auth/google/", {
        access_token: tokenResponse.access_token,
      })
        .then((response) => {
          // Handle successful login
          console.log("Login successful:", response.data)
          console.log("Navigating to dashboard")
          // navigate("/dashboard")
          window.location.href = "/dashboard"
        })
        .catch((error) => {
          console.error("Google login error:", error)
          if (error.response) {
            // The request was made and the server responded with a status code
            // that falls out of the range of 2xx
            if (error.response.status === 403) {
              setError("Access forbidden. Your Google account may not have permission to access this application.")
            } else {
              setError(`Error: ${error.response.data.detail || "Failed to authenticate with Google"}`)
            }
          } else if (error.request) {
            // The request was made but no response was received
            setError("No response from server. Please check your internet connection.");
          } else {
            // Something happened in setting up the request that triggered an Error
            setError(`Error: ${error.message}`);
          }
        });
    },
    onError: (err) => console.error("Login Failed:", err),
    scope: "profile email https://www.googleapis.com/auth/drive.readonly",
  });
  const handleTogglePasswordVisibility = () => {
    setShowPassword(!showPassword)
  }

  return (
    <Box
      sx={{
        minHeight: "100vh",
        background: "linear-gradient(135deg, #0A0A0A 0%, #121212 100%)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: theme.spacing(2),
        position: "relative",
        overflow: "hidden",
      }}
    >
      <FinancialChartBackground />

      <Container maxWidth="sm">
        <Box
          sx={{
            backdropFilter: "blur(16px)",
            background: "rgba(25, 25, 25, 0.85)",
            borderRadius: 6,
            border: "1px solid rgba(255, 255, 255, 0.15)",
            padding: theme.spacing(4),
            boxShadow: "0 12px 40px rgba(0, 0, 0, 0.3)",
            maxWidth: 450,
            margin: "0 auto",
            position: "relative",
            overflow: "hidden",
            "&::before": {
              content: '""',
              position: "absolute",
              top: -50,
              left: -50,
              width: 100,
              height: 100,
              background: `radial-gradient(circle, ${theme.palette.primary.main} 33%, rgba(0, 0, 0, 0) 70%)`,
              borderRadius: "50%",
              filter: "blur(40px)",
            },
          }}
        >
          <Box
            sx={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              position: "relative",
              zIndex: 1,
            }}
          >
            <Avatar
              sx={{
                m: 1,
                bgcolor: theme.palette.primary.main,
                width: 56,
                height: 56,
              }}
            >
              <LockOutlinedIcon />
            </Avatar>

            <Typography
              component="h1"
              variant="h4"
              sx={{
                fontWeight: 700,
                mb: 1,
                color: "white",
                textAlign: "center",
                background: `linear-gradient(45deg, ${theme.palette.primary.main} 30%, ${theme.palette.secondary.main} 90%)`  ,
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
              }}
            >
              Welcome to IntelliWealth
            </Typography>

            <Typography
              variant="body2"
              sx={{
                color: "rgba(255, 255, 255, 0.7)",
                mb: 4,
                textAlign: "center",
              }}
            >
              Sign in to access your financial dashboard
            </Typography>

            <Box component="form" onSubmit={handleSubmit} sx={{width: "100%"}}>
              {error && (
                <Alert
                  severity="error"
                  sx={{
                    mb: 3,
                    borderRadius: theme.shape.borderRadius,
                    backgroundColor: "rgba(211, 47, 47, 0.2)",
                    color: "#ff8a80",
                    border: "1px solid rgba(211, 47, 47, 0.3)",
                  }}
                >
                  {error}
                </Alert>
              )}

              <Box sx={{mb: 2}}>
                <TextField
                  fullWidth
                  id="email"
                  label="Email Address"
                  variant="outlined"
                  value={email}
                  onChange={(e) => {
                    setEmail(e.target.value)
                    setError("")
                  }}
                  InputProps={{
                    sx: {
                      color: "white",
                      backgroundColor: "rgba(35, 35, 35, 0.5)",
                      border: "1px solid rgba(255, 255, 255, 0.1)",
                      borderRadius: 2,
                      "&:hover": {
                        backgroundColor: "rgba(45, 45, 45, 0.5)",
                      },
                    },
                  }}
                  InputLabelProps={{
                    sx: {
                      color: "rgba(255, 255, 255, 0.7)",
                    },
                  }}
                />
              </Box>

              <Box sx={{mb: 3}}>
                <TextField
                  fullWidth
                  id="password"
                  label="Password"
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(e) => {
                    setPassword(e.target.value)
                    setError("")
                  }}
                  onKeyPress={(e) => {
                    if (e.key === "Enter") {
                      handleArrowClick()
                    }
                  }}
                  InputProps={{
                    sx: {
                      color: "white",
                      backgroundColor: "rgba(35, 35, 35, 0.5)",
                      border: "1px solid rgba(255, 255, 255, 0.1)",
                      borderRadius: 2,
                      "&:hover": {
                        backgroundColor: "rgba(45, 45, 45, 0.5)",
                      },
                    },
                    endAdornment: (
                      <InputAdornment position="end">
                        <IconButton
                          onClick={handleTogglePasswordVisibility}
                          edge="end"
                          sx={{color: "rgba(255, 255, 255, 0.5)"}}
                        >
                          {showPassword ? <VisibilityOff /> : <Visibility />}
                        </IconButton>
                      </InputAdornment>
                    ),
                  }}
                  InputLabelProps={{
                    sx: {
                      color: "rgba(255, 255, 255, 0.7)",
                    },
                  }}
                />
              </Box>

              <Button
                fullWidth
                variant="contained"
                onClick={handleSubmit}
                endIcon={<ArrowForwardIcon />}
                sx={{
                  mb: 3,
                  py: 1.5,
                  fontWeight: 600,
                  fontSize: "1rem",
                  borderRadius: 2,
                  background: `linear-gradient(45deg, ${theme.palette.primary.main} 0%, ${theme.palette.primary.dark} 100%)`,
                  boxShadow: "0 4px 20px rgba(0, 0, 0, 0.2)",
                  "&:hover": {
                    background: `linear-gradient(45deg, ${theme.palette.primary.dark} 0%, ${theme.palette.primary.main} 100%)`,
                  },
                }}
              >
                Sign In
              </Button>

              <Box sx={{my: 3, display: "flex", alignItems: "center", gap: 2}}>
                <Divider
                  sx={{flex: 1, borderColor: "rgba(255, 255, 255, 0.1)"}}
                />
                <Typography variant="body2" color="rgba(255, 255, 255, 0.5)">
                  OR
                </Typography>
                <Divider
                  sx={{flex: 1, borderColor: "rgba(255, 255, 255, 0.1)"}}
                />
              </Box>

              <Button
                fullWidth
                variant="outlined"
                startIcon={<GoogleIcon />}
                onClick={() => googleLogin()}
                sx={{
                  mb: 2,
                  py: 1.5,
                  color: "white",
                  border: "1px solid rgba(255, 255, 255, 0.1)",
                  backgroundColor: "rgba(35, 35, 35, 0.5)",
                  borderRadius: 2,
                  "&:hover": {
                    backgroundColor: "rgba(45, 45, 45, 0.5)",
                    borderColor: "rgba(255, 255, 255, 0.2)",
                  },
                }}
              >
                Continue with Google
              </Button>

              <Button
                fullWidth
                variant="outlined"
                startIcon={<TwitterIcon />}
                onClick={() => alert("X authentication not implemented yet")}
                sx={{
                  mb: 3,
                  py: 1.5,
                  color: "white",
                  border: "1px solid rgba(255, 255, 255, 0.1)",
                  backgroundColor: "rgba(35, 35, 35, 0.5)",
                  borderRadius: 2,
                  "&:hover": {
                    backgroundColor: "rgba(45, 45, 45, 0.5)",
                    borderColor: "rgba(255, 255, 255, 0.2)",
                  },
                }}
              >
                Continue with X
              </Button>

              <Box
                sx={{mt: 2, display: "flex", justifyContent: "space-between"}}
              >
                <Link
                  href="/forgot-password"
                  variant="body2"
                  sx={{
                    color: "rgba(255, 255, 255, 0.7)",
                    textDecoration: "none",
                    "&:hover": {
                      color: theme.palette.primary.main,
                      textDecoration: "underline",
                    },
                  }}
                >
                  Forgot password?
                </Link>

                <Box sx={{display: "flex"}}>
                  <Typography
                    variant="body2"
                    color="rgba(255, 255, 255, 0.7)"
                    sx={{mr: 1}}
                  >
                    New user?
                  </Typography>
                  <Link
                    href="/register"
                    variant="body2"
                    sx={{
                      color: theme.palette.primary.main,
                      textDecoration: "none",
                      "&:hover": {
                        textDecoration: "underline",
                      },
                    }}
                  >
                    Sign up
                  </Link>
                </Box>
              </Box>
            </Box>
          </Box>
        </Box>
        <Copyright sx={{mt: 4, color: "rgba(255, 255, 255, 0.5)"}} />
      </Container>
    </Box>
  )
}