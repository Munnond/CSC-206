import axios from "axios"
import Cookies from "js-cookie"

// Set default headers for JSON content type instead of multipart/form-data
axios.defaults.headers.post["Content-Type"] = "application/json"
axios.defaults.headers.put["Content-Type"] = "application/json"
axios.defaults.headers.patch["Content-Type"] = "application/json"

const BackendClient = axios.create({
  baseURL: "http://localhost:8000/api/",
  withCredentials: true,
  headers: {
    "X-CSRFToken": Cookies.get("csrftoken"),
  },
})

BackendClient.interceptors.request.use(
  (config) => {
    // Update the X-CSRFToken header with the latest CSRF token from the cookie
    config.headers["X-CSRFToken"] = Cookies.get("csrftoken")
    
    // Special case for multipart/form-data (file uploads)
    if (config.data instanceof FormData) {
      config.headers["Content-Type"] = "multipart/form-data"
    }
    
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

export default BackendClient