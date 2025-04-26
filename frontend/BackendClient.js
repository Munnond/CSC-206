import axios from "axios"
import Cookies from "js-cookie"

axios.defaults.headers.post["Content-Type"] = "multipart/form-data"

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
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

export default BackendClient