// src/api/newsApi.js
import axios from 'axios';

const API_BASE_URL = 'http://localhost:3000/api'; // Your backend endpoint

export const getFinancialNews = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/news`);
    return response.data.articles || [];
  } catch (error) {
    console.error('Error fetching news:', error);
    
    // Fallback to sample data if API fails
    return [
      {
        title: "Market Update: Stocks Reach Record High",
        description: "Global markets surged today following positive economic indicators.",
        publishedAt: new Date().toISOString(),
        url: "#",
        source: { name: "Financial Times" }
      },
      {
        title: "New Investment Opportunities",
        description: "Emerging markets show strong growth potential this quarter.",
        publishedAt: new Date().toISOString(),
        url: "#",
        source: { name: "Investor Daily" }
      }
    ];
  }
};