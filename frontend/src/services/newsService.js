// src/services/newsService.js
const BACKEND_API_URL = 'http://localhost:3000/api'; // Your backend endpoint

export const getFinancialNews = async () => {
  try {
    const response = await fetch(`${BACKEND_API_URL}/news`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    return data.articles || []; // Ensure we always return an array
    
  } catch (error) {
    console.error('Error fetching financial news:', error);
    
    // Return sample data if API fails (for development)
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