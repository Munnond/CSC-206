import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Button,
  Card,
  CardContent,
  Grid,
  CircularProgress,
  Divider,
  Alert,
  TextField,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow
} from '@mui/material';
import { styled } from '@mui/material/styles';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import DateRangeIcon from '@mui/icons-material/DateRange';
import axios from 'axios';

// Stock row with sentiment indicator
const StockRow = ({ stock, positive = true }) => {
  const Icon = positive ? TrendingUpIcon : TrendingDownIcon;
  const color = positive ? '#4caf50' : '#f44336';
  
  return (
    <TableRow sx={{ '&:last-child td, &:last-child th': { border: 0 } }}>
      <TableCell>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <Box sx={{ 
            backgroundColor: color, 
            borderRadius: '50%', 
            p: 0.5, 
            display: 'flex',
            mr: 1
          }}>
            <Icon sx={{ color: 'white', fontSize: 18 }} />
          </Box>
          <Typography variant="body1" fontWeight="medium">
            {stock.Stock_Name}
          </Typography>
        </Box>
      </TableCell>
      <TableCell align="center">
        <Typography 
          variant="body2" 
          sx={{ 
            fontWeight: 'bold',
            color: stock.sentiment > 0 ? '#4caf50' : stock.sentiment < 0 ? '#f44336' : 'text.secondary'
          }}
        >
          {stock.sentiment.toFixed(4)}
        </Typography>
      </TableCell>
      <TableCell align="right">
        {stock.price ? (
          <Typography variant="body2">
            ${stock.price.toFixed(2)}
          </Typography>
        ) : (
          <Typography variant="body2" color="text.secondary">
            N/A
          </Typography>
        )}
      </TableCell>
    </TableRow>
  );
};

export default function MarketSentiment() {
  const [selectedDate, setSelectedDate] = useState('');
  const [loading, setLoading] = useState(false);
  const [sentimentData, setSentimentData] = useState(null);
  const [error, setError] = useState(null);

  // Date constraints - using dates that are likely to have data in the backend
  const minDate = '2022-04-01';
  const maxDate = '2023-04-01';

  const fetchSentimentData = async () => {
    if (!selectedDate) {
      setError("Please select a date");
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      // Using POST endpoint instead of GET as specified in the API documentation
      const response = await axios.post(`http://localhost:8000/api/news/sentiment/`, {
        date: selectedDate
      });
      
      setSentimentData(response.data);
      setLoading(false);
    } catch (err) {
      console.error("Error fetching sentiment data:", err);
      let errorMessage = "Failed to fetch sentiment data. Please try again.";
      
      if (err.response) {
        // Get specific error message from the backend if available
        if (err.response.data && err.response.data.error) {
          errorMessage = err.response.data.error;
        }
      }
      
      setError(errorMessage);
      setLoading(false);
    }
  };

  return (
    <Box sx={{ p: 2 }}>
      <Paper elevation={3} sx={{ p: 4 }}>
        <Typography variant="h4" gutterBottom align="center">
          Market Sentiment Analysis
        </Typography>
        <Typography variant="body1" paragraph align="center">
          Select a date to analyze Twitter sentiment for various stocks.
          We'll show you the most positive and negative sentiment stocks for the week ending on that date.
        </Typography>
        
        <Box sx={{ mb: 4 }}>
          <Grid container spacing={3} alignItems="center" justifyContent="center">
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                type="date"
                label="Select End Date"
                value={selectedDate}
                onChange={(e) => setSelectedDate(e.target.value)}
                InputProps={{
                  inputProps: { 
                    min: minDate,
                    max: maxDate
                  },
                  startAdornment: (
                    <DateRangeIcon sx={{ color: 'action.active', mr: 1 }} />
                  ),
                }}
                InputLabelProps={{ 
                  shrink: true 
                }}
                helperText={`Select a date between ${minDate} and ${maxDate}. We'll analyze the week prior to this date.`}
              />
            </Grid>
          </Grid>
          
          <Box sx={{ mt: 3, display: 'flex', justifyContent: 'center' }}>
            <Button
              variant="contained"
              onClick={fetchSentimentData}
              disabled={loading || !selectedDate}
              sx={{ px: 4, py: 1 }}
            >
              {loading ? <CircularProgress size={24} /> : 'Analyze Market Sentiment'}
            </Button>
          </Box>
          
          {error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {error}
            </Alert>
          )}
        </Box>
        
        {sentimentData && (
          <Box sx={{ mt: 4 }}>
            <Divider sx={{ mb: 4 }} />
            
            <Typography variant="h5" gutterBottom align="center">
              Market Sentiment Analysis Results
            </Typography>
            
            <Typography variant="subtitle1" gutterBottom align="center" color="text.secondary">
              Analysis period: {sentimentData.start_date} to {sentimentData.end_date}
            </Typography>
            
            <Grid container spacing={3} sx={{ mt: 2 }}>
              <Grid item xs={12} md={6}>
                <Card sx={{ height: '100%' }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                      <TrendingUpIcon sx={{ color: '#4caf50', mr: 1 }} />
                      Most Positive Sentiment Stocks
                    </Typography>
                    
                    <TableContainer>
                      <Table>
                        <TableHead>
                          <TableRow>
                            <TableCell>Stock</TableCell>
                            <TableCell align="center">Sentiment Score</TableCell>
                            <TableCell align="right">Current Price</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {sentimentData.top5.map((stock, index) => (
                            <StockRow key={index} stock={stock} positive={true} />
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Card sx={{ height: '100%' }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                      <TrendingDownIcon sx={{ color: '#f44336', mr: 1 }} />
                      Most Negative Sentiment Stocks
                    </Typography>
                    
                    <TableContainer>
                      <Table>
                        <TableHead>
                          <TableRow>
                            <TableCell>Stock</TableCell>
                            <TableCell align="center">Sentiment Score</TableCell>
                            <TableCell align="right">Current Price</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {sentimentData.bottom5.map((stock, index) => (
                            <StockRow key={index} stock={stock} positive={false} />
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
            
            <Box sx={{ mt: 4, p: 2, bgcolor: 'background.default', borderRadius: 1 }}>
              <Typography variant="body2" color="text.secondary">
                <strong>Note:</strong> Sentiment scores range from -1 (extremely negative) to 1 (extremely positive). 
                These scores are calculated based on Twitter data and may not reflect actual market performance.
              </Typography>
            </Box>
          </Box>
        )}
      </Paper>
    </Box>
  );
} 