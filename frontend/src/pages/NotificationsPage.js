// src/pages/NotificationsPage.js
import React, { useState, useEffect, useContext } from 'react';
import { 
  Box, 
  Typography, 
  List, 
  ListItem, 
  ListItemText, 
  ListItemAvatar, 
  Avatar, 
  Divider, 
  Button,
  Badge,
  IconButton,
  CircularProgress
} from '@mui/material';
import { Notifications as NotificationsIcon, TrendingUp as TrendingUpIcon, ArrowForward } from '@mui/icons-material';
import { UserContext } from '../context/UserContext';
import { getFinancialNews } from '../services/newsService';

export default function NotificationsPage() {
  const [notifications, setNotifications] = useState([]);
  const [news, setNews] = useState([]);
  const [loading, setLoading] = useState(true);
  const { user } = useContext(UserContext);

  useEffect(() => {
    const fetchAllNotifications = async () => {
      try {
        // Fetch financial news
        const financialNews = await getFinancialNews();
        const formattedNews = financialNews.map((article, index) => ({
          id: `news-${index}`,
          title: article.title,
          description: article.description || "Click to read more about this financial news",
          date: article.publishedAt,
          read: false,
          source: article.source?.name || "News Source",
          url: article.url,
          isNews: true
        }));
        
        setNews(formattedNews);
        setLoading(false);
      } catch (error) {
        console.error("Error fetching notifications:", error);
        setLoading(false);
      }
    };

    fetchAllNotifications();
  }, []);

  const markAsRead = (id) => {
    setNews(prev => 
      prev.map(item => 
        item.id === id ? { ...item, read: true } : item
      )
    );
  };

  const markAllAsRead = () => {
    setNews(prev => 
      prev.map(item => ({ ...item, read: true }))
    );
  };

  const unreadCount = news.filter(n => !n.read).length;

  const handleNotificationClick = (url) => {
    window.open(url, '_blank');
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1" fontWeight="bold">
          Notifications
          {unreadCount > 0 && (
            <Badge 
              badgeContent={unreadCount} 
              color="primary" 
              sx={{ ml: 2 }}
            />
          )}
        </Typography>
        <Button 
          variant="outlined" 
          onClick={markAllAsRead}
          disabled={unreadCount === 0}
        >
          Mark all as read
        </Button>
      </Box>

      <Typography variant="h6" gutterBottom sx={{ mt: 3, mb: 2 }}>
        Financial News
      </Typography>
      
      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress />
        </Box>
      ) : (
        <List>
          {news.slice(0, 10).map((article) => (
            <React.Fragment key={article.id}>
              <ListItem 
                alignItems="flex-start"
                sx={{ 
                  backgroundColor: article.read ? 'inherit' : 'rgba(62, 237, 191, 0.08)',
                  borderRadius: 2,
                  mb: 1
                }}
                secondaryAction={
                  <IconButton 
                    edge="end" 
                    onClick={() => {
                      markAsRead(article.id);
                      handleNotificationClick(article.url);
                    }}
                  >
                    <ArrowForward />
                  </IconButton>
                }
              >
                <ListItemAvatar>
                  <Avatar sx={{ bgcolor: 'secondary.main' }}>
                    <TrendingUpIcon />
                  </Avatar>
                </ListItemAvatar>
                <ListItemText
                  primary={
                    <Typography fontWeight={article.read ? 'normal' : 'bold'}>
                      {article.title}
                    </Typography>
                  }
                  secondary={
                    <>
                      <Typography
                        sx={{ display: 'inline' }}
                        component="span"
                        variant="body2"
                        color="text.primary"
                      >
                        {article.description}
                      </Typography>
                      <Typography variant="caption" display="block" sx={{ mt: 0.5 }}>
                        {new Date(article.date).toLocaleString()} • {article.source}
                      </Typography>
                    </>
                  }
                />
              </ListItem>
              <Divider variant="inset" component="li" />
            </React.Fragment>
          ))}
        </List>
      )}
    </Box>
  );
}