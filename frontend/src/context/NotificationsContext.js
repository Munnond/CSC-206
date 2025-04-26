// src/context/NotificationsContext.js
import { createContext, useState, useEffect } from 'react';
import { getFinancialNews } from '../api/newsApi';

const NotificationsContext = createContext();

const localNotifications = [
  // ... your local notifications
];

export const NotificationsProvider = ({ children }) => {
  const [notifications, setNotifications] = useState([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchNotifications = async () => {
      try {
        const news = await getFinancialNews();
        const formattedNews = news.map((article, index) => ({
          // format news articles
        }));
        
        const allNotifications = [...localNotifications, ...formattedNews];
        setNotifications(allNotifications);
        setUnreadCount(allNotifications.filter(n => !n.read).length);
      } catch (error) {
        console.error('Error loading notifications:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchNotifications();
  }, []);

  const markAsRead = (id) => {
    // ... implementation
  };

  const markAllAsRead = () => {
    // ... implementation
  };

  return (
    <NotificationsContext.Provider value={{ 
      notifications, 
      unreadCount, 
      loading,
      markAsRead,
      markAllAsRead
    }}>
      {children}
    </NotificationsContext.Provider>
  );
};

export default NotificationsContext;