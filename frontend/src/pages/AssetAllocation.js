"use client"

import { useEffect, useState } from "react"
import {
  Box,
  Typography,
  Paper,
  Grid,
  Divider,
  Card,
  CardContent,
  CardHeader,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  CircularProgress,
} from "@mui/material"
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  BarElement,
} from 'chart.js'
import { Doughnut, Bar } from "react-chartjs-2"

ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement)

const assetClasses = [
  { id: "stocks", name: "Stocks", color: "#1976d2", riskLevel: "High" },
  { id: "bonds", name: "Bonds", color: "#4caf50", riskLevel: "Medium" },
  { id: "cash", name: "Cash", color: "#9c27b0", riskLevel: "Low" },
  { id: "real_estate", name: "Real Estate", color: "#ff9800", riskLevel: "Medium-High" },
  { id: "commodities", name: "Commodities", color: "#f44336", riskLevel: "High" },
]

const fetchCurrentHoldings = async () => {
  await new Promise(resolve => setTimeout(resolve, 500));
  return {
    stocks: 45000,
    bonds: 30000,
    cash: 10000,
    real_estate: 10000,
    commodities: 3000,
  };
};

export default function AssetAllocationOverview() {
  const [loading, setLoading] = useState(true)
  const [currentHoldings, setCurrentHoldings] = useState(null)

  useEffect(() => {
    const loadData = async () => {
      try {
        const data = await fetchCurrentHoldings()
        setCurrentHoldings(data)
      } catch (err) {
        console.error("Failed to fetch holdings", err)
      } finally {
        setLoading(false)
      }
    }
    loadData()
  }, [])

  const getCurrentAllocationPercentages = () => {
    if (!currentHoldings) return null
    const total = Object.values(currentHoldings).reduce((sum, val) => sum + val, 0)
    return {
      stocks: (currentHoldings.stocks / total) * 100,
      bonds: (currentHoldings.bonds / total) * 100,
      cash: (currentHoldings.cash / total) * 100,
      real_estate: (currentHoldings.real_estate / total) * 100,
      commodities: (currentHoldings.commodities / total) * 100,
    }
  }

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: "right" },
      tooltip: {
        callbacks: {
          label: (context) => `${context.label}: ${context.raw?.toFixed(2)}%`,
        },
      },
    },
  }

  const barChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        max: 100,
        beginAtZero: true,
      },
    },
    plugins: {
      legend: {
        position: "top",
      },
    },
  }

  const currentPercentages = getCurrentAllocationPercentages()

  const doughnutChartData = currentHoldings && {
    labels: assetClasses.map((a) => a.name),
    datasets: [
      {
        data: assetClasses.map((a) => currentHoldings[a.id] || 0),
        backgroundColor: assetClasses.map((a) => a.color),
        borderWidth: 0,
      },
    ],
  }

  const barChartData = currentPercentages && {
    labels: assetClasses.map((a) => a.name),
    datasets: [
      {
        label: 'Current Allocation (%)',
        data: assetClasses.map((a) => currentPercentages[a.id] || 0),
        backgroundColor: assetClasses.map((a) => a.color),
      },
    ],
  }

  return (
    <Box sx={{ p: 2, bgcolor: "background.default", minHeight: "100vh" }}>
      <Paper elevation={3} sx={{ p: { xs: 2, md: 6 }, maxWidth: 1200, mx: "auto", mt: 4 }}>
        <Typography variant="h4" gutterBottom align="center" color="primary">
          Asset Allocation Overview
        </Typography>
        <Typography variant="body1" align="center" color="text.secondary" paragraph>
          A comprehensive breakdown of your current portfolio distribution
        </Typography>
        <Divider sx={{ my: 4 }} />

        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
            <CircularProgress />
          </Box>
        ) : (
          <Grid container spacing={4}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardHeader title="Current Allocation" />
                <Divider />
                <CardContent sx={{ height: 350 }}>
                  <Doughnut data={doughnutChartData} options={chartOptions} />
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card>
                <CardHeader title="Percentage Breakdown" />
                <Divider />
                <CardContent sx={{ height: 350 }}>
                  <Bar data={barChartData} options={barChartOptions} />
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12}>
              <Card>
                <CardHeader title="Detailed Holdings" />
                <Divider />
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Asset Class</TableCell>
                        <TableCell>Risk Level</TableCell>
                        <TableCell align="right">Current Amount</TableCell>
                        <TableCell align="right">Current %</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {assetClasses.map((asset) => (
                        <TableRow key={asset.id}>
                          <TableCell>
                            <Box sx={{ display: "flex", alignItems: "center" }}>
                              <Box
                                sx={{
                                  width: 12,
                                  height: 12,
                                  borderRadius: "50%",
                                  bgcolor: asset.color,
                                  mr: 1.5,
                                }}
                              />
                              {asset.name}
                            </Box>
                          </TableCell>
                          <TableCell>{asset.riskLevel}</TableCell>
                          <TableCell align="right">${currentHoldings[asset.id].toLocaleString()}</TableCell>
                          <TableCell align="right">{currentPercentages[asset.id].toFixed(2)}%</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Card>
            </Grid>
          </Grid>
        )}
      </Paper>
    </Box>
  )
}