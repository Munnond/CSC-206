import base64
import io 
from scipy.optimize import minimize, Bounds, LinearConstraint
from numpy.linalg import norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from dateutil.parser import parse
import math
from flask import Flask, jsonify, render_template, request

# Load data
PCA_Predicted_Prices = pd.read_csv('optimization/PCA_Predicted_Prices1.csv')
PCA_Predicted_Prices['Date'] = pd.to_datetime(PCA_Predicted_Prices['Date'])
PCA_Predicted_Prices = PCA_Predicted_Prices.set_index('Date')

PCA_Actual_Prices = pd.read_csv('optimization/PCA_Actual_Prices1.csv')
PCA_Actual_Prices['Date'] = pd.to_datetime(PCA_Actual_Prices['Date'])
PCA_Actual_Prices = PCA_Actual_Prices.set_index('Date')

PCA_Predicted_Returns = PCA_Predicted_Prices.apply(lambda x: np.log(x) - np.log(x.shift(1))).iloc[1:]
PCA_Actual_Returns = PCA_Actual_Prices.apply(lambda x: np.log(x) - np.log(x.shift(1))).iloc[1:]

app = Flask(__name__)

def mean_returns(df, length):
    mu = df.sum(axis=0)/length
    return mu

def monthdelta(date, delta):
    m, y = (date.month+delta) % 12, date.year + ((date.month)+delta-1) // 12
    if not m: m = 12
    d = min(date.day, [31,29 if y%4==0 and not y%400==0 else 28,31,30,31,30,31,31,30,31,30,31][m-1])
    new_date = (date.replace(day=d,month=m, year=y))
    return parse(new_date.strftime('%Y-%m-%d'))

def windowGenerator(dataframe, lookback, horizon, step, cummulative=False):
    if cummulative:
        c = lookback
        step = horizon

    initial = min(dataframe.index)
    windows = []
    horizons = []

    while initial <= monthdelta(max(dataframe.index), -lookback):
        windowStart = initial
        windowEnd = monthdelta(windowStart, lookback)
        if cummulative:
            windowStart = min(dataframe.index)
            windowEnd = monthdelta(windowStart, c) + timedelta(days=1)
            c += horizon
        horizonStart = windowEnd + timedelta(days=1)
        horizonEnd = monthdelta(horizonStart, horizon)

        lookbackWindow = dataframe[windowStart:windowEnd]
        horizonWindow = dataframe[horizonStart:horizonEnd]

        windows.append(lookbackWindow)
        horizons.append(horizonWindow)

        initial = monthdelta(initial, step)

    return windows, horizons

def actual_return(actual_returns, w):
    mean_return = mean_returns(actual_returns, actual_returns.shape[0])
    actual_covariance = actual_returns.cov()

    portfolio_returns = mean_return.T.dot(w)
    portfolio_variance = w.T.dot(actual_covariance).dot(w)
    return portfolio_returns, portfolio_variance

def scipy_opt(predicted_returns, actual_returns, lam1, lam2):
    mean_return = mean_returns(predicted_returns, predicted_returns.shape[0])
    predicted_covariance = predicted_returns.cov()

    def f(w):
        return -(mean_return.T.dot(w) - lam1*(w.T.dot(predicted_covariance).dot(w)) + lam2*norm(w, ord=1))
    
    opt_bounds = Bounds(0, 1)

    def h(w):
        return sum(w) - 1

    cons = ({
        'type': 'eq',
        'fun': lambda w: h(w)
    })

    sol = minimize(f,
                   x0=np.ones(mean_return.shape[0]),
                   constraints=cons,
                   bounds=opt_bounds,
                   options={'disp': False},
                   tol=10e-10)

    w = sol.x
    predicted_portfolio_returns = w.dot(mean_return)
    portfolio_STD = w.T.dot(predicted_covariance).dot(w)

    portfolio_actual_returns, portfolio_actual_variance = actual_return(actual_returns, w)
    sharpe_ratio = portfolio_actual_returns/np.sqrt(portfolio_actual_variance) 

    ret_dict = {'weights': w,
                'predicted_returns': predicted_portfolio_returns,
                'predicted_variance': portfolio_STD,
                'actual_returns': portfolio_actual_returns,
                'actual_variance': portfolio_actual_variance,
                'sharpe_ratio': sharpe_ratio}

    return ret_dict

def metrics(returns_series):
    """Calculate performance metrics from a series of returns"""
    sharpe = returns_series.mean() / returns_series.std() if returns_series.std() != 0 else 0
    annualized_sharpe = sharpe * math.sqrt(252)  # Assuming daily returns

    annualized_return = returns_series.mean() * 252  # Annualized return
    annualized_vol = returns_series.std() * math.sqrt(252)  # Annualized volatility

    max_drawdown = (returns_series.cumsum() - returns_series.cumsum().cummax()).min()

    return {
        "Annualized Return": round(annualized_return, 4),
        "Annualized Volatility": round(annualized_vol, 4),
        "Annualized Sharpe Ratio": round(annualized_sharpe, 4),
        "Maximum Drawdown": round(max_drawdown, 4)
    }

def plot_equity_curve(equity_series, timestamps):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(timestamps, equity_series)
    ax.set_title("Portfolio Equity Growth Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity ($)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    img_bytes = buf.read()
    buf.close()
    plt.close(fig)  
    
    return base64.b64encode(img_bytes).decode()

@app.route('/predict-portfolio', methods=['POST'])
def predict_portfolio():
    try:
        data = request.get_json()
        lookback = int(data['lookback'])
        horizon = int(data['horizon'])
        initial_equity = float(data['initial_equity'])
       
        if lookback <= 0 or horizon <= 0 or initial_equity <= 0:
            return jsonify({'error': 'Parameters must be positive values'}), 400

        pred_windows, pred_horizons = windowGenerator(PCA_Predicted_Returns, lookback, 1, 1)
        act_windows, act_horizons = windowGenerator(PCA_Actual_Returns, lookback, 1, 1)
        
        if len(act_horizons) < horizon:
            return jsonify({'error': f'Not enough data for the specified horizon. Maximum horizon: {len(act_horizons)}'}), 400

        start = len(act_horizons) - horizon
        returns, variance, sharperatio, timestamps, equity = [], [], [], [], [initial_equity]
        weights_history = []

        for i in range(start, start + horizon):
            r = scipy_opt(pred_horizons[i], act_horizons[i], 0.5, 2)
            returns.append(r['actual_returns'])
            variance.append(r['actual_variance'])
            sharperatio.append(r['sharpe_ratio'])
            timestamps.append(act_horizons[i].index[0])
            equity.append(equity[-1] * math.exp(r['actual_returns']))
            weights_history.append(r['weights'])
            print(i,"complete")

        returns_series = pd.Series(returns)
        performance_metrics = metrics(returns_series)
        
        graph = plot_equity_curve(equity[1:], timestamps)
        final_equity = equity[-1]
        
        asset_names = PCA_Actual_Returns.columns.tolist()
        formatted_weights = []
        
        for period_weights in weights_history:
            period_dict = {}
            for i, asset in enumerate(asset_names):
                if period_weights[i] > 0.01:  # Only show significant weights
                    period_dict[asset] = round(period_weights[i] * 100, 2)  # As percentage
            formatted_weights.append(period_dict)

        return jsonify({
            'portfolio_returns': [round(r, 4) for r in returns],
            'equity_growth': [round(e, 2) for e in equity[1:]],
            'final_equity': round(final_equity, 2),
            'initial_equity': initial_equity,
            'timestamps': [t.strftime('%Y-%m-%d') for t in timestamps],
            'equity_plot_base64': graph,
            'performance_metrics': performance_metrics,
            'weights_history': formatted_weights
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)