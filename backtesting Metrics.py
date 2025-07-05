import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Parameters
tickers = ['AAPL', 'MSFT']
start_date = '2023-01-01'
end_date = '2024-01-01'
risk_free_rate = 0.01  # Annual risk-free rate (1%)

#User-defined portfolio weights
weight_stock1 = 0.0  # AAPL weight
weight_stock2 = 1 - weight_stock1  # MSFT weight
weights = np.array([weight_stock1, weight_stock2])

#Download adjusted closing price
data = yf.download(tickers, start=start_date, end=end_date)['Close']
df = data.dropna()

#Compute daily returns
returns = df.pct_change().dropna()

#Portfolio daily returns
returns['Portfolio'] = returns.dot(weights)

#Cumulative performance
df_portfolio = (1 + returns['Portfolio']).cumprod()
rolling_max = df_portfolio.cummax()
drawdown = (df_portfolio - rolling_max) / rolling_max
max_drawdown = drawdown.min()

#Performance metrics
cumulative_return = df_portfolio.iloc[-1] - 1
annualized_return = (1 + cumulative_return) ** (252 / len(returns)) - 1
annualized_volatility = returns['Portfolio'].std() * np.sqrt(252)
sharpe_ratio = (returns['Portfolio'].mean() - risk_free_rate / 252) / returns['Portfolio'].std()

#Print results
print(f"📊 Portfolio Performance: {tickers[0]} ({weight_stock1*100:.0f}%) + {tickers[1]} ({weight_stock2*100:.0f}%)")
print(f"Cumulative Return:       {cumulative_return:.2%}")
print(f"Annualized Return:       {annualized_return:.2%}")
print(f"Annualized Volatility:   {annualized_volatility:.2%}")
print(f"Sharpe Ratio:            {sharpe_ratio:.2f}")
print(f"Maximum Drawdown:        {max_drawdown:.2%}")

#Plot equity curve ===
df_portfolio.plot(
    title=f'Equity Curve: {weight_stock1*100:.0f}% {tickers[0]} + {weight_stock2*100:.0f}% {tickers[1]}',
    figsize=(10, 5),
    color='purple'
)

plt.xlabel('Date')
plt.ylabel('Portfolio Value (Normalized)')
plt.grid(True)
plt.tight_layout()
plt.show()
