import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#User Inputs
n = int(input("Enter number of assets in the portfolio: "))
tickers = []
weights = []

for i in range(n):
    ticker = input(f"Enter ticker symbol for asset #{i+1}: ").strip().upper()
    tickers.append(ticker)
    if i < n - 1:
        w_percent = float(input(f"Enter portfolio weight for {ticker} (in %): "))
        weights.append(w_percent / 100)

#Auto-calculate last weight
last_weight = 1 - sum(weights)
weights.append(last_weight)

print("\nTicker symbols:", tickers)
print("Portfolio weights:", weights)

#Download historical prices
start_date = '2023-01-01'
end_date = '2026-01-01'
data = yf.download(tickers, start=start_date, end=end_date)['Close']
data = data.dropna()

#Calculate daily returns
returns = data.pct_change().dropna()
weights = np.array(weights)

#Portfolio return
returns['Portfolio'] = returns.dot(weights)
portfolio = (1 + returns['Portfolio']).cumprod()

#Performance metrics
risk_free_rate = 0.01
rolling_max = portfolio.cummax()
drawdown = (portfolio - rolling_max) / rolling_max
max_drawdown = drawdown.min()

cumulative_return = portfolio.iloc[-1] - 1
annualized_return = (1 + cumulative_return) ** (252 / len(returns)) - 1
annualized_volatility = returns['Portfolio'].std() * np.sqrt(252)
sharpe_ratio = (returns['Portfolio'].mean() - risk_free_rate / 252) / returns['Portfolio'].std()

#Output
print("\n📊 Portfolio Performance Metrics")
print(f"Cumulative Return:       {cumulative_return:.2%}")
print(f"Annualized Return:       {annualized_return:.2%}")
print(f"Annualized Volatility:   {annualized_volatility:.2%}")
print(f"Sharpe Ratio:            {sharpe_ratio:.2f}")
print(f"Maximum Drawdown:        {max_drawdown:.2%}")

#Print weights and tickers
print("\n📌 Final Asset Allocation:")
for t, w in zip(tickers, weights):
    print(f"{t}: {w:.2%}")

#Plot equity curve
portfolio.plot(title="Portfolio Equity Curve", figsize=(10, 5), color='navy')
plt.xlabel("Date")
plt.ylabel("Portfolio Value (normalized)")
plt.grid(True)
plt.tight_layout()
plt.show()
