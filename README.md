#Optimal Portfolio Allocation using Modern Portfolio Theory
-This project implements a portfolio optimization algorithm based on Modern Portfolio Theory (MPT). It analyzes historical data to allocate weights across a selection of assets in a way that maximizes the Sharpe Ratio, balancing expected return against risk.

#Overview
-Assets considered: SPY, BND, GLD, QQQ, VTI
-Data source: Historical price data via yfinance

#Methodology:
-Computes log returns from adjusted closing prices
-Annualizes return and covariance to evaluate performance
-Uses Sharpe Ratio as the optimization objective
-Solves the constrained optimization using SLSQP

#Features
-Pulls 8 years of price data
-Calculates expected return, volatility, and risk
-Prevents short selling and overexposure via constraints
-Visualizes the optimal allocation using a bar chart

#Installation
-Make sure you have Python 3.x installed. Then install the required libraries:

#bash
pip install numpy pandas yfinance scipy matplotlib

#Running the Script
-bash
python app.py

---You’ll see the optimal weights printed in the console along with the expected return, volatility, and Sharpe ratio. A bar chart will also display the final portfolio allocation.

#Example Output
yaml
Optimal Weights:
 SPY: 0.2010
 BND: 0.2391
 GLD: 0.1573
 QQQ: 0.3026
 VTI: 0.1000

Expected Annual Return: 0.1364
Expected Volatility: 0.1497
Sharpe Ratio: 0.7776

#Notes
-The weights are subject to constraints:
-Must sum to 1 (fully invested)
-No short positions (minimum weight = 0)
-Maximum 40% allocation to any single asset
-The risk-free rate is assumed to be 2% for Sharpe Ratio calculations
-You can modify the tickers list or adjust constraints to experiment with other asset mixes
