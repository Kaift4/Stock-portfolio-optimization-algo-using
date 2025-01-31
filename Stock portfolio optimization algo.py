import numpy as np
import yfinance as yf       # Pull stock prices
import pandas as pd         # Put data into dataframe table
from datetime import datetime, timedelta
from scipy.optimize import minimize

# List of tickers
tickers = ['SPY', 'BND', 'GLD', 'QQQ', 'VTI']

# Setting end date
end_date = datetime.today()

# Setting start time (8 years window)
start_date = end_date - timedelta(days=8*365)
print(f"Start date: {start_date}")

# Download adjusted close prices
# Using auto_adjust=True will adjust for dividends and stock splits
adj_close_df = pd.DataFrame()

# Download adjusted close prices for each ticker
for ticker in tickers:
    ticker_data = yf.Ticker(ticker)  # Create Ticker object for each symbol
    hist = ticker_data.history(start=start_date, auto_adjust=True, actions=False)  # Fetch adjusted data
    
    # Check if data is empty
    if hist.empty:
        print(f"No data found for {ticker}")
    else:
        # Print available columns for troubleshooting
        print(f"Columns for {ticker}: {hist.columns}")
        
        # Extract adjusted 'Close' price and store it in the dataframe
        adj_close_df[ticker] = hist['Close']

# Display the dataframe
print("\nAdjusted Close Prices DataFrame:")
print(adj_close_df)

#calculating the lognormal returns for each ticker 
log_returns = np.log(adj_close_df/adj_close_df.shift(1)) 

#dropping missing values 
#the vales wont get mixed up if there s data missing
log_returns = log_returns.dropna()    

#calculate the covariance matrix 
# this is how we measure risk in porfolio , calc std deviation and risk
#multiplied by 252 to annualise the value 
cov_matrix = log_returns.cov()*252 
print("this is the covariance matrix :\n", cov_matrix)      

#calculate portfolio std deviation 
def standard_deviation(weights,cov_matrix):
    variance = weights.T@cov_matrix@weights
    return np.sqrt(variance)                     #std deviation is square root of variance)
    


#calculate the expected return
def expected_returns(weights,log_returns):
    return np.sum(log_returns.mean()*weights)*252         #we are getting the avg daily returns and multiplying 252

#calculate sharpe ratio
def sharpe_ratio(weights,log_returns,_cov_matrix,risk_free_rate):
    return(expected_returns(weights,log_returns)-risk_free_rate)/standard_deviation(weights,cov_matrix)


#set the risk free rate 
risk_free_rate = 0.02
print("the risk free rate is :\n",risk_free_rate)

#define function to minimizw ( negative sharpe ratio)
def neg_sharpe_ratio(weights,log_returns,cov_matrix,risk_free_rate):
    return - sharpe_ratio(weights,log_returns,cov_matrix,risk_free_rate)

#set constrains and bounds 
constraints = {'type':'eq','fun':lambda weights:np.sum(weights)-1}
bounds = [(0,0.4)for _ in range(len(tickers))]                        #0 is lower bound , we cant go short in any of tht assests,cany sell assests the we dont own
                                                                       # 0.5 denotes we cant have more than 40% in a single security

#set initial weights 
initial_weights = np.array([1/len(tickers)]*len(tickers))
print( "Initial weights:\n",initial_weights)

#optimize weights to maximize the sharpe ratio
#SLSQP stands for sequential least squares quadratic programming , a numerical opt technique for solving non linear opt prblms with constraints
optimized_results = minimize(neg_sharpe_ratio,initial_weights ,args=(log_returns,cov_matrix,risk_free_rate),method='SLSQP', constraints=constraints, bounds=bounds)


#get optimal weights
optimal_weights=optimized_results.x

#display analytics of optimal portfolio 
print("Optimal Weights: ")
for ticker, weight in zip (tickers, optimal_weights):
    print (f" {ticker}: {weight:.4f}")

print()

optimal_portfolio_return = expected_returns(optimal_weights,log_returns)
optimal_portfolio_volatility=standard_deviation (optimal_weights, cov_matrix)
optimal_sharpe_ratio =sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate)
print(f"Expected Annual Return: {optimal_portfolio_return:.4f}")
print(f"Expected Volatility: {optimal_portfolio_volatility:.4f}")
print(f"Sharpe Ratio: {optimal_sharpe_ratio:.4f}")

#display final portfolio 
import matplotlib.pyplot as plt

#import requuired library 
import matplotlib.pyplot as plt

#create bar chart of optimal weights
plt.figure(figsize=(10,6))
plt.bar(tickers,optimal_weights)

#add labels and a title 
plt.xlabel('Assests')
plt.ylabel(' Optimal Weights')
plt.title('Optimal Portfolio Weights')

#display the chart 
plt.show