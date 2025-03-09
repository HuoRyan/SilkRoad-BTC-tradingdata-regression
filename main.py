#Import necessary libraries, pandas Used for data manipulation and operations, numpy Provides numerical computing functions, 
#matplotlib Used for plotting data visualization charts,sklearn.linear_model, statsmodels, linearmodels Import linear regression model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from statsmodels.tsa.ardl import ARDL

#Load datasets from CSV files into Pandas DataFrames
price_df = pd.read_csv("price_data.csv")
item_df = pd.read_csv("item.snappy.csv")
feedback_df = pd.read_csv("feedback.snappy.csv")
mtgox_df = pd.read_csv("BCHARTS-MTGOXUSD.csv")

# Convert timestamps in multiple datasets to datetime format and extract only the date (year, month, day) for consistency.    
# all relevant datasets are sorted by date. 
price_df['time'] = pd.to_datetime(price_df['time'], unit='s')
feedback_df['time'] = pd.to_datetime(feedback_df['feedback_time'], unit='s')
mtgox_df['Date'] = pd.to_datetime(mtgox_df['Date'])
feedback_df['date'] = feedback_df['time'].dt.date
price_df['date'] = price_df['time'].dt.date
mtgox_df['date'] = mtgox_df['Date'].dt.date
price_df = price_df.sort_values('date')
feedback_df = feedback_df.sort_values('date')
mtgox_df.sort_values(by='Date', inplace=True)



# sum the numbers in one day from price_df to obtain transaction prices.  
# filters relevant columns and aggregates daily BTC transaction volume (sum of prices) and transaction count (number of transactions).  
# merges exchange rate data (BTC/USD) from mtgox_df based on date, enabling the conversion of BTC trading volume into USD.  
# calculates the total USD transaction volume by multiplying the daily BTC volume with the corresponding BTC-USD exchange rate,  
daily = price_df.groupby('date').agg(
    btc_volume = ('price', 'sum'),        
    transaction_count = ('item_id', 'count')  
)
# print(daily)
# raise SystemExit("程序被终止")
daily = daily.merge(mtgox_df[['date', 'Weighted Price', 'Close']], on='date', how='left')
daily['usd_volume'] = daily['btc_volume'] * daily['Weighted Price']

# Compile a dataset where each row is a daily observation.
daily.to_csv("daily observation", index=False, encoding='utf-8')

# Create a time-series plot of BTC-denominated transaction volume over time.
plt.figure(figsize=(12, 6))
plt.plot(daily['date'], daily['btc_volume'], marker='o', linestyle='-')
plt.xlabel('Date')
plt.ylabel('BTC Volume')
plt.title('Daily BTC Transaction Volume Over Time')
plt.grid(True)
plt.savefig("Daily_BTC_Transaction_Volume_Over_Time.pdf", bbox_inches='tight')
plt.show()


# calculates and visualizes the Amihud illiquidity measure for BTC/USD trading on Mt. Gox.  
# computes daily returns as the percentage change in the closing price.  
# replaces infinite values with NaN to avoid computational issues.  
# The Amihud illiquidity measure is calculated as the absolute return divided by trading volume,  
# with NaN assigned when volume is zero to prevent division errors.   
# Finally, generates a time-series plot of the 30-day rolling Amihud measure
mtgox_df['Return'] = mtgox_df['Close'].pct_change()
mtgox_df.replace([np.inf, -np.inf], np.nan, inplace=True)
mtgox_df['Amihud_daily'] = mtgox_df.apply(
    lambda row: abs(row['Return']) / row['Volume (Currency)'] if row['Volume (Currency)'] > 0 else np.nan,
    axis=1
)
mtgox_df['Amihud_30d'] = mtgox_df['Amihud_daily'].rolling(window=30).mean()
print(mtgox_df)

plt.figure(figsize=(10, 6))
plt.plot(mtgox_df['Date'], mtgox_df['Amihud_30d'], label='30-day rolling Amihud')
plt.xlabel('Date')
plt.ylabel('Amihud (30-day average)')
plt.title('30-day Rolling Amihud Illiquidity - Mt.Gox BTC/USD')
plt.legend()
plt.grid(True)
plt.savefig("30-day_Rolling_Amihud_Illiquidity.pdf", bbox_inches='tight')
plt.show()


mtgox_df.to_csv("Amihud_30d_data.csv", index=False, encoding='utf-8')

#Regress contemporaneous (btc denominated) market volume on daily returns
# Initially, I used a linear regression model but found that the fit was too low. To improve it, I switched to an AutoReg(3) model. 
# Later, considering that Return might impact future trading volume with a lag effect, I adopted the ARDL(3,3) model. 
# The results showed a better fit, so I decided to use the ARDL model.
# Here is the linear regression model I used before:
# ============================================================================
# mtgox_df = mtgox_df.dropna(subset=['Return', 'Volume (Currency)'])
# y = mtgox_df['Volume (Currency)'] 
# X = sm.add_constant(mtgox_df['Return']) 
# ols_model = sm.OLS(y, X).fit()
# # pritn below are used for check model
# print(ols_model.summary())
# ============================================================================
mtgox_df = pd.read_csv("Amihud_30d_data.csv")
mtgox_df = mtgox_df.dropna(subset=['Return','Volume (Currency)'])
ardl_model = ARDL(mtgox_df['Volume (Currency)'], lags=5, exog=mtgox_df[['Return']], order=(3,2)).fit()
print(ardl_model.summary())





# An instrumental variables regression in which instrument daily BTC volume using one week-lagged silk road transaction volume.
mtgox_df['sr_volume_btc_lag7'] = mtgox_df['Volume (Currency)'].shift(7)
df_iv = mtgox_df.dropna(subset=['Return', 'Volume (Currency)', 'sr_volume_btc_lag7'])
y_iv = df_iv['Return']
X_iv = pd.DataFrame({'const': np.ones(len(df_iv))}, index=df_iv.index)
endog_iv = df_iv['Volume (Currency)']
instr_iv = df_iv['sr_volume_btc_lag7']
iv_model = IV2SLS(dependent=y_iv, exog=X_iv, endog=endog_iv, instruments=instr_iv).fit()
# print(iv_model.summary)