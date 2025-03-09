import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.ardl import ARDL


mtgox_df = pd.read_csv("Amihud_30d_data.csv")
mtgox_df = mtgox_df.dropna(subset=['Return','Volume (Currency)'])

ardl_model = ARDL(mtgox_df['Volume (Currency)'], lags=3, exog=mtgox_df[['Return']], order=(3,2)).fit()

# 打印模型摘要
print(ardl_model.summary())
# # 取目标变量（成交量）
# y = mtgox_df['Volume (Currency)']

# # 训练 AR 模型，假设阶数 p=3
# ar_model = AutoReg(y, lags=3).fit()

# # 打印模型摘要
# print(ar_model.summary())