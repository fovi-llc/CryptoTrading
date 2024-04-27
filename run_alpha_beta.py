import pandas as pd
from config import RUN

df = pd.read_csv("processed_data/%strain_test_data.csv" % RUN['folder'].replace('/', '_'), index_col=0)

"""
    .
    .   HOLD
    |
----------  beta

   BUY

----------  alpha
    |
    |
  ----- HOLD
    |
    |
----------  -alpha

   SELL

----------  -beta
    |
    .   HOLD
    .

"""

intl_hold = 0.85  # marks the threshold of hold strip
intl_buy_sell = 0.997  # marks the buy/sell upper/lower limits
alpha = df["pct_change"].abs().quantile(intl_hold)
beta = df["pct_change"].abs().quantile(intl_buy_sell)  # to ignore the outliers in the beta computation

print("alpha: %.4f" % alpha)
print("beta: %.4f" % beta)

