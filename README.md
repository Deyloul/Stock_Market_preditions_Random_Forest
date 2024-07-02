# Stock_Market_preditions_Random_Forest

import numpy as np
import pandas as pd
import os
df_stock = pd.read_csv('C:/infolimpioavanzadoTarget.csv')
df_stock.head()
         date       open       high  ...      diff  INCREMENTO  TARGET
0  2022-01-03  17.799999  18.219000  ... -1.900001   -9.664295       0
1  2022-01-04  17.700001  18.309999  ... -1.379999   -7.247895       0
2  2022-01-05  17.580000  17.799999  ... -0.930000   -5.201344       0
3  2022-01-06  16.650000  16.879999  ... -0.360000   -2.177856       0
4  2022-01-07  16.219999  16.290001  ... -0.120000   -0.758054       0

[5 rows x 1285 columns]
df_stock.shape
(7781, 1285)
df_stock.describe()

              open         high  ...    INCREMENTO       TARGET
count  7781.000000  7781.000000  ...   7626.000000  7781.000000
mean     34.990220    35.655999  ...     -2.674224     0.183010
std      99.841502   101.451058  ...    268.268134     0.386699
min       0.410000     0.435000  ... -23399.465955     0.000000
25%       4.050000     4.130000  ...     -4.494383     0.000000
50%      10.080000    10.110000  ...     -0.304004     0.000000
75%      24.350000    24.500000  ...      2.812552     0.000000
max     795.739990   799.359985  ...    425.000000     1.000000

[8 rows x 1283 columns]
df_stock.isnull().sum()
date                  0
open                  0
high                  0
low                   0
close                 0
                   ... 
stochastic-kd-15    587
volumenrelativo     215
diff                155
INCREMENTO          155
TARGET                0
Length: 1285, dtype: int64
data_new = df_stock[['date','open', 'high', 'low', 'close']]
data_new
            date       open       high        low      close
0     2022-01-03  17.799999  18.219000  17.500000  17.760000
1     2022-01-04  17.700001  18.309999  17.620001  17.660000
2     2022-01-05  17.580000  17.799999  16.910000  16.950001
3     2022-01-06  16.650000  16.879999  16.139999  16.170000
4     2022-01-07  16.219999  16.290001  15.630000  15.710000
...          ...        ...        ...        ...        ...
7776  2022-12-23  23.250000  23.540001  23.250000  23.290001
7777  2022-12-27  23.350000  23.610001  23.250000  23.350000
7778  2022-12-28  23.450001  23.570000  23.219999  23.350000
7779  2022-12-29  23.330000  23.740000  23.330000  23.610001
7780  2022-12-30  23.680000  23.760000  23.610001  23.610001

[7781 rows x 5 columns]
data_new.isnull().sum()
date     0
open     0
high     0
low      0
close    0
dtype: int64
import seaborn as sns
import matplotlib.pyplot as plt

# Adding another column to have the start stock value for the next day #

data_new["tomorrow"]= data_new["close"].shift(-1)

            date       open       high        low      close   tomorrow
0     2022-01-03  17.799999  18.219000  17.500000  17.760000  17.660000
1     2022-01-04  17.700001  18.309999  17.620001  17.660000  16.950001
2     2022-01-05  17.580000  17.799999  16.910000  16.950001  16.170000
3     2022-01-06  16.650000  16.879999  16.139999  16.170000  15.710000
4     2022-01-07  16.219999  16.290001  15.630000  15.710000  15.860000
...          ...        ...        ...        ...        ...        ...
7776  2022-12-23  23.250000  23.540001  23.250000  23.290001  23.350000
7777  2022-12-27  23.350000  23.610001  23.250000  23.350000  23.350000
7778  2022-12-28  23.450001  23.570000  23.219999  23.350000  23.610001
7779  2022-12-29  23.330000  23.740000  23.330000  23.610001  23.610001
7780  2022-12-30  23.680000  23.760000  23.610001  23.610001        NaN

[7781 rows x 6 columns]

# Creating a target to return a boolean from the difference in stock value from previous day #
data_new["target"] = data_new["tomorrow"] > data_new["close"].astype(float)

            date       open       high        low      close   tomorrow  target
0     2022-01-03  17.799999  18.219000  17.500000  17.760000  17.660000   False
1     2022-01-04  17.700001  18.309999  17.620001  17.660000  16.950001   False
2     2022-01-05  17.580000  17.799999  16.910000  16.950001  16.170000   False
3     2022-01-06  16.650000  16.879999  16.139999  16.170000  15.710000   False
4     2022-01-07  16.219999  16.290001  15.630000  15.710000  15.860000    True
...          ...        ...        ...        ...        ...        ...     ...
7776  2022-12-23  23.250000  23.540001  23.250000  23.290001  23.350000    True
7777  2022-12-27  23.350000  23.610001  23.250000  23.350000  23.350000   False
7778  2022-12-28  23.450001  23.570000  23.219999  23.350000  23.610001    True
7779  2022-12-29  23.330000  23.740000  23.330000  23.610001  23.610001   False
7780  2022-12-30  23.680000  23.760000  23.610001  23.610001        NaN   False

[7781 rows x 7 columns]

# Import Machine learning library which has less tendency to overfit and can detect non linear patterns. #
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=200, min_samples_split= 100, random_state=1)

# Train set will have all the rows except the last 100 Test set will have only the last 100 rows #
train = data_new.iloc[:-100]
test = data_new.iloc[-100:]
predictors = ["open", "high", "low", "close"] # Exempt the datetime so that the model doesnt train on it. So as to prevent data leakage #
model.fit(train[predictors], train["target"])
RandomForestClassifier(min_samples_split=100, n_estimators=200, random_state=1)

# Checking level of accuracy with precision_score #

from sklearn.metrics import precision_score
pred = model.predict(test[predictors])
pred
array([False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False,  True,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False,  True,
       False, False, False, False, False,  True, False, False, False,
       False, False,  True,  True,  True, False, False, False,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
       False, False, False,  True,  True, False,  True,  True,  True,
        True])
