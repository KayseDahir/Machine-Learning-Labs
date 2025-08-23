from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/pu9kbeSaAtRZ7RxdJKX9_A/yellow-tripdata.csv'

raw_data = pd.read_csv(url)
# print(raw_data)


# ***** Dataset Preprocessing ****

y = raw_data[['tip_amount']].values.astype('float32')
proc_data = raw_data.drop(['tip_amount'], axis=1)
x= proc_data.values
x = normalize(x, axis=1, norm='l1', copy=False)

correlation_values = raw_data.corr()['tip_amount'].drop('tip_amount')
plt.figure(figsize=(10,6))
correlation_values.plot(kind='barh')
plt.title('Correlation with Tip Amount')
plt.xlabel("Correlation")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
