from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

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
# plt.figure(figsize=(10,6))
# correlation_values.plot(kind='barh')
# plt.title('Correlation with Tip Amount')
# plt.xlabel("Correlation")
# plt.ylabel("Features")
# plt.tight_layout()
# plt.show()


# ***** Dataset Train/Test Split ****
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)

# ***** Build a Decision Tree Regressor model with Scikit-Learn ****
dt_reg = DecisionTreeRegressor(criterion = 'squared_error', max_depth =4, random_state =35)

# Train the Decision Tree Regressor model
dt_reg.fit(x_train, y_train)

# ***** Evaluate the Scikit-Learn and Snap ML Decision Tree Regressor Models  ****

y_pred = dt_reg.predict(x_test)

mse_score = mean_squared_error(y_test, y_pred)
print("MSE score : {0:.3f}".format(mse_score))

r2_score = dt_reg.score(x_test, y_test)
print("R2 score : {0:.3f}".format(r2_score))

#  Remove 4 features which are not correlated with the target variable.
raw_data =raw_data.drop(['payment_type', 'VendorID', 'store_and_fwd_flag', 'improvement_surcharge'], axis =1)