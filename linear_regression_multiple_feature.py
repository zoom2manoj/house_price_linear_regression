import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits


df_app = pd.read_csv('train.csv')

X_mulitple = df_app.drop(['MSZoning', 'Street', 'Alley',  'LotShape', 'LandContour', 'Utilities', 'LotConfig','LandSlope', 'HeatingQC', 'BsmtQual', 'MasVnrType',  'Neighborhood','Condition1','Condition2','BldgType','HouseStyle','LotConfig','LandSlope','Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond','BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu','GarageType', 'GarageFinish', 'GarageQual','GarageCond', 'PavedDrive', 'PoolQC', 'RoofStyle','RoofMatl', 'Exterior1st', 'Exterior2nd', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition'], axis=1)
y_sales = df_app.SalePrice
X_mulitple = X_mulitple.fillna(0)
y_cov_arr = np.array(y_sales)

corr_matrix = X_mulitple.corr()
corr_matrix["SalePrice"].sort_values(ascending=False)



# Linear regression

X_train, X_test, y_train, y_test = train_test_split(X_mulitple.drop('SalePrice',axis=1), y_sales, test_size=0.3, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Calculate R squared
y_pred = regressor.predict(X_test)
my_submission = pd.DataFrame({'Id': X_test.Id, 'SalePrice':y_pred})
my_submission.to_csv('submission.csv', index=False)
print('Linear Regression R squared": ',  regressor.score(X_test, y_test))


#Calculate root-mean-square error (RMSE)

lin_mse = mean_squared_error(y_pred, y_test)
lin_rmse = np.sqrt(lin_mse)
print('Linear Regression RMSE: ', lin_rmse)

#Calculate mean absolute error (MAE):

lin_mae = mean_absolute_error(y_pred, y_test)
print('Linear Regression MAE: ' ,  lin_mae)
