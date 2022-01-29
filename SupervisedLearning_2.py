# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 13:07:30 2022

@author: renata.fernandes
"""

#CHAPTER 2 - REGRESSION

# Import necessary modules
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame: df
df = pd.read_csv('Dados/gm_2008_region.csv')

# Create arrays for features and target variable
y = df['life'].values
X = df['fertility'].values

# Print the dimensions of y and X before reshaping
print("Dimensions of y before reshaping: ", y.shape)
print("Dimensions of X before reshaping: ", X.shape)

# Reshape X and y
y_reshaped = y.reshape(-1,1)
X_reshaped = X.reshape(-1,1) 

# Print the dimensions of y_reshaped and X_reshaped
print("Dimensions of y after reshaping: ", y_reshaped.shape)
print("Dimensions of X after reshaping: ", X_reshaped.shape)


# Mapa de calor - Seaborn's heatmap function
sns.heatmap(df.corr(), square=True, cmap='RdYlGn')

# Create the regressor: reg
reg = LinearRegression()

# Create the prediction space
X_fertility = X_reshaped
prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1,1)

# Fit the model to the data
reg.fit(X_fertility,y)

# Compute predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)

# Print R^2 
#print(reg.score(prediction_space, y_pred))
print(reg.score(X_fertility,y))

# Plot regression line
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()
