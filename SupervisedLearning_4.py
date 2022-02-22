# -*- coding: utf-8 -*-
"""
@author: renata.fernandes
"""

# Preprocessing and pipelines

# Import necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import scale, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import ElasticNet

#Exploring categorical features

# Read 'gapminder.csv' into a DataFrame: df
df = pd.read_csv('Dados/gm_2008_region.csv')

# Create a boxplot of life expectancy per region
df.boxplot('life', 'Region', rot=60)

# Show the plot
plt.show()


# Creating dummy variables
# Print the columns of df
print("Columns of Gapminder dataframe: {}".format(df.columns))

# Create dummy variables: df_region
df_region = pd.get_dummies(df)

# Print the columns of df_region
print("Columns of Gapminder dataframe with dummy variables: {}".format(df_region.columns))

# Create dummy variables with drop_first=True: df_region (drop_first=True to drop the unneeded dummy variable)
df_region = pd.get_dummies(df,drop_first=True)

# Print the new columns of df_region ( Columns without 'Region_America')
print("Columns of Gapminder dataframe with dummy variables (without the unneeded dummy variable): {}".format(df_region.columns))


#Regression with categorical features

y = df_region['life'].values
X = df_region.drop(columns=['life'])

# Instantiate a ridge regressor: ridge
ridge = Ridge(alpha=0.5, normalize=True)

# Perform 5-fold cross-validation: ridge_cv
ridge_cv = cross_val_score(ridge, X,y,cv=5)

# Print the cross-validated scores
print("Gapminder dataframe - Ridge regression cross-validated scores : {}".format(ridge_cv))


#Dropping missing data

df = pd.read_csv('Dados/house-votes-84.csv', sep=',')

# Convert '?' to NaN
df[df == '?'] = np.nan

# Print the number of NaNs
print("HouseVotes dataframe - Print the number of NaNs: {}".format(df.isnull().sum()))

# Print shape of original DataFrame
print("HouseVotes dataframe - Shape of Original DataFrame: {}".format(df.shape))

# Drop missing values and print shape of new DataFrame
df = df.dropna()

# Print shape of new DataFrame
print("HouseVotes dataframe - Shape of DataFrame After Dropping All Rows with Missing Values: {}".format(df.shape))


#Imputing missing data in a ML Pipeline I

# Setup the Imputation transformer: imp
# imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imp = SimpleImputer(missing_values= np.nan, strategy='most_frequent')

# Instantiate the SVC classifier: clf
clf = SVC()

# Setup the pipeline with the required steps: steps
steps = [('imputation', imp),
        ('SVM', clf)]

#Imputing missing data in a ML Pipeline II
# Setup the pipeline steps: steps
# steps = [('imputation', Imputer(missing_values='NaN', strategy='most_frequent', axis=0)),
#         ('SVM', SVC())]
steps = [('imputation', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
        ('SVM', SVC())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Reading data
df = pd.read_csv('Dados/house-votes-84.csv', sep=',')
# Convert '?' to NaN
df[df == '?'] = np.nan
df[df == 'y'] = 1
df[df == 'n'] = 0

# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party', axis=1).values

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# Fit the pipeline to the train set
pipeline.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = pipeline.predict(X_test)

# Compute metrics
print("HouseVotes dataframe - Classification report: {}".format(classification_report(y_test, y_pred)))


#Centering and scaling your data

# Reading data
# df = pd.read_csv('Dados/winequality-red.csv', sep=';')
df = pd.read_csv('Dados/white-wine.csv', sep=',')
# Create arrays for the features and the response variable
y = df['quality'].values
X = df.drop('quality', axis=1).values

# Scale the features: X_scaled
X_scaled = scale(X)

# Print the mean and standard deviation of the unscaled features
print("White-wine dataframe - Mean of Unscaled Features: {}".format(np.mean(X))) 
print("White-wine dataframe - Standard Deviation of Unscaled Features: {}".format(np.std(X)))

# Print the mean and standard deviation of the scaled features
print("White-wine dataframe - Mean of Scaled Features: {}".format(np.mean(X_scaled))) 
print("White-wine dataframe -Standard Deviation of Scaled Features: {}".format(np.std(X_scaled)))


#Centering and scaling in a pipeline

# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())]
        
# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)

# Fit the pipeline to the training set: knn_scaled
knn_scaled = pipeline.fit(X_train, y_train)

# Instantiate and fit a k-NN classifier to the unscaled data
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)

# Compute and print metrics
print('White-wine dataframe - Accuracy with Scaling: {}'.format(knn_scaled.score(X_test,y_test)))
print('White-wine dataframe - Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test,y_test)))


#Bringing it all together I: Pipeline for classification
# Setup the pipeline
steps = [('scaler', StandardScaler()),
          ('SVM', SVC())]

pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=21)

# Instantiate the GridSearchCV object: cv
cv = GridSearchCV(pipeline, parameters, cv=3)

# Fit to the training set
cv.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = cv.predict(X_test)

# Compute and print metrics
print("White-wine dataframe - GridSearchCV Accuracy: {}".format(cv.score(X_test, y_test)))
print("White-wine dataframe - GridSearchCV classification report: {}".format(classification_report(y_test, y_pred)))
print("White-wine dataframe - GridSearchCV Tuned Model Parameters: {}".format(cv.best_params_))


#Bringing it all together II: Pipeline for regression
ELNET_MAX_ITER = 100000
ELNET_tolerance = 0.5
# Setup the pipeline steps: steps
steps = [('imputation', SimpleImputer(missing_values=np.nan, strategy='mean')),
          ('scaler', StandardScaler()),
          ('elasticnet', ElasticNet(max_iter= ELNET_MAX_ITER, tol=ELNET_tolerance))]

# Create the pipeline: pipeline 
pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'elasticnet__l1_ratio':np.linspace(0,1,30)}

# Read 'gapminder.csv' into a DataFrame: df
df = pd.read_csv('Dados/gm_2008_region.csv')
# Create dummy variables: df_region
df_region = pd.get_dummies(df)

y = df_region['life'].values
X = df_region.drop(columns=['life'])

# Create train and test sets
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size=0.4, random_state=42)

# Create the GridSearchCV object: gm_cv (cv=3 - default)
gm_cv = GridSearchCV(pipeline, parameters)

# Fit to the training set
gm_cv.fit(X_train, y_train)

# Compute and print the metrics
r2 = gm_cv.score(X_test, y_test)
print("Gapminder dataframe - Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))
print("Gapminder dataframe - Tuned ElasticNet R squared: {}".format(r2))
