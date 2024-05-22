# Multiple Linear Regression in Python

This repository contains a Jupyter Notebook that demonstrates how to perform multiple linear regression using the `scikit-learn` library in Python. The notebook includes detailed steps for data exploration, model fitting, visualization, and evaluation, providing a comprehensive guide to understanding and applying multiple linear regression.

## Notebook Content Overview

### 1. Introduction
- Overview of multiple linear regression
- Importing necessary libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, and `scikit-learn`

### 2. Data Loading and Exploration
- Reading the dataset from a CSV file
- Exploring the dataset's structure and contents

### 3. Visualizing the Data
- Creating 3D scatter plots to visualize relationships between predictors and the response variable (miles per gallon)

### 4. Fitting a Multivariate Regression Model
- Splitting the dataset into predictors (X) and response (y)
- Using `train_test_split` to create training and testing datasets
- Fitting a multiple linear regression model using the training data
- Extracting and displaying the model's intercept and coefficients

### 5. Visualizing Regression Results
- Creating scatter plots with regression lines for key predictors
- Comparing the observed values with the model's predictions

### 6. Assessing Model Accuracy
- Training a simple linear regression model for comparison
- Calculating and comparing Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) for both models on training and testing data

## Key Code Snippets

### Data Loading and Preparation
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Data/regression_sprint/mtcars.csv', index_col=0)
df.head()
```

### Visualizing the Data
```python
# Create a 3D scatter plot
fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111, projection='3d')

ax.set_zlabel('MPG')
ax.set_xlabel('No. of Cylinders')
ax.set_ylabel('Weight (1000 lbs)')
ax.scatter(df['cyl'], df['wt'], df['mpg'])
plt.show()
```

### Fitting a Multiple Linear Regression Model
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Split the dataset into predictors and response
X = df.drop(['mpg'], axis=1)
y = df['mpg']

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

# Train the model
lm = LinearRegression()
lm.fit(X_train, y_train)

# Extract model parameters
beta_0 = float(lm.intercept_)
beta_js = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
print("Intercept:", beta_0)
beta_js
```

### Visualizing Regression Results
```python
fig, axs = plt.subplots(2, 2, figsize=(9,7))

axs[0,0].scatter(df['wt'], df['mpg'])
axs[0,0].plot(df['wt'], lm.intercept_ + lm.coef_[4]*df['wt'], color='red')
axs[0,0].title.set_text('Weight (wt) vs. mpg')

axs[0,1].scatter(df['disp'], df['mpg'])
axs[0,1].plot(df['disp'], lm.intercept_ + lm.coef_[1]*df['disp'], color='red')
axs[1,0].scatter(df['cyl'], df['mpg'])
axs[1,0].plot(df['cyl'], lm.intercept_ + lm.coef_[0]*df['cyl'], color='red')
axs[1,1].scatter(df['hp'], df['mpg'])
axs[1,1].plot(df['hp'], lm.intercept_ + lm.coef_[2]*df['hp'], color='red')

fig.tight_layout(pad=3.0)
plt.show()
```

### Assessing Model Accuracy
```python
from sklearn import metrics
import math

# Train a simple linear regression model for comparison
slr = LinearRegression()
slr.fit(X_train[['disp']], y_train)

# Calculate MSE and RMSE for both models
results_dict = {'Training MSE': {
                    "SLR": metrics.mean_squared_error(y_train, slr.predict(X_train[['disp']])),
                    "MLR": metrics.mean_squared_error(y_train, lm.predict(X_train))
                },
                'Test MSE': {
                    "SLR": metrics.mean_squared_error(y_test, slr.predict(X_test[['disp']])),
                    "MLR": metrics.mean_squared_error(y_test, lm.predict(X_test))
                },
                'Test RMSE': {
                    "SLR": math.sqrt(metrics.mean_squared_error(y_test, slr.predict(X_test[['disp']]))),
                    "MLR": math.sqrt(metrics.mean_squared_error(y_test, lm.predict(X_test)))
                }
               }

# Display results
results_df = pd.DataFrame(data=results_dict)
results_df
```

## Conclusion
This notebook provides a step-by-step guide to implementing multiple linear regression using Python's `scikit-learn` library. It covers data exploration, model training, visualization, and evaluation, helping you understand the process of building and assessing multiple linear regression models. Feel free to explore the notebook and apply these techniques to your own datasets. Contributions and feedback are welcome!
