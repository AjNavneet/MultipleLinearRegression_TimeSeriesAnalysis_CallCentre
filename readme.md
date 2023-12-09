# Time Series Analysis Using Multiple Linear Regression

## Business Objective

A time series is simply a series of data points ordered in time. In a time series, time is often the independent variable, and the goal is usually to make a forecast for the future. Time series data can be helpful for many applications in day-to-day activities like:

- Tracking daily, hourly, or weekly weather data
- Monitoring changes in application performance
- Medical devices to visualize vitals in real-time

Linear regression is widely used in practice and adapts naturally to even complex forecasting tasks.

---

## Data Description

We will be using "CallCentres" data. This data is at the month level wherein the calls are segregated at the domain level as the call center operates for various domains. There are also external regressors like the number of channels and the number of phone lines which essentially indicate the traffic prediction of the in-house analyst and the resources available.

There are about 130 rows and 8 columns in the dataset:
- Month, healthcare, telecom, banking, technology, insurance, no of phonelines, and no of channels.

The multiple linear regression model will be built using three variables: 
- banking (dependent variable) 
- no of phonelines
- no of channels (independent variables)

---

## Aim

This project aims to build a Multiple linear regression model for time series analysis on the given dataset.

---

## Tech Stack

- Language: `Python`
- Libraries: `pandas`, `numpy`, `matplotlib`, `scipy`, `scikit-learn`, `gplearn`

---

## Approach

1. Import the required libraries and read the dataset.
2. Data pre-processing:
   - Setting date as the index.
   - Setting frequency as month.
3. Exploratory Data Analysis (EDA):
   - Data Visualization.
4. Check for normality:
   - Density plots.
   - Q-Q plots.
5. Multiple linear regression model:
   - Train-test split.
   - Train the model.
   - Fit the model.
   - Make predictions.
   - Plot the results.
6. Residual analysis:
   - Remove autocorrelation with varying lag values.
   - Check for the normality of the variables.
   - Train and fit the model.
   - Make predictions and plot the results.
7. Symbolic regression model:
   - Create a model.
   - Train the model.
   - Fit the model.
   - Make predictions and plot the results.

---

## Modular Code Overview

1. **input**: It contains all the data that we have for analysis. The following CSV is used:
   - CallCenterData.xlsx

2. **src**: This is the most important folder of the project. This folder contains all the modularized code for all the above steps in a modularized manner. This folder consists of:
   - Engine.py
   - ML_Pipeline

   The ML_pipeline is a folder that contains all the functions put into different python files which are appropriately named. These python functions are then called inside the engine.py file.

1. **output**: The output folder contains all the visualization graphs. There are around 20 different plots. The symbolic regression model is saved in a pickle file. Similarly, you can save other models that can be used later.

2. **lib**: This is a reference folder. It contains the original iPython notebook.

---

