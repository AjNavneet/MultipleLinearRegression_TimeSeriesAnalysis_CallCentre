import pandas as pd
import pickle
from ML_Pipeline.PreprocessPlots import PreprocessPlots
from ML_Pipeline.MLR import MLR
from ML_Pipeline.SymbolicRegression import SymbolicRegression

# Importing the data from an Excel file
raw_csv_data = pd.read_excel("../input/CallCenterData.xlsx")

# Checking for missing values in the dataset
df_comp = raw_csv_data.copy()
print("Number of missing values in each column:")
print(df_comp.isna().sum())

# Setting the 'month' column as the index
df_comp.set_index("month", inplace=True)

# Setting the frequency of the dataset to be monthly
df_comp = df_comp.asfreq('M')
print("Number of missing values in each column after setting frequency:")
print(df_comp.isna().sum())

# Preprocess the data using the PreprocessPlots class
PreprocessPlots(df_comp)

# Multiple Linear Regression (MLR) analysis
multipleLR = MLR()

# Running MLR with lag 0
multipleLR.run(df_comp, lag=0)

# Running MLR with lag 1
multipleLR.run(df_comp, lag=1)

# Running MLR with lag 2
multipleLR.run(df_comp, lag=2)

# Symbolic regression analysis
symbolic_model = SymbolicRegression(df_comp)
print("Symbolic Regression Model Summary:")
print(symbolic_model)

# Saving the symbolic regression model in pickle format for future use
pickle.dump(symbolic_model, open("../output/symbolic_regression_model.pkl", "wb"))
