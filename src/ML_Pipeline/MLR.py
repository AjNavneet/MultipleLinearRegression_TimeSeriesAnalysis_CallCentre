import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot

# Create a class for training multiple linear regression model
class MLR:

    def run(self, df_comp, lag=0):
        """
        Run the Multiple Linear Regression model.
        :param df_comp: The input DataFrame.
        :param lag: Lag value for differencing the data.
        :return:
        """
        self.df_comp = df_comp[["Banking", "#noofchannels", "#ofphonelines"]]
        if lag == 0:
            self.data_prep_nolag()
        else:
            self.df_comp["delta_"+str(lag)+"_Banking"] = self.df_comp.Banking.diff(lag)
            self.df_comp["delta_"+str(lag)+"_noofchannels"] = self.df_comp["#noofchannels"].diff(lag)
            self.df_comp["delta_"+str(lag)+"_ofphonelines"] = self.df_comp["#ofphonelines"].diff(lag)
            self.data_prep_lag(lag)
        self.train()

    def data_prep_nolag(self):
        """
        Prepare the data for training with no lag.
        :return:
        """
        # Train-test split
        test_size = 22
        df_train = self.df_comp[:-test_size]
        df_test = self.df_comp[-test_size:]

        # Plot the train and test sets
        plt.title('Train and Test Sets', size=20)
        plt.plot(df_train["Banking"], label='Training set')
        plt.plot(df_test["Banking"], label='Test set', color='orange')
        plt.legend()

        self.X_train = df_train[["#noofchannels", "#ofphonelines"]].values
        self.y_train = df_train["Banking"].values
        self.X_test = df_test[["#noofchannels", "#ofphonelines"]].values
        self.y_test = df_test["Banking"].values

    def data_prep_lag(self, lag):
        """
        Prepare the data for training with lag.
        :param lag: Lag value for differencing the data.
        :return:
        """
        # Plot density of delta values
        self.df_comp["delta_"+str(lag)+"_Banking"].plot(kind='kde')
        pyplot.savefig("../output/" + "density_delta"+str(lag)+"_banking.png")

        # Density Plots
        self.df_comp["delta_"+str(lag)+"_noofchannels"].plot(kind='kde')
        pyplot.savefig("../output/" + "density_delta"+str(lag)+"_noofchannels.png")

        # Density Plots
        self.df_comp["delta_"+str(lag)+"_ofphonelines"].plot(kind='kde')
        pyplot.savefig("../output/" + "density_delta"+str(lag)+"_phonelines.png")

        # Train-test split
        test_size = 22
        df_train = self.df_comp[:-test_size]
        df_test = self.df_comp[-test_size:]

        # Plot the train and test sets
        plt.title('Train and Test Sets', size=20)
        plt.plot(df_train["delta_"+str(lag)+"_Banking"], label='Training set')
        plt.plot(df_test["delta_"+str(lag)+"_Banking"], label='Test set', color='orange')
        plt.legend()

        self.X_train = df_train[["delta_"+str(lag)+"_noofchannels", "delta_"+str(lag)+"_ofphonelines"]].values[lag:]
        self.y_train = df_train["delta_"+str(lag)+"_Banking"].values[lag:]

        self.X_test = df_test[["delta_"+str(lag)+"_noofchannels", "delta_"+str(lag)+"_ofphonelines"]].values[lag:]
        self.y_test = df_test["delta_"+str(lag)+"_Banking"].values[lag:]

    def train(self):
        """
        Train a Linear Regression Model.
        :return:
        """
        lr_model = LinearRegression()
        lr_model.fit(self.X_train, self.y_train)
        t0 = time.time()
        lr_y = self.y_test
        lr_y_fit = lr_model.predict(self.X_train)
        lr_y_pred = lr_model.predict(self.X_test)
        tF = time.time()
        print('Time to train = %.2f seconds' % (tF - t0))
        lr_residuals = lr_y_pred - lr_y
        lr_rmse = np.sqrt(np.sum(np.power(lr_residuals, 2)) / len(lr_residuals))
        print('RMSE = %.2f' % lr_rmse)

        # Plot the results for the training set
        plt.figure(figsize=(20, 5))
        plt.plot(lr_y_fit)
        plt.plot(self.y_train, color='red')
        plt.savefig("../output/" + "LR_train.png")

        # Plot the results for the test set
        plt.figure(figsize=(20, 5))
        plt.plot(lr_y_pred)
        plt.plot(lr_y, color='red')
        plt.savefig("../output/" + "LR_test.png")
