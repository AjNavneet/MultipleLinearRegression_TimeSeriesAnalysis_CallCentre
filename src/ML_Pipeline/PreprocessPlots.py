import matplotlib.pyplot as plt
import scipy
import pylab
from matplotlib import pyplot

# Create a class for preprocessing plots
class PreprocessPlots:

    def __init__(self, df_comp):
        """
        Initialize the PreprocessPlots class.
        :param df_comp: The input DataFrame.
        """
        # Plotting Healthcare data
        df_comp.Healthcare.plot(figsize=(20, 5), title="Healthcare")
        plt.savefig("../output/" + "healthcare.png")

        # Plotting Telecom data
        df_comp.Telecom.plot(figsize=(20, 5), title="Telecom")
        plt.savefig("../output/" + "telecome.png")

        # Plotting Banking data
        df_comp.Banking.plot(figsize=(20, 5), title="Banking")
        plt.savefig("../output/" + "banking.png")

        # Plotting Technology data
        df_comp.Technology.plot(figsize=(20, 5), title="Technology")
        plt.savefig("../output/" + "technology.png")

        # Plotting Insurance data
        df_comp.Insurance.plot(figsize=(20, 5), title="Insurance")
        plt.savefig("../output/" + "Insurance.png")

        # Plotting #noofchannels data
        df_comp["#noofchannels"].plot(figsize=(20, 5), title="#noofchannels")
        plt.savefig("../output/" + "noofchannels.png")

        # Plotting #ofphonelines data
        df_comp["#ofphonelines"].plot(figsize=(20, 5), title="#ofphonelines")
        plt.savefig("../output/" + "noofphoneline.png")

        # Density Plots for Banking data
        df_comp["Banking"].plot(kind='kde', figsize=(20, 10))
        pyplot.savefig("../output/" + "density_banking.png")

        # Density Plots for #noofchannels data
        df_comp["#noofchannels"].plot(kind='kde', figsize=(20, 10))
        pyplot.savefig("../output/" + "density_noofchannels.png")

        # Density Plots for #ofphonelines data
        df_comp["#ofphonelines"].plot(kind='kde', figsize=(20, 10))
        pyplot.savefig("../output/" + "density_noofphonelines.png")

        # QQ plot for Banking data
        scipy.stats.probplot(df_comp["Banking"], plot=pylab)
        plt.title("QQ plot for Banking")
        pylab.savefig("../output/" + "banking_qq.png")

        # QQ plot for #ofphonelines data
        scipy.stats.probplot(df_comp["#ofphonelines"], plot=pylab)
        plt.title("QQ plot for ofphonelines")
        pylab.savefig("../output/" + "noofchanel_qq.png")
