""" Data distribution class"""
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from eda import EDABase
from config import PATH_RESULTS_EDA

class DataDistribution(EDABase):
    """ Data distribution class which visualizes
    the distribution of data

    Parameters
    ----------
    col_names: [str]
        Column names which should be used in data
        distribution visualization

    Attributes
    ----------

    """
    
    def __init__(self, col_names):
        self.col_names = col_names
    
    def run(self, X: pd.DataFrame) -> None:
        """ Runs specific EDA model
        
        Parameters
        ----------
        X : pandas DataFrame
            Data which has to be visualized

        Returns
        -------
        None
        """
        self.gen_histograms(X)

    def gen_histograms(self, X: pd.DataFrame) -> None:
        """Generates histogram for each feature
        and saves as a picture in the path folder
        
        Parameters
        ----------
        X : pandas DataFrame
            Data which has to be visualized

        Returns
        -------
        None
        """
        cols_all_names = X.columns
        for i, name in enumerate(cols_all_names):
            if name in self.col_names:
                ax_temp = sns.distplot(X.iloc[:, i],
                                       axlabel=cols_all_names[i])
                ax_temp.savefig(PATH_RESULTS_EDA + "/" 
                                + cols_all_names[i]+ ".png")