import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 

from eda import EDABase

class FeatureCorrelation(EDABase):
    """ Feature correlation class which visualizes the 
    feature correlation of specific data

    Parameters
    ----------
    col_names: [str]
        Column names which should be used in data
        distribution visualization

    figsize: tuple
        Size of the figure which will be saved as a file
    """

    def __init__(self, col_names, path_save, figsize=(10,10)):
        self.col_names = col_names
        self.figsize = figsize
        self.path_save = path_save
    
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
        print("Feature Correlation Heatmap")    
        self.gen_heatmap(X)
        print("Feature Correlation Visualization finished\n")

    def gen_heatmap(self, X: pd.DataFrame) -> None:
        """ Generates pearson and spearman correlation heatmap
        for features and saves it as a picture in the specified
        path folder.
        
        Parameters
        ----------
        X : pandas DataFrame
            Data which has to be visualized

        Returns
        -------
        None
        """
        data = X[self.col_names]

        plt.figure(figsize=self.figsize)
        
        ax_temp = sns.heatmap(data=data.corr())
        ax_temp.figure.savefig(self.path_save + "/"
                        + "heatmap_pearson.png", bbox_inches="tight")
        ax_temp.set_xticklabels(ax_temp.get_yticklabels(), rotation =90)
        plt.clf()

        ax_temp = sns.heatmap(data=data.corr(method="spearman"))
        ax_temp.figure.savefig(self.path_save + "/"
                        + "heatmap_spearman.png", bbox_inches="tight")
        ax_temp.set_xticklabels(ax_temp.get_yticklabels(), rotation =90)        
        plt.clf()