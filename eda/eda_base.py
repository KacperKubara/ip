"""
Generalized EDA models
"""
from abc import ABC, abstractmethod

import pandas as pd 


class EDABase(ABC):
    """Base class for EDA models"""

    @abstractmethod
    def run(self, X: pd.DataFrame) -> None:
        """ Runs specific EDA model"""
        pass