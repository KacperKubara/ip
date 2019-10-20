""" Utility functions for general, package-wide purposes"""
import numpy as np
def average(x: list) -> float:
    """ Averages all elements in the list
    
    Parameters
    ----------
    x : [int, float]
        list with int or float elements
    
    Returns
    -------
    float, np.nan:
        Returns float value when averaging is successful.
        np.nan otherwise
    """
    try:
        average = sum(x)/float(len(x))
    except (TypeError, ValueError):
        average = np.nan
    
    return average 