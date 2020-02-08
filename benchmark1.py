# Script implementing the 1st benchmark
import os

import pandas as pd 
import numpy as np
import tensorflow as tf
import deepchem as dc

from data import load_wang_data_gcn 

if __name__ == "__main__":
    wang_tasks, (train, valid, test), transformers = load_wang_data_gcn()
    print(f"wang_tasks: {wang_tasks}, train: {train}, transformers: {transformers}")