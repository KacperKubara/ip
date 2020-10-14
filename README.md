# Description
Final Year Individual Project at the University of Southampton. Awarded With a Zepler Project Prize
# Installation tips
The conda environment has been saved as env_info.yaml
You can reproduce the environment by simply running the command:
`conda env create -f env_info.yaml`
It will create a conda enviroment with all libraries that were used for the project. If that doesn't work, install them manually from the file.
## Rdkit
Works only when downloaded with Conda.

## Tensorflow GPU
TF 2.0 and TF 1.14 Works only with CUDA 10.0. Version 10.1 doesn't work!

# Description of the folder structure and files
## data
contains a raw/preprocessed Wang dataset and solubility challenge dataset. It also contains custom classes that load the data used with the benchmarks.
## eda
Folder with custom classes that visualize the stats and distribution of features for the Wang dataset
## preprocessing
Folder with custom classes that preprocess dataset
notebooks
## benchmark1.py
First version of the method 2 from the report, without the grid search.
## benchmark1_2.py
Aka. method 2 from the report.
## benchmark1_3.py
Aka. method 3 from the report.
## config.py
Paths to data and important parameters
## scaffold_analyze.py
Script to analyze statistical properties from generated clusters with different splitting methods.
## utils.py
Functions to plot results
