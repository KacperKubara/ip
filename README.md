# Description
Final Year Individual Project at the University of Southampton. Awarded With a Zepler Project Prize

# Abstract
Machine learning can simplify and quicken the drug development process by predicting specific
properties of molecules. Recently, much progress has been made on predicting molecular features
with Convolutional Neural Networks (CNNs). However, CNNs require a fixed-size vector as
an input which implies that the 3D structure of the molecule has to be projected into the
lower-dimensional representation. This dimensionality reduction leads to information loss and a
worse explainability of the model. As the molecular data is innately represented by the graph
structures, convolutions over graphs might be a better approach. Graph Convolutional Networks
(GCNs) is a novel deep learning architecture that allows training models directly on graph-like
structures. This work applies GCNs to predict solubility which is a crucial molecular property
in the drug development process. The novelty of the work is that instead of focusing on the
accuracy of predictions, different explainability methods are applied to deepen our understanding
of molecular features that affect the solubility.

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
