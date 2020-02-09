# Description
Final Year Individual Project at the University of Southampton.
# Installation tips
The conda environment has been saved as env_info.yaml
You can reproduce the environment by simply running the command:
`conda env create -f env_info.yaml`
There is a GCN package that requires manual installation:
https://github.com/tkipf/gcn

## Other installation tips
Direct port 6006 from Tensorboard to your local host 16006 port so you can access it on your laptop
`ssh -L 16006:127.0.0.1:6006 kjk1u17@uglogin.soton.ac.uk`
## Tensorflow GPU
TF 2.0 and TF 1.14 Works only with CUDA 10.0. Version 10.1 doesn't work!

## Rdkit
Works only when downloaded with Conda.
# ToDo
*) Move preprocessing functionality from sol_challenge to the Preprocessor class - done
*) Do EDA for Solubility Challenge Dataset - done
*) Remove outliers from data for EDA - done (added option in EDA classes to remove outliers)
*) Finish docs - currently done
*) Create Preprocessing for SMILES - done for ECFP
*) Implement random search for all models in solubility challenge - not needed now

# Description of the folder structure
## eda
## models
## pipeline