import pandas as pd
from preprocessing import SMILESEncoder
from config import PATH_WANG_DATA

if __name__ == "__main__":
    dataset = pd.read_csv(PATH_WANG_DATA)
    wang_task = ['SLN']
    
    # Converting from SLN to SMILES
    preprocessor = SMILESEncoder(wang_task)
    dataset[wang_task] = preprocessor.sln_to_smiles(dataset)
    dataset.to_csv(PATH_WANG_DATA[:-4] + "_smiles.csv", index=False)