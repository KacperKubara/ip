import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


def find_duplicated_smiles(wang_path):
    wang_data = pd.read_csv(wang_path)
    smiles_list = wang_data["smiles"]
    smiles_dict = {}
    ix_duplicated = []
    for i in range(len(smiles_list.index)):
        smiles = smiles_list[i]
        if smiles in smiles_dict:
            smiles_dict[smiles] += 1
            ix_duplicated.append(i)
        else:
            smiles_dict[smiles] = 1
    print(len(smiles_list.index))
    print(len(smiles_dict))
    print(abs(len(smiles_dict) - len(smiles_list.index)))
    return ix_duplicated


def remove_duplicated_smiles(path_read, path_write):
    ix_duplicated = find_duplicated_smiles(path_read)
    df = pd.read_csv(path_read).drop(ix_duplicated)
    print(f"Removed duplicated entries, df is now of size: {len(df.index)}")
    df = df.reset_index(drop=True)
    df.to_csv(path_write)
    return df
    

def remove_heavy_atoms(path_read, path_write, remove_dups = True, threshold=250):
    if remove_dups == True:
        df = remove_duplicated_smiles(path_read, path_write)
    else: 
        df = pd.read_csv(path_read)
    ix_heavy = []
    print(df.head())
    for ix in range (len(df.index)):
        smiles = df.loc[ix, "smiles"]
        mol = Chem.MolFromSmiles(smiles)
        mw = Chem.rdMolDescriptors.CalcExactMolWt(mol)
        if mw > threshold:
            ix_heavy.append(ix)

    print(f"Size of dataset after remving duplicates: {len(df.drop(ix_heavy).index)}")
    df.drop(ix_heavy).to_csv(path_write)

if __name__== "__main__":
    path_read = "./data/wang_dataset/wang_data_smiles.csv"
    path_write_no_dups = "./data/wang_dataset/wang_data_smiles_no_dups.csv"
    path_write_no_dups_no_heavy = "./data/wang_dataset/wang_data_smiles_no_dups_no_heavy.csv"
    path_write_no_heavy = "./data/wang_dataset/wang_data_smiles_no_heavy.csv"

    print(find_duplicated_smiles(path_read))
    #remove_duplicated_smiles(path_read, path_write_no_dups)
    remove_heavy_atoms(path_read, path_write_no_dups_no_heavy)
    remove_heavy_atoms(path_read, path_write_no_heavy, remove_dups=False)
    print("duplicates withou heavy atoms")
    find_duplicated_smiles(path_write_no_heavy)
