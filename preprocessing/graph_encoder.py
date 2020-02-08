""" Graph Encoder class"""
from preprocessing.smiles_encoder import SMILESEncoder


class GraphEncoder(SMILESEncoder):
    """ Converts SMILES into Graph-like input. Generates adjacency
    matrix, feature vectors, one-hot labels ('y') from SMILES
    
    Parameters
    ----------
    col_name: [str]
        Column name which contains SMILES string
    """
    def __init__(self, col_name: str):
        super().__init__(col_name)