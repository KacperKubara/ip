import pytest 
# Fix import problem with pytest
# And run: python -m pytest ***command**
import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from preprocessing import SMILESEncoder
from data import load_sol_challenge

def test_ecfp():
    df = load_sol_challenge()
    enc = SMILESEncoder(col_name="SMILES")
    enc.to_ecfp(df)