import os
import sys

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Prediction_property.TgEr.property_prediction_model import PropertyPredictor
from pathlib import Path
from rdkit import Chem


def load_predictor():
    """Load the trained predictor model"""
    root_dir = Path(__file__).parent.parent
    #print(root_dir)
    model_dir = root_dir / 'TgEr' / 'saved_models22'
    #print(model_dir)
    return PropertyPredictor(model_path=str(model_dir))

def predict_properties(smiles1: str, smiles2: str, ratio_1: float, ratio_2: float) -> tuple:
    """Predict Er and Tg for a given SMILES pair"""
    predictor = load_predictor()
    return predictor.predict(smiles1, smiles2, ratio_1, ratio_2)

def predict_property(smiles1, smiles2, ratio_1, ratio_2):
    er, tg = predict_properties(smiles1, smiles2, ratio_1, ratio_2)
    return {
        "tg_score": tg,
        "er_score": er
    }
    
    
    
      
    



if __name__ == "__main__":
  
    smiles1 = 'CCCCCCCCC1OC1CCCCCCCC(=O)OCC(COC(=O)CCCCCCCC2OC2CC3OC3CC4OC4CC)OC(=O)CCCCCCCC5OC5CC6OC6CC7OC7CC'
    smiles2 = 'CCCCNCCCNCCCNCCCNCCCN(CCN)CCN'
    mis_result = predict_property(smiles1, smiles2)
    print(f"Predictions - {mis_result}")

    smiles1 = 'C=CCn1c(=O)n(CC=C)c(=O)n(CC=C)c1=O'
    smiles2 = 'CCC(COC(=O)CCSC)(COC(=O)CCSC)COC(=O)CCSC'
    deepseek_result = predict_property(smiles1, smiles2)
    print(f"Predictions - {deepseek_result}")

    smiles1 = 'C=C(C)C(=O)OCCOCCOCCOCCOCCOCCOCCOCCOC(=O)C(C)=O'
    smiles2 = 'C=C(C)C(=O)OCCOCCOCCOCCOCCOC(=O)C(C)=O'
    llama_result = predict_property(smiles1, smiles2)
    print(f"Predictions - {llama_result}")
