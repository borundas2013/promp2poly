import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from property_prediction_model import PropertyPredictor
from pathlib import Path
from rdkit import Chem
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score




def load_predictor():
    """Load the trained predictor model"""
    root_dir = Path(__file__).parent.parent
    model_dir = root_dir / 'Property_Prediction/saved_models22'
    return PropertyPredictor(model_path=str(model_dir))



def predict_property(smiles1, smiles2):
    predictor = load_predictor()
    er, tg = predictor.predict(smiles1, smiles2, 0.1, 0.9)
    return er,tg


if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent
    file_path = root_dir / "DeepSeek" / "Output" / "Combined_deepseek.csv"
    #file_path = root_dir / "Llama32" / "Output" / "Combined_Llama.csv"
    #file_path = root_dir / "GPT4" / "Output" / "Combined_gpt.csv"
    df = pd.read_csv(str(file_path))
    tg_window = 30
    er_window = 30
  

    # Initialize columns (optional but neat)
    for col in ["Predicted Tg","Predicted Er","Difference Tg","Difference Er"]:
        if col not in df.columns:
            df[col] = np.nan

    abs_errs_tg = []
    abs_errs_er = []
    sq_errs_tg  = []
    sq_errs_er  = []

    actual_tg_arr = []
    actual_er_arr = []
    predicted_tg_arr = []
    predicted_er_arr = []

    # For "outside window" stats
    outside_abs_tg = []
    outside_abs_er = []
 

    n_eval = 0

    for index, row in df.iterrows():
        smiles1 = row['Fixed Monomer 1']
        smiles2 = row['Fixed Monomer 2']
        

        m1 = Chem.MolFromSmiles(smiles1) if pd.notna(smiles1) else None
        m2 = Chem.MolFromSmiles(smiles2) if pd.notna(smiles2) else None
        if m1 is None or m2 is None:
            continue

        actual_tg = row['Tg']
        actual_er = row['Er']
        if pd.isna(actual_tg) or pd.isna(actual_er):
            continue

        try:
            er_pred, tg_pred = predict_property(smiles1, smiles2)  # returns (er, tg)
        except Exception:
            # If your predictor can fail, skip this sample gracefully
            continue

        df.at[index, 'Predicted Tg'] = tg_pred
        df.at[index, 'Predicted Er'] = er_pred

        diff_tg = abs(actual_tg - tg_pred)
        diff_er = abs(actual_er - er_pred)
        if diff_tg > tg_window:
            diff_tg = diff_tg
        else:
            diff_tg = 0
        if diff_er > er_window:
            diff_er = diff_er
        else:
            diff_er = 0
       

        print(f"Actual Tg: {actual_tg}, Predicted Tg: {tg_pred}, Difference Tg: {diff_tg}")
        print(f"Actual Er: {actual_er}, Predicted Er: {er_pred}, Difference Er: {diff_er}")
        df.at[index, 'Difference Tg'] = diff_tg
        df.at[index, 'Difference Er'] = diff_er
        

        abs_errs_tg.append(diff_tg)
        abs_errs_er.append(diff_er)
      
        actual_tg_arr.append(actual_tg)
        actual_er_arr.append(actual_er)
        predicted_tg_arr.append(tg_pred)
        predicted_er_arr.append(er_pred)

       
        n_eval += 1
        

        

    # --- Metrics ---
    if n_eval == 0:
        print("No valid rows evaluated.")
    else:
        mae_tg = float(np.mean(abs_errs_tg))
        mae_er = float(np.mean(abs_errs_er))


        print(f"Samples evaluated: {n_eval}")
        print(f"MAE Tg:  {mae_tg:.4f}")
        print(f"MAE Er:  {mae_er:.4f}")
      



