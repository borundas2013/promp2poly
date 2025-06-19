import deepchem as dc
from rdkit import Chem
import numpy as np
import os
import pickle

def load_model_and_featurizer():
    with open('Prediction_property/Solubility/saved_models/model_info.pkl', 'rb') as f:
        model_info = pickle.load(f)

    featurizer = model_info['featurizer']
    tasks = model_info['tasks']

    model = dc.models.MultitaskRegressor(
        n_tasks=len(tasks),
        n_features=1024,
        layer_sizes=[512, 256, 128],
        dropouts=[0.3, 0.3, 0.3],
        learning_rate=0.0005,
        weight_decay_penalty=0.01,
        weight_decay_penalty_type='l2'
    )
    model.restore(model_dir='Prediction_property/Solubility/saved_models')

    # print(f"Model loaded from epoch {model_info['best_epoch']}")
    # print(f"Best validation MSE: {model_info['best_valid_mse']:.4f}")
    
    return model, featurizer, tasks

def load_model_and_featurizer_esol():
    with open('Prediction_property/Solubility/deepchem_ESOL/saved_models/model_info.pkl', 'rb') as f:
        model_info = pickle.load(f)

    featurizer = model_info['featurizer']
    tasks = model_info['tasks']

    model = dc.models.MultitaskRegressor(
        n_tasks=len(tasks),
        n_features=1024,
        layer_sizes= [512, 256, 128],  # Smaller layers
        dropouts=[0.3, 0.3, 0.3],     # Higher dropou,
        learning_rate=0.0001,         # Lower learning rate
        weight_decay_penalty=0.01,    # L2 regularization
        weight_decay_penalty_type='l2'
    )
    model.restore(model_dir='Prediction_property/Solubility/deepchem_ESOL/saved_models')

    # print(f"Model loaded from epoch {model_info['best_epoch']}")
    # print(f"Best validation MSE: {model_info['best_valid_mse']:.4f}")
    
    return model, featurizer, tasks

def predict_two_monomers(model, featurizer, tasks, monomer1_smiles: str, monomer2_smiles: str):
    """
    Predict hydration free energy for a pair of monomers
    """
    # Validate SMILES
    mol1 = Chem.MolFromSmiles(monomer1_smiles)
    mol2 = Chem.MolFromSmiles(monomer2_smiles)
    
    if mol1 is None:
        return {"error": f"Invalid SMILES for monomer 1: {monomer1_smiles}"}
    if mol2 is None:
        return {"error": f"Invalid SMILES for monomer 2: {monomer2_smiles}"}
    
    # Create combined SMILES (dot notation for separate molecules)
    combined_smiles = f"{monomer1_smiles}.{monomer2_smiles}"
    
    # Create features for the combined molecule
    features = featurizer.featurize([mol1, mol2])  # Featurize both monomers separately
    dataset = dc.data.NumpyDataset(X=features)
    
    # Make prediction
    prediction = model.predict(dataset)
    
    # Reshape if needed
    if prediction.ndim == 3:
        prediction = prediction.reshape(prediction.shape[0], -1)
    
    return {
        "monomer1_smiles": monomer1_smiles,
        "monomer2_smiles": monomer2_smiles,
        "combined_smiles": combined_smiles,
        "monomer1_hydration_free_energy": float(prediction[0][0]),  # kcal/mol
        "monomer2_hydration_free_energy": float(prediction[1][0]),  # kcal/mol
        "average_hydration_free_energy": float(np.mean(prediction[:, 0])),  # kcal/mol
        "property": "Hydration Free Energy",
        "units": "kcal/mol"
    }

def predict_two_monomers_esol(model, featurizer, tasks, monomer1_smiles: str, monomer2_smiles: str):
    """
    Predict solubility for a pair of monomers
    """
    mol1 = Chem.MolFromSmiles(monomer1_smiles)
    mol2 = Chem.MolFromSmiles(monomer2_smiles)

    if mol1 is None:
        return {"error": f"Invalid SMILES for monomer 1: {monomer1_smiles}"}
    if mol2 is None:
        return {"error": f"Invalid SMILES for monomer 2: {monomer2_smiles}"}
    
    # Create combined SMILES (dot notation for separate molecules)
    combined_smiles = f"{monomer1_smiles}.{monomer2_smiles}"

    # Create features for the combined molecule
    features = featurizer.featurize([mol1, mol2])  # Featurize both monomers separately
    dataset = dc.data.NumpyDataset(X=features)
    
    # Make prediction
    prediction = model.predict(dataset)
    average_logS = float(np.mean(prediction[:, 0]))

    if average_logS >= 0:
        solubility = "Very Soluble"
    elif average_logS >= -1:
        solubility = "Soluble"
    elif average_logS >= -2:
        solubility = "Moderately Soluble"
    elif average_logS >= -3:
        solubility = "Slightly Soluble"
    else:
        solubility = "Insoluble"


    return {
        "monomer1_smiles": monomer1_smiles,
        "monomer2_smiles": monomer2_smiles,
        "combined_smiles": combined_smiles,
        "monomer1_logS": float(prediction[0][0]),  
        "monomer2_logS": float(prediction[1][0]),  
        "average_logS": float(np.mean(prediction[:, 0])),  
        "property": "Solubility",
        "units": "mol/L",
        "solubility": solubility
    }

# def predict_batch_monomers(model, featurizer, tasks, monomer_pairs):
#     """
#     Predict hydration free energy for a batch of monomer pairs
#     """
#     results = []
    
#     for i, (monomer1, monomer2) in enumerate(monomer_pairs):
#         result = predict_two_monomers(model, featurizer, tasks, monomer1, monomer2)
#         result['sample_id'] = i + 1
#         results.append(result)
        
#         print(f"Sample {i+1}:")
#         print(f"  Monomer 1: {monomer1}")
#         print(f"  Monomer 2: {monomer2}")
#         print(f"  Monomer 1 HFE: {result['monomer1_hydration_free_energy']:.4f} kcal/mol")
#         print(f"  Monomer 2 HFE: {result['monomer2_hydration_free_energy']:.4f} kcal/mol")
#         print(f"  Average HFE: {result['average_hydration_free_energy']:.4f} kcal/mol")
#         print()
    
#     return results


def predict_solubility(smiles1, smiles2):
  model, featurizer, tasks = load_model_and_featurizer()
  model_esol, featurizer, tasks = load_model_and_featurizer_esol()
  result = predict_two_monomers(model, featurizer, tasks, smiles1, smiles2)
  result_esol = predict_two_monomers_esol(model_esol, featurizer, tasks, smiles1, smiles2)
  return result, result_esol


if __name__ == "__main__":
    result,result_esol =predict_solubility("CC(C)OC(=O)C=C", "C1=CC=CC=C1OC(=O)C=C")
    print(result['average_hydration_free_energy'])
    print(result_esol['average_logS'], result_esol['solubility'])
    
    