import deepchem as dc
from rdkit import Chem
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

class Tox21MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Tox21MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.out(x))

def load_model(input_dim, output_dim, model_path="Prediction_property/Toxicity/saved_models/tox21_torch.pth"):
    model = Tox21MLP(input_dim, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_toxicity_for_smiles_pair(smiles1, smiles2):
    model = load_model(input_dim=1024, output_dim=12)
    featurizer = dc.feat.CircularFingerprint(size=1024)
    
    # Define tasks with full names
    tasks_info = {
        "NR-AR": "Nuclear Receptor - Androgen Receptor",
        "NR-AR-LBD": "Nuclear Receptor - Androgen Receptor Ligand Binding Domain", 
        "NR-AhR": "Nuclear Receptor - Aryl Hydrocarbon Receptor",
        "NR-Aromatase": "Nuclear Receptor - Aromatase",
        "NR-ER": "Nuclear Receptor - Estrogen Receptor",
        "NR-ER-LBD": "Nuclear Receptor - Estrogen Receptor Ligand Binding Domain",
        "NR-PPAR-gamma": "Nuclear Receptor - PPAR-gamma",
        "SR-ARE": "Stress Response - Antioxidant Response Element",
        "SR-ATAD5": "Stress Response - ATAD5",
        "SR-HSE": "Stress Response - Heat Shock Response Element",
        "SR-MMP": "Stress Response - Mitochondrial Membrane Potential",
        "SR-p53": "Stress Response - p53"
    }
    
    tasks = list(tasks_info.keys())
    combined_smiles = f"{smiles1}.{smiles2}"  # Treat as mixture or copolymer unit
    mol = Chem.MolFromSmiles(combined_smiles)
    if mol is None:
        return {"error": "Invalid SMILES"}

    features = featurizer.featurize([combined_smiles])
    inputs = torch.FloatTensor(features)
    with torch.no_grad():
        predictions = model(inputs).numpy()[0]

    # Build table string
    table_lines = []
    table_lines.append(f"Toxicity Predictions for: {smiles1} + {smiles2}")
    table_lines.append(f"Combined SMILES: {combined_smiles}")
    table_lines.append("=" * 120)
    table_lines.append(f"{'Abbreviation':<15} {'Full Name':<50} {'Probability':<12} {'Risk Level':<12} {'Prediction'}")
    table_lines.append("-" * 120)
    
    for task, p in zip(tasks, predictions):
        full_name = tasks_info[task]
        if p >= 0.7:
            risk_level = "🔴 HIGH"
            prediction = "Toxic"
        elif p >= 0.5:
            risk_level = "🟡 MEDIUM"
            prediction = "Toxic"
        else:
            risk_level = "🟢 LOW"
            prediction = "Non-Toxic"
        
        table_lines.append(f"{task:<15} {full_name:<50} {p:<12.3f} {risk_level:<12} {prediction}")
    
    # Summary
    avg_score = np.mean(predictions)
    high_risk_count = sum(1 for p in predictions if p >= 0.7)
    table_lines.append("-" * 120)
    table_lines.append(f"Overall Average Score: {avg_score:.3f}")
    table_lines.append(f"High Risk Endpoints: {high_risk_count}/12")
    
    if high_risk_count >= 3:
        table_lines.append("🔴 OVERALL ASSESSMENT: HIGH TOXICITY RISK")
    elif high_risk_count >= 1:
        table_lines.append("🟡 OVERALL ASSESSMENT: MODERATE TOXICITY RISK")
    else:
        table_lines.append("🟢 OVERALL ASSESSMENT: LOW TOXICITY RISK")

    # Return both the table string and the dictionary
    table_string = "\n".join(table_lines)
    
    return {
        "table": table_string,
        "predictions": {
            task: ("Toxic" if p >= 0.5 else "Non-Toxic") + f" ({p:.2f})"
            for task, p in zip(tasks, predictions)
        },
        "raw_scores": {task: float(p) for task, p in zip(tasks, predictions)},
        "summary": {
            "average_score": float(avg_score),
            "high_risk_count": high_risk_count,
            "overall_assessment": "HIGH TOXICITY RISK" if high_risk_count >= 3 else "MODERATE TOXICITY RISK" if high_risk_count >= 1 else "LOW TOXICITY RISK"
        }
    }



if __name__ == "__main__":
   
    result = predict_toxicity_for_smiles_pair("CCOC(=O)C=C", "C1=CC=CC=C1OC(=O)C=C")
    print(result)

