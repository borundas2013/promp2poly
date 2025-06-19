import deepchem as dc
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np
from rdkit import Chem

# Step 1: Load & featurize Tox21 dataset
def load_data():
    tasks, datasets, transformers = dc.molnet.load_tox21(featurizer='ECFP', splitter='random')
    train, valid, test = datasets
    return tasks, train, valid, test

# Step 2: Define simple PyTorch MLP for multitask classification
class Tox21MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Tox21MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.out(x))  # for multi-label classification

# Step 3: Torch training utility
def train_model(model, train, valid, tasks, epochs=10, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    best_score = -1.0  # Best mean ROC-AUC
    model_path = "Prediction_property/Toxicity/saved_models/tox21_torch.pth"

    for epoch in range(epochs):
        X = train.X
        y = train.y
        mask = ~np.isnan(y)

        model.train()
        optimizer.zero_grad()
        inputs = torch.FloatTensor(X)
        targets = torch.FloatTensor(np.nan_to_num(y))
        outputs = model(inputs)

        mask_tensor = torch.FloatTensor(mask.astype(float))
        loss = loss_fn(outputs * mask_tensor, targets * mask_tensor)

        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}: loss = {loss.item():.4f}")

        # Evaluate
        val_scores = evaluate_model(model, valid, tasks)
        mean_score = np.nanmean(list(val_scores.values()))
        print(f"Validation Mean ROC-AUC: {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            torch.save(model.state_dict(), model_path)
            print(f"✅ New best model saved (ROC-AUC = {mean_score:.4f})")


def load_model(input_dim, output_dim, model_path="Prediction_property/Toxicity/saved_models/tox21_torch.pth"):
    model = Tox21MLP(input_dim, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model
# Step 4: Evaluate model using ROC-AUC
def evaluate_model(model, dataset, tasks):
    model.eval()
    X = dataset.X
    y = dataset.y
    inputs = torch.FloatTensor(X)
    
    with torch.no_grad():
        outputs = model(inputs).numpy()

    results = {}
    for i, task in enumerate(tasks):
        y_true = y[:, i]
        y_pred = outputs[:, i]
        mask = ~np.isnan(y_true)
        if np.sum(mask) == 0:
            results[task] = np.nan
        else:
            results[task] = roc_auc_score(y_true[mask], y_pred[mask])
    return results

def predict_toxicity_for_smiles_pair(smiles1, smiles2, model, featurizer, tasks):
    combined_smiles = f"{smiles1}.{smiles2}"  # Treat as mixture or copolymer unit
    mol = Chem.MolFromSmiles(combined_smiles)
    if mol is None:
        return {"error": "Invalid SMILES"}

    features = featurizer.featurize([combined_smiles])
    inputs = torch.FloatTensor(features)
    with torch.no_grad():
        predictions = model(inputs).numpy()[0]

    return {
        task: ("Toxic" if p >= 0.5 else "Non-Toxic") + f" ({p:.2f})"
        for task, p in zip(tasks, predictions)
    }

if __name__ == "__main__":
    tasks, train, valid, test = load_data()
    model=Tox21MLP(input_dim=train.X.shape[1], output_dim=len(tasks))
    train_model(model, train,valid, tasks, epochs=100)

    
    model = load_model(input_dim=train.X.shape[1], output_dim=len(tasks))
    
    

    # Use same featurizer as training
    featurizer = dc.feat.CircularFingerprint(size=1024)

    # Predict for SMILES pair
    result = predict_toxicity_for_smiles_pair(
        "CCOC(=O)C=C", "C1=CC=CC=C1OC(=O)C=C", model, featurizer, tasks
    )

    print("\nToxicity Prediction for Combined SMILES:")
    for k, v in result.items():
        print(f"{k}: {v}")
