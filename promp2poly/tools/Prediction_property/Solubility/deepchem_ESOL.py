import deepchem as dc
from rdkit import Chem
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import pickle

def train_and_evaluate_model():
    """
    Train, validate, and test a model on the FreeSolv dataset with early stopping
    """
    print("Loading FreeSolv dataset...")

    # Load the dataset
    #featurizer = dc.feat.CircularFingerprint(size=1024,radius=2)
    featurizer = dc.feat.CircularFingerprint(size=1024,radius=2)
    tasks, datasets, transformers = dc.molnet.load_delaney(featurizer=featurizer, split='random')
    
    print(f"Number of tasks: {len(tasks)}")
    print(f"Task names: {tasks}")
    print(f"Training set size: {len(datasets[0])} molecules")
    print(f"Validation set size: {len(datasets[1])} molecules")
    print(f"Test set size: {len(datasets[2])} molecules")
    
    # Initialize model with more regularization
#     model = dc.models.GraphConvModel(
#     len(tasks),
#     mode='regression',
#     dropout=0.2,
#     learning_rate=0.0005,
#     batch_size=64
# )
    model = dc.models.MultitaskRegressor(
        n_tasks=len(tasks),
        n_features=1024,
        layer_sizes= [512, 256, 128],  # Smaller layers
        dropouts=[0.3, 0.3, 0.3],     # Higher dropou,
        learning_rate=0.0001,         # Lower learning rate
        weight_decay_penalty=0.01,    # L2 regularization
        weight_decay_penalty_type='l2'
    )
    
    # Training parameters with early stopping
    max_epochs = 100
    patience = 15  # Stop if no improvement for 15 epochs
    best_valid_mse = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    train_scores = []
    valid_scores = []
    
    print(f"\nTraining model with early stopping (max {max_epochs} epochs, patience {patience})...")
    print("=" * 60)
    
    # Training loop with early stopping
    for epoch in range(max_epochs):
        # Train for one epoch
        model.fit(datasets[0], nb_epoch=1)
        
        # Calculate training metrics
        train_pred = model.predict(datasets[0])
        # Reshape predictions to 2D if needed
        if train_pred.ndim == 3:
            train_pred = train_pred.reshape(train_pred.shape[0], -1)
        train_mse = mean_squared_error(datasets[0].y, train_pred)
        train_mae = mean_absolute_error(datasets[0].y, train_pred)
        train_r2 = r2_score(datasets[0].y, train_pred)
        train_scores.append({
            'mse': train_mse,
            'mae': train_mae,
            'r2': train_r2
        })
        
        # Calculate validation metrics
        valid_pred = model.predict(datasets[1])
        # Reshape predictions to 2D if needed
        if valid_pred.ndim == 3:
            valid_pred = valid_pred.reshape(valid_pred.shape[0], -1)
        valid_mse = mean_squared_error(datasets[1].y, valid_pred)
        valid_mae = mean_absolute_error(datasets[1].y, valid_pred)
        valid_r2 = r2_score(datasets[1].y, valid_pred)
        valid_scores.append({
            'mse': valid_mse,
            'mae': valid_mae,
            'r2': valid_r2
        })
        
        # Early stopping check
        if valid_mse < best_valid_mse:
            best_valid_mse = valid_mse
            patience_counter = 0
            best_epoch = epoch + 1
            # Save the best model immediately
            save_model_and_featurizer(model, featurizer, tasks, best_epoch, best_valid_mse)
        else:
            patience_counter += 1
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{max_epochs}:")
            print(f"  Training - MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
            print(f"  Validation - MSE: {valid_mse:.4f}, MAE: {valid_mae:.4f}, R²: {valid_r2:.4f}")
            print(f"  Best validation MSE: {best_valid_mse:.4f} (epoch {best_epoch})")
            print(f"  Patience counter: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
            print(f"Best validation MSE: {best_valid_mse:.4f} at epoch {best_epoch}")
            break
    
    # Load the best model for final evaluation
    print(f"Loading best model from epoch {best_epoch} for final evaluation...")
    model, featurizer, tasks = load_model_and_featurizer()
    
    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("FINAL TEST SET EVALUATION")
    print("=" * 60)
    
    test_pred = model.predict(datasets[2])
    # Reshape predictions to 2D if needed
    if test_pred.ndim == 3:
        test_pred = test_pred.reshape(test_pred.shape[0], -1)
    test_mse = mean_squared_error(datasets[2].y, test_pred)
    test_mae = mean_absolute_error(datasets[2].y, test_pred)
    test_r2 = r2_score(datasets[2].y, test_pred)
    
    print(f"Test Set Results:")
    print(f"  Mean Squared Error (MSE): {test_mse:.4f}")
    print(f"  Mean Absolute Error (MAE): {test_mae:.4f}")
    print(f"  R-squared (R²): {test_r2:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {np.sqrt(test_mse):.4f}")
    
    print(f"\nTraining Summary:")
    print(f"  Total epochs trained: {len(train_scores)}")
    print(f"  Best validation performance at epoch {best_epoch}:")
    print(f"    Validation MSE: {best_valid_mse:.4f}")
    print(f"    Validation MAE: {valid_scores[best_epoch-1]['mae']:.4f}")
    print(f"    Validation R²: {valid_scores[best_epoch-1]['r2']:.4f}")
    
    # Plot training curves
    
    return model, featurizer, tasks

def save_model_and_featurizer(model, featurizer, tasks, best_epoch, best_valid_mse):
    os.makedirs('Prediction_property/Solubility/deepchem_ESOL/saved_models/saved_models', exist_ok=True)

    # Save model weights
    model.save_checkpoint(model_dir='Prediction_property/Solubility/deepchem_ESOL/saved_models')

    # Save featurizer and metadata
    model_info = {
        'featurizer': featurizer,
        'tasks': tasks,
        'best_epoch': best_epoch,
        'best_valid_mse': best_valid_mse,
        'model_type': 'MultitaskRegressor'
    }

    with open('Prediction_property/Solubility/deepchem_ESOL/saved_models/model_info.pkl', 'wb') as f:
        pickle.dump(model_info, f)
def load_model_and_featurizer():
    with open('Prediction_property/Solubility/deepchem_ESOL/saved_models/model_info.pkl', 'rb') as f:
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
    model.restore(model_dir='Prediction_property/Solubility/deepchem_ESOL/saved_models/')

    print(f"Model loaded from epoch {model_info['best_epoch']}")
    print(f"Best validation MSE: {model_info['best_valid_mse']:.4f}")
    
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
        "monomer1_logS": float(prediction[0][0]),  # kcal/mol
        "monomer2_logS": float(prediction[1][0]),  # kcal/mol
        "average_logS": float(np.mean(prediction[:, 0])),  # kcal/mol
        "property": "Solubility",
        "units": "kcal/mol"
    }

def predict_batch_monomers(model, featurizer, tasks, monomer_pairs):
    """
    Predict hydration free energy for a batch of monomer pairs
    """
    results = []
    
    for i, (monomer1, monomer2) in enumerate(monomer_pairs):
        result = predict_two_monomers(model, featurizer, tasks, monomer1, monomer2)
        result['sample_id'] = i + 1
        results.append(result)
        
        print(f"Sample {i+1}:")
        print(f"  Monomer 1: {monomer1}")
        print(f"  Monomer 2: {monomer2}")
        print(f"  Monomer 1 logS: {result['monomer1_logS']:.4f} mol/L")
        print(f"  Monomer 2 logS: {result['monomer2_logS']:.4f} mol/L")
        print(f"  Average logS: {result['average_logS']:.4f} mol/L")
        print()
    
    return results


def predict_property_combined(model, featurizer, tasks, smiles_combined: str):
    """
    Predict hydration free energy for a combined SMILES string
    """
    mol = Chem.MolFromSmiles(smiles_combined)
    if mol is None:
        return {"error": "Invalid combined SMILES"}
    
    # Create features and dataset
    features = featurizer.featurize([mol])
    dataset = dc.data.NumpyDataset(X=features)
    
    # Make prediction
    prediction = model.predict(dataset)
    
    return {
        "combined_smiles": smiles_combined,
        "logS": float(prediction[0][0]),  # logS
        "property": "Solubility",
        "units": "mol/L"
    }

if __name__ == "__main__":
    # Check if model already exists
    if os.path.exists('Prediction_property/Solubility/deepchem_ESOL/saved_models/model_info.pkl'):
        print("Loading existing model...")
        model, featurizer, tasks = load_model_and_featurizer()
    else:
        print("Training new model...")
        model, featurizer, tasks = train_and_evaluate_model()
    
    # Test prediction on monomer pairs
    print("\n" + "=" * 60)
    print("PREDICTIONS FOR MONOMER PAIRS")
    print("=" * 60)
    
    # Example monomer pairs
    monomer_pairs = [
        ("CC(C)OC(=O)C=C", "C1=CC=CC=C1OC(=O)C=C"),  # Methyl methacrylate + Phenyl methacrylate
        ("CCOC(=O)C=C", "CC(C)OC(=O)C=C"),           # Ethyl methacrylate + Methyl methacrylate
        ("C1=CC=CC=C1OC(=O)C=C", "CCOC(=O)C=C"),     # Phenyl methacrylate + Ethyl methacrylate
    ]
    
    # Predict for all monomer pairs
    results = predict_batch_monomers(model, featurizer, tasks, monomer_pairs)
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for result in results:
        print(f"Sample {result['sample_id']}: {result['monomer1_smiles']} + {result['monomer2_smiles']}")
        print(f"  Average logS: {result['average_logS']:.4f} mol/L")
        print()
