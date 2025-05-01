import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import DataStructs
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
from pathlib import Path
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem import Descriptors

class PropertyNet(nn.Module):
    def __init__(self, input_size=208):
        super(PropertyNet, self).__init__()
        self.model = nn.Sequential(
            # Input layer with dropout
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # First hidden layer
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Second hidden layer
            nn.Linear(256, 128),
            nn.ReLU(), 
            nn.Dropout(0.2),
            
            # Third hidden layer
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Output layer
            nn.Linear(64, 1)
        )
        
        # Apply weight initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
                
    def forward(self, x):
        return self.model(x)

class MolecularDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class PropertyPredictor:
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Get the root directory of the project
        self.root_dir = Path(__file__).parent.parent.parent
        self.model_dir = self.root_dir / "Reward_model" / "PropertyRewards" / "Property_Prediction" / "saved_models"
        
        self.feature_size = 200
        self.radius = 2
        
        # Create models
        self.er_model = PropertyNet().to(device)
        self.tg_model = PropertyNet().to(device)
        
        # Initialize optimizers
        self.er_optimizer = optim.Adam(self.er_model.parameters())
        self.tg_optimizer = optim.Adam(self.tg_model.parameters())
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Initialize scalers
        self.feature_scaler = StandardScaler()
        self.er_scaler = StandardScaler()
        self.tg_scaler = StandardScaler()
        
        if model_path:
            self.load_models(model_path)

    def _get_molecular_features(self, smiles1, smiles2):
        """Generate molecular features from SMILES pairs"""
        try:
            # Convert SMILES to RDKit molecules
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)

            if mol1 is None or mol2 is None:
                raise ValueError("Invalid SMILES string")

            # Generate Morgan fingerprints
            morgan_gen = GetMorganGenerator(radius=self.radius, fpSize=self.feature_size)
            fp1 = np.array(morgan_gen.GetFingerprintAsNumPy(mol1), dtype=np.float32)
            fp2 = np.array(morgan_gen.GetFingerprintAsNumPy(mol2), dtype=np.float32)

            # Combine features
            combined_fp = fp1 + fp2

            # Add additional descriptors
            desc1 = np.array([
                Descriptors.MolWt(mol1),
                Descriptors.MolLogP(mol1),
                Descriptors.TPSA(mol1)
            ], dtype=np.float32)

            desc2 = np.array([
                Descriptors.MolWt(mol2),
                Descriptors.MolLogP(mol2),
                Descriptors.TPSA(mol2)
            ], dtype=np.float32)

            # Combine all features
            features = np.concatenate([combined_fp, desc1, desc2])

            return features.reshape(1, -1)

        except Exception as e:
            print(f"Error in feature generation: {str(e)}")
            return None

    def train(self, train_data, validation_split=0.2, epochs=100, batch_size=32):
        # Generate features for all pairs
        features = []
        for s1, s2, r1, r2 in zip(train_data['smiles1'], train_data['smiles2'], 
                                 train_data['ratio_1'], train_data['ratio_2']):
            feat = self._get_molecular_features(s1, s2)
            if feat is not None:
                feat_with_ratios = np.concatenate([feat.squeeze(), [r1, r2]])
                features.append(feat_with_ratios)
        
        features = np.array(features)
        
        # Scale features and targets
        self.feature_scaler.fit(features)
        features_scaled = self.feature_scaler.transform(features)
        
        er_values = np.array(train_data['er']).reshape(-1, 1)
        tg_values = np.array(train_data['tg']).reshape(-1, 1)
        
        self.er_scaler.fit(er_values)
        self.tg_scaler.fit(tg_values)
        
        er_scaled = self.er_scaler.transform(er_values)
        tg_scaled = self.tg_scaler.transform(tg_values)
        
        # Create datasets
        er_dataset = MolecularDataset(features_scaled, er_scaled)
        tg_dataset = MolecularDataset(features_scaled, tg_scaled)
        
        # Create dataloaders
        er_loader = DataLoader(er_dataset, batch_size=batch_size, shuffle=True)
        tg_loader = DataLoader(tg_dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        print("Training Er model...")
        self._train_model(self.er_model, self.er_optimizer, er_loader, epochs)
        
        print("Training Tg model...")
        self._train_model(self.tg_model, self.tg_optimizer, tg_loader, epochs)

    def _train_model(self, model, optimizer, dataloader, epochs):
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_features, batch_targets in dataloader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = self.criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')

    def predict(self, smiles1, smiles2, ratio_1, ratio_2):
        """Predict Er and Tg values for a SMILES pair"""
        self.er_model.eval()
        self.tg_model.eval()
        
        with torch.no_grad():
            features = self._get_molecular_features(smiles1, smiles2)
            if features is None:
                return None, None
                
            features_with_ratios = np.concatenate([features.squeeze(), [ratio_1, ratio_2]])
            features_with_ratios = features_with_ratios.reshape(1, -1)
            
            # Scale features
            features_scaled = self.feature_scaler.transform(features_with_ratios)
            features_tensor = torch.FloatTensor(features_scaled).to(self.device)
            
            # Make predictions
            er_pred_scaled = self.er_model(features_tensor).cpu().numpy()
            tg_pred_scaled = self.tg_model(features_tensor).cpu().numpy()
            
            # Inverse transform predictions
            er_pred = self.er_scaler.inverse_transform(er_pred_scaled)
            tg_pred = self.tg_scaler.inverse_transform(tg_pred_scaled)
            
            return er_pred[0][0], tg_pred[0][0]

    def save_models(self, save_dir=None):
        """Save models and scalers"""
        if save_dir is None:
            save_dir = self.model_dir
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save neural network models
        torch.save(self.er_model.state_dict(), save_dir / 'er_model.pt')
        torch.save(self.tg_model.state_dict(), save_dir / 'tg_model.pt')
        
        # Save scalers
        joblib.dump(self.feature_scaler, save_dir / 'feature_scaler.pkl')
        joblib.dump(self.er_scaler, save_dir / 'er_scaler.pkl')
        joblib.dump(self.tg_scaler, save_dir / 'tg_scaler.pkl')
        
        print(f"Models and scalers saved to {save_dir}")

    def load_models(self, model_dir=None):
        """Load saved models and scalers"""
        if model_dir is None:
            model_dir = self.model_dir
            
        model_dir = Path(model_dir)
        
        try:
            er_model_path = model_dir / 'er_model.pt'
            tg_model_path = model_dir / 'tg_model.pt'
            
            if not er_model_path.exists() or not tg_model_path.exists():
                print(f"No saved models found in {model_dir}")
                return
            
            # Load neural network models
            self.er_model.load_state_dict(torch.load(er_model_path))
            self.tg_model.load_state_dict(torch.load(tg_model_path))
            
            # Load scalers
            self.feature_scaler = joblib.load(model_dir / 'feature_scaler.pkl')
            self.er_scaler = joblib.load(model_dir / 'er_scaler.pkl')
            self.tg_scaler = joblib.load(model_dir / 'tg_scaler.pkl')
            
            #print(f"Models and scalers loaded successfully from {model_dir}")
            
        except Exception as e:
            print(f"Error loading models from {model_dir}: {str(e)}")
            print("Initializing new models")