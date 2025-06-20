import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem
from rdkit.Chem import DataStructs
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
from pathlib import Path
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem import Descriptors

class RewardNet(nn.Module):
    def __init__(self):
        super(RewardNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(216, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.ReLU(), 
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 1)
        )
        
        # Add L2 regularization
        self.l2_lambda = 0.01
        
    def forward(self, x):
        return self.model(x)
        
    def get_l2_loss(self):
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)
        return self.l2_lambda * l2_loss

class GroupRewardScorePredictor:
    def __init__(self, model_path=None):
        # Get the root directory of the project
        self.root_dir = Path(__file__).parent.parent.parent  # Goes up three levels to COT_TSMP
        self.model_dir = self.root_dir / "Two_Monomers_Group" / "RLHF" / "saved_models"
        
        self.feature_size = 200
        self.radius = 2
        
        # Create model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.score_model = RewardNet().to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.score_model.parameters())
        self.criterion = nn.MSELoss()
        
        # Initialize scaler
        self.feature_scaler = StandardScaler()
        
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
            combined_fp = fp1 + fp2  # Simple addition of fingerprints

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
        
    def get_group_onehot_encoding(self,group_label):
        # Define all possible functional groups
        GROUP_MAPPING = {
            'C=C': [1, 0, 0, 0, 0],      # Vinyl
            'CCS': [0, 1, 0, 0, 0],      # Thiol
            'C=C(C=O)': [0, 0, 1, 0, 0], # Acryl
            'C1OC1': [0, 0, 0, 1, 0],    # Epoxy
            'NC': [0, 0, 0, 0, 1],       # Imine
            'No_group': [0, 0, 0, 0, 0]  # No functional group
        }
        return GROUP_MAPPING.get(group_label, [0, 0, 0, 0, 0])

    def train(self, train_data, validation_split=0.2, epochs=100):
        """Train the property prediction models"""
        # Generate features for all pairs
        features = []
        scores = []
        for s1, s2, g1, g2, score in zip(train_data['smiles1'], train_data['smiles2'], 
                                       train_data['group1'], train_data['group2'],
                                       train_data['score']):
            feat = self._get_molecular_features(s1, s2)
            g1_onehot = self.get_group_onehot_encoding(g1)
            g2_onehot = self.get_group_onehot_encoding(g2)
            if feat is not None:
                feat_with_groups = np.concatenate([feat.squeeze(), g1_onehot, g2_onehot])
                features.append(feat_with_groups)
                scores.append(score)
                
        features = np.array(features)
        scores = np.array(scores)
        
        # Scale features
        self.feature_scaler.fit(features)
        features_scaled = self.feature_scaler.transform(features)
        
        # Convert to PyTorch tensors
        features_tensor = torch.FloatTensor(features_scaled).to(self.device)
        scores_tensor = torch.FloatTensor(scores).to(self.device)
        
        # Split into train/validation
        n_samples = len(features)
        n_val = int(n_samples * validation_split)
        indices = torch.randperm(n_samples)
        
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]
        
        # Training loop
        print("Training Reward model...")
        for epoch in range(epochs):
            # Training
            self.score_model.train()
            self.optimizer.zero_grad()
            
            outputs = self.score_model(features_tensor[train_idx])
            loss = self.criterion(outputs.squeeze(), scores_tensor[train_idx])
            loss += self.score_model.get_l2_loss()
            
            loss.backward()
            self.optimizer.step()
            
            # Validation
            self.score_model.eval()
            with torch.no_grad():
                val_outputs = self.score_model(features_tensor[val_idx])
                val_loss = self.criterion(val_outputs.squeeze(), scores_tensor[val_idx])
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

    def predict(self, smiles1, smiles2, group1, group2):
        """Predict score for a SMILES pair"""
        self.score_model.eval()
        features = self._get_molecular_features(smiles1, smiles2)
        if features is None:
            return None
            
        group1_onehot = self.get_group_onehot_encoding(group1)  
        group2_onehot = self.get_group_onehot_encoding(group2)
        features_with_groups = np.concatenate([features.squeeze(), group1_onehot, group2_onehot])
        features_with_groups = features_with_groups.reshape(1, -1)
        features_scaled = self.feature_scaler.transform(features_with_groups)
        
        # Convert to tensor and predict
        features_tensor = torch.FloatTensor(features_scaled).to(self.device)
        with torch.no_grad():
            score_pred = self.score_model(features_tensor)
            
        return score_pred.item()

    def save_models(self, save_dir=None):
        """Save models and scalers"""
        if save_dir is None:
            save_dir = self.model_dir
            
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save PyTorch model
        torch.save({
            'model_state_dict': self.score_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, save_dir / 'score_model.pt')
        
        # Save scaler
        joblib.dump(self.feature_scaler, save_dir / 'feature_scaler.pkl')
        
        print(f"Models and scalers saved to {save_dir}")

    def load_models(self, model_dir=None):
        """Load saved models and scalers"""
        if model_dir is None:
            model_dir = self.model_dir
            
        model_dir = Path(model_dir)
        
        try:
            # Check if model files exist
            score_model_path = model_dir / 'score_model.pt'
            
            if not score_model_path.exists():
                print(f"No saved models found in {model_dir}")
                return
            
            # Load PyTorch model
            checkpoint = torch.load(score_model_path)
            self.score_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scaler
            self.feature_scaler = joblib.load(model_dir / 'feature_scaler.pkl')
            
            print(f"Models and scalers loaded successfully from {model_dir}")
            
        except Exception as e:
            print(f"Error loading models from {model_dir}: {str(e)}")
            print("Initializing new models")