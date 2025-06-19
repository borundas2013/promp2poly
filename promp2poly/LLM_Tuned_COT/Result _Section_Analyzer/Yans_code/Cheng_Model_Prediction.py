# -*- coding: utf-8 -*-
"""
Created on Wed May 12 13:04:53 2021

@author: Cheng
"""
from itertools import combinations
import numpy as np
import itertools
import string
import csv
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPool1D, Dropout, BatchNormalization, Dense, TFSMLayer, Input
import os
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
import pandas as pd
from typing import List, Tuple, Dict, Any
import tensorflow as tf

class ChengModelPredictor:
    def __init__(self, model_paths: Dict[str, str], df: pd.DataFrame):
        """
        Initialize the Cheng Model Predictor
        
        Args:
            model_paths (Dict[str, str]): Dictionary containing paths to model files
            smiles_file (str): Path to the SMILES input file
        """
        self.charset = ['-', 'F', 'S', '9', 'N', '(', 'l', 'P', 'L', 'T', 'p', 'r', 'A', 'K', 't', ']', '1', 'X', 'R', 
                       'o', '!', 'c', '#', 'C', '+', 'B', 's', 'a', 'H', '8', 'n', '6', '4', '[', '3', ')', '0', '%', 
                       'i', '.', '=', 'g', 'O', 'Z', 'E', '/', '@', 'e', '\\', 'I', 'b', '7', '2', 'M', '5']
        self.char_to_int = dict((c, i) for i, c in enumerate(self.charset))
        self.int_to_char = dict((i, c) for i, c in enumerate(self.charset))
        self.latent_dim = 256
        self.embed = 205
        
        # Load models using TFSMLayer for Keras 3 compatibility
        self.smiles_to_latent_model = TFSMLayer(model_paths['smiles_to_latent'], call_endpoint='serving_default')
        self.latent_to_states_model = TFSMLayer(model_paths['latent_to_states'], call_endpoint='serving_default')
        self.sample_model = TFSMLayer(model_paths['sample'], call_endpoint='serving_default')
        
        # Initialize neutral model
        self.neutral_model = self._create_neutral_model()
        
        # Load SMILES data
       # self.smiles_data = self._load_smiles_data(smiles_file)
        self.smiles_1, self.smiles_2 = self._process_smiles_data(df)

    def _create_neutral_model(self) -> Sequential:
        """Create and compile the neutral model"""
        # Create model with proper input layer
        Neutral_model = Sequential()
        
        Neutral_model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(256,1)))
        Neutral_model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        
        # Neutral_model.add(MaxPooling1D(pool_size=2))
        Neutral_model.add(GlobalMaxPool1D())
        
        # Neutral_model.add(Flatten())
        Neutral_model.add(BatchNormalization())
        Dropout(0.4)
        Neutral_model.add(Dense(256, activation='relu'))
        
        Neutral_model.add(Dense(64, activation="relu"))
        # Dropout(0.4)
        Neutral_model.add(Dense(64, activation="relu"))
        Neutral_model.add(Dense(64, activation="relu"))
        Neutral_model.add(Dense(32, activation="relu"))
        Neutral_model.add(Dense(32, activation="relu"))
        # check to see if the regression node should be added
        Neutral_model.add(Dense(1, activation="linear"))

        
        model = Neutral_model #Model(inputs=inputs, outputs=outputs)
        
        def root_mean_squared_error(y_true, y_pred):
            return K.mean(K.abs(y_pred - y_true)/K.abs(y_true))
            
        model.compile(loss="mae", optimizer='adam', metrics=[root_mean_squared_error])
        return model

    def _load_smiles_data(self, smiles_file: str) -> pd.DataFrame:
        """Load and preprocess SMILES data from file"""
        df = pd.read_excel(smiles_file)
        smiles_total = df['SMILES']
        
        # Clean SMILES data
        for i in range(len(smiles_total)):
            smiles_total[i] = smiles_total[i].translate({ord(c): None for c in string.whitespace})
            smiles_total[i] = smiles_total[i].split(',')
            smiles_total[i] = [item.replace("{", "").replace("}", "") for item in smiles_total[i]]
            
        return smiles_total

    def _process_smiles_data(self, df: pd.DataFrame) -> Tuple[List, List]:
        # """Process SMILES data into separate lists"""
        # smiles_1 = [[] for _ in range(len(self.smiles_data))]
        # smiles_2 = [[] for _ in range(len(self.smiles_data))]
        
        # for i in range(len(self.smiles_data)):
        #     smiles_1[i] = self.smiles_data[i][0]
        #     if len(self.smiles_data[i]) > 1:
        #         smiles_2[i] = self.smiles_data[i][1]
                
        # return smiles_1, smiles_2
        return df['SMILE1'], df['SMILE2']

    def vector_to_smiles(self, X: np.ndarray) -> np.ndarray:
        """Convert vector to SMILES representation"""
        X = X.reshape(1, X.shape[0], X.shape[1], 1)
        
        # Convert input to tensor and ensure it's float32
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        
        # Call TFSMLayer with the input tensor
        x_latent = self.smiles_to_latent_model(X)
        if isinstance(x_latent, dict):
            x_latent = list(x_latent.values())[0]
            
        states = self.latent_to_states_model(x_latent)
        if isinstance(states, dict):
            states = list(states.values())[0]
        
        startidx = self.char_to_int["!"]
        samplevec = np.zeros((1, 1, len(self.charset)), dtype=np.float32)
        samplevec[0, 0, startidx] = 1.0
        
        smiles = ""
        for _ in range(205):
            o = self.sample_model(samplevec)
            if isinstance(o, dict):
                o = list(o.values())[0]
            sampleidx = np.argmax(o)
            samplechar = self.int_to_char[sampleidx]
            if samplechar != "E":
                smiles = smiles + self.int_to_char[sampleidx]
                samplevec = np.zeros((1, 1, len(self.charset)), dtype=np.float32)
                samplevec[0, 0, sampleidx] = 1.0
            else:
                break
        return x_latent
    
    def split1(self,word):
        return [char for char in word]

    def vectorize1(self, smiles: str) -> np.ndarray:
        """Convert SMILES string to vector representation"""
        smiles = list(smiles)
        # Create array with size embed (not embed-1) to accommodate all characters
        one_hot = np.zeros((self.embed, len(self.charset)), dtype=np.float32)
        one_hot[0, self.char_to_int["!"]] = 1.0
        
        for j, c in enumerate(smiles):
            if j < self.embed - 1:  # Ensure we don't exceed array bounds
                one_hot[j+1, self.char_to_int[c]] = 1.0
            
        # Fill remaining positions with E
        one_hot[len(smiles)+1:, self.char_to_int["E"]] = 1.0
        return one_hot

    def predict_properties(self, model_weights: Dict[str, str], output_file: str) -> None:
        """
        Predict properties for different molar ratios and save results
        
        Args:
            model_weights (Dict[str, str]): Dictionary containing paths to model weights
            output_file (str): Path to save prediction results
        """
        molar_ratio = 0.0
        combined_vetor_all=np.zeros((int(len(self.smiles_1)),1,self.latent_dim))
        
        for n in range(1):
            molar_ratio += 0.1
            print(f"Processing molar ratio: {molar_ratio}")
            
            for i in range(0, int(len(self.smiles_1))):
                vec1 = self.vectorize1(self.smiles_1[i])
                latent_v1 = self.vector_to_smiles(vec1)
                #combined_vetor = latent_v1 * molar_ratio
                combined_vetor = latent_v1
                if len(self.smiles_1[i]) > 1:
                    vec2 = self.vectorize1(self.smiles_2[i])
                    latent_v2 = self.vector_to_smiles(vec2)
                    #combined_vetor = latent_v1 * molar_ratio + latent_v2 * (1-molar_ratio)
                    combined_vetor = latent_v2 + latent_v1
                combined_vetor_all[i] = combined_vetor
            
            # Predict Tg
            self.neutral_model.load_weights(model_weights['Tg'])
            pred_Tg = np.ones(len(self.smiles_1), dtype=np.float32)
            for k in range(len(self.smiles_1)):
                to_predict = combined_vetor_all[k].reshape(1, 256, 1)
                pred_Tg[k] = self.neutral_model(to_predict)
            
            # Predict Er
            self.neutral_model.load_weights(model_weights['Er'])
            pred_Er = np.ones(len(self.smiles_1), dtype=np.float32)
            for k in range(len(self.smiles_1)):
                to_predict = combined_vetor_all[k].reshape(1, 256, 1)
                pred_Er[k] = self.neutral_model(to_predict)
            
            # Save results
            self._save_predictions(pred_Tg, pred_Er, molar_ratio, output_file)

    def _save_predictions(self, pred_Tg: np.ndarray, pred_Er: np.ndarray, 
                         molar_ratio: float, output_file: str) -> None:
        """Save prediction results to CSV file"""
        good_sample = list(range(len(self.smiles_1)))
        selected_smiles1 = [self.smiles_1[index] for index in good_sample]
        selected_smiles2 = [self.smiles_2[index] for index in good_sample]
        selected_Er = [pred_Er[index] for index in good_sample]
        selected_Tg = [pred_Tg[index] for index in good_sample]
        
        molar_ratio1 = np.ones(len(selected_smiles1)) * molar_ratio
        molar_ratio2 = np.ones(len(selected_smiles1)) * (1 - molar_ratio)
        
        # Create DataFrame with proper column names
        df = pd.DataFrame({
            'SMILE1': selected_smiles1,
            'SMILE2': selected_smiles2,
            'Molar_Ratio_1': molar_ratio1,
            'Molar_Ratio_2': molar_ratio2,
            'Predicted_Er': selected_Er,
            'Predicted_Tg': selected_Tg
        })
        
        # Save to CSV, append if file exists
        if os.path.exists(output_file):
            df.to_csv(output_file, mode='a', header=False, index=False)
        else:
            df.to_csv(output_file, index=False)

def main():
    # Define model paths
    model_paths = {
        'smiles_to_latent': "Yans_code/Blog_simple_smi2lat7_150",
        'latent_to_states': "Yans_code/Blog_simple_lat7state7_150",
        'sample': "Yans_code/Blog_simple_samplemodel7_150"
    }
    
    # Define model weights
    model_weights = {
        'Tg': 'Yans_code/conv1d_model1_Tg245_3.h5',
        'Er': 'Yans_code/conv1d_model1_Er245_2.h5'
    }
    
    # Initialize predictor
    predictor = ChengModelPredictor(
        model_paths=model_paths,
        smiles_file='Yans_code/smiles.xlsx'
    )
    
    # Run predictions
    predictor.predict_properties(
        model_weights=model_weights,
        output_file='Yans_code/prediction_smiles.csv'
    )

if __name__ == "__main__":
    main()