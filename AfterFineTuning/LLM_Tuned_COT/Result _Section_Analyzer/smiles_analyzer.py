"""
Main SMILES analyzer module
"""
import json
import pandas as pd
import numpy as np
import os
import sys
import re
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import os
import sys

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from base_analyzer import BaseAnalyzer

# Add DeepSeek directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
deepseek_dir = os.path.join(parent_dir, 'DeepSeek')
sys.path.append(deepseek_dir)

from dual_smile_process import process_dual_monomer_data

# Add Result_Section_Analyzer directory to path for Property_Prediction
sys.path.append(parent_dir)
from Property_Prediction.predict import predict_property

class SMILESAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__()
        # Initialize empty arrays for reference data
        self.monomer1 = np.array([])
        self.monomer2 = np.array([])
        self.er = np.array([])
        self.tg = np.array([])
        
        # Initialize analysis variables
        self.total_pairs = 0
        self.data = []
        
        # TODO: Load reference data when files are available
        data_dir = os.path.join(os.path.dirname(current_dir), 'Data')
        if os.path.exists(os.path.join(data_dir, 'unique_smiles_Er.csv')) and \
           os.path.exists(os.path.join(data_dir, 'smiles.xlsx')):
            try:
                self.monomer1, self.monomer2, self.er, self.tg = process_dual_monomer_data(
                    os.path.join(data_dir, 'unique_smiles_Er.csv'),
                    os.path.join(data_dir, 'smiles.xlsx')
                )
                self.monomer1 = np.array(self.monomer1)
                self.monomer2 = np.array(self.monomer2)
            except Exception as e:
                print(f"Error loading reference data: {str(e)}")

    def get_property_from_prompt(self, prompt):
        """Extract Tg and Er values from prompt"""
        text = prompt.lower()
        
        # Patterns for Tg
        tg_patterns = [
            # First format with °C
            r"(\d+\.?\d*)\s*(?:°c|degrees?\s*c|celsius)\s*(?:tg|glass transition)",
            # Second format with Tg first
            r"tg\s*(?:≈|~|=|:|is)?\s*(\d+\.?\d*)\s*(?:°c|degrees?\s*c|celsius)",
            # Very specific Tg pattern
            r"glass transition temperature\s*(?:≈|~|=|:|is)?\s*(\d+\.?\d*)\s*(?:°c|degrees?\s*c|celsius)",
        ]
        
        # Patterns for Er
        er_patterns = [
            # First format with MPa
            r"(\d+\.?\d*)\s*(?:mpa|MPa|mega\s*pascal|MP|mega pascal)\s*(?:er|Er|elastic recovery|stress recovery)",
            # Very specific Er pattern
            r"Er\s*(?:≈|~|=|:|is)?\s*(\d+\.?\d*)\s*(?:mpa|MPa|mega\s*pascal|MP|mega pascal)",
        ]

        # Search for Tg using multiple patterns
        tg_value = None
        for pattern in tg_patterns:
            tg_match = re.search(pattern, text, re.IGNORECASE)
            if tg_match:
                try:
                    tg_value = float(tg_match.group(1))
                    break
                except (ValueError, IndexError):
                    continue

        # Search for Er using multiple patterns
        er_value = None
        for pattern in er_patterns:
            er_match = re.search(pattern, text, re.IGNORECASE)
            if er_match:
                try:
                    er_value = float(er_match.group(1))
                    break
                except (ValueError, IndexError):
                    continue

        return tg_value, er_value

    def is_valid_smiles(self, smiles):
        try:
            Chem.MolFromSmiles(smiles)
            return True
        except ValueError:
            return False

    def analyze(self, input_files):
        """Analyze SMILES pairs from input files"""
        seen_pairs = set()
        
        for file_path in input_files:
            try:
                with open(file_path, 'r') as f:
                    entries = json.load(f)
                
                for entry in entries:
                    all_pairs = []
                    if 'unique_pairs' in entry:
                        all_pairs.extend(entry['unique_pairs'])
                    if 'duplicates' in entry:
                        all_pairs.extend(entry['duplicates'])
                        
                    for pair in all_pairs:
                        self.total_pairs += 1
                        smile1, smile2 = pair.get('smile1', ''), pair.get('smile2', '')
                        
                        # Check SMILES validity
                        valid_smile1 = self.is_valid_smiles(smile1)
                        valid_smile2 = self.is_valid_smiles(smile2)
                        valid_smiles = valid_smile1 and valid_smile2
                        
                        # Check uniqueness
                        pair_key = tuple(sorted([smile1, smile2]))
                        is_unique = pair_key not in seen_pairs
                        if is_unique:
                            seen_pairs.add(pair_key)
                        
                        # Store analysis results
                        analysis_result = {
                            'prompt': pair.get('prompt', ''),
                            'smile1': smile1,
                            'smile2': smile2,
                            'valid_smiles': valid_smiles,
                            'is_unique': is_unique,
                            'temperature': pair.get('temperature', 0.0)
                        }
                        self.data.append(analysis_result)
                        
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
        
        # Print summary
        valid_pairs = sum(1 for entry in self.data if entry['valid_smiles'])
        unique_pairs = sum(1 for entry in self.data if entry['is_unique'])
        
        print("\nSMILES Analysis Summary:")
        print(f"Total pairs analyzed: {self.total_pairs}")
        print(f"Valid SMILES pairs: {valid_pairs} ({valid_pairs/self.total_pairs*100:.2f}%)")
        print(f"Unique pairs: {unique_pairs} ({unique_pairs/self.total_pairs*100:.2f}%)")
        
        # Create visualizations
        self.visualize()

    def visualize(self, save_path=None):
        """Create visualizations for property analysis"""
        if self.data is None or len(self.data) == 0:
            print("No data available for visualization")
            return
            
        save_path = save_path or self.output_dir
        if not save_path:
            print("No save path specified. Plots will not be saved.")
            return
            
        # Create error distribution plots
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(self.tg_errors, bins=20, alpha=0.7)
        plt.title('Distribution of Tg Errors')
        plt.xlabel('Error (°C)')
        plt.ylabel('Count')
        
        plt.subplot(1, 2, 2)
        plt.hist(self.er_errors, bins=20, alpha=0.7)
        plt.title('Distribution of Er Errors')
        plt.xlabel('Error (MPa)')
        plt.ylabel('Count')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'property_errors.png'))
        plt.close()
