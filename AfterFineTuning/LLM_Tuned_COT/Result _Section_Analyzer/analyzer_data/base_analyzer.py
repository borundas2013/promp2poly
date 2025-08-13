"""
Base analyzer module providing common functionality for all analyzers.
"""
import json
import os
from abc import ABC, abstractmethod
import pandas as pd

class BaseAnalyzer(ABC):
    def __init__(self):
        """Initialize base analyzer"""
        self.data = None
        self.output_dir = None
    
    def set_output_directory(self, output_dir):
        """Set the output directory for saving results"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def load_json_data(self, file_paths):
        """Load data from JSON files"""
        all_data = []
        for file_path in file_paths:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    all_data.extend(data)
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
        return all_data
    
    def extract_pairs(self, data, include_duplicates=False):
        """Extract SMILES pairs from data"""
        all_pairs = []
        for entry in data:
            if 'unique_pairs' in entry:
                all_pairs.extend(entry['unique_pairs'])
            if include_duplicates and 'duplicates' in entry:
                all_pairs.extend(entry['duplicates'])
        return all_pairs

    def extract_pairs_d(self, data, include_duplicates=False):
        """Extract SMILES pairs from data"""
        unique_pairs = []
        duplicates = []
        for entry in data:
            if 'unique_pairs' in entry:
                unique_pairs.extend(entry['unique_pairs'])
            if include_duplicates and 'duplicates' in entry:
                duplicates.extend(entry['duplicates'])
        return unique_pairs, duplicates
    
    @abstractmethod
    def analyze(self, input_files):
        """Perform analysis on input files"""
        pass
    
    @abstractmethod
    def visualize(self, data):
        """Create visualizations from analyzed data"""
        pass
    
    def save_results(self, results, filename):
        """Save analysis results"""
        if self.output_dir:
            if isinstance(results, pd.DataFrame):
                results.to_csv(os.path.join(self.output_dir, filename), index=False)
            elif isinstance(results, dict):
                with open(os.path.join(self.output_dir, filename), 'w') as f:
                    json.dump(results, f, indent=2)
