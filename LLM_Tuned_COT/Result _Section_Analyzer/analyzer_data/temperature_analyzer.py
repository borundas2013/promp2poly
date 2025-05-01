"""
Temperature analyzer module for analyzing temperature distributions in SMILES pairs.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import os
import sys

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

import os
import sys

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from base_analyzer import BaseAnalyzer

class TemperatureAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__()
        self.temperature_data = []
        
    def analyze(self, input_files):
        """
        Analyze temperature distribution of SMILES pairs
        Args:
            input_files (list): List of JSON file paths to analyze
        Returns:
            pd.DataFrame: DataFrame containing temperature analysis results
        """
        data = self.load_json_data(input_files)
        all_pairs = self.extract_pairs(data)
        
        temperature_data = []
        for pair in all_pairs:
            if 'temperature' not in pair or 'smile1' not in pair or 'smile2' not in pair:
                continue
                
            temperature_data.append({
                'temperature': pair['temperature'],
                'smile1': pair['smile1'],
                'smile2': pair['smile2'],
                'source_file': os.path.basename(pair.get('source_file', 'unknown'))
            })
        
        if not temperature_data:
            print("No temperature data found in the input files.")
            return None
        
        self.data = pd.DataFrame(temperature_data)
        self._print_statistics()
        return self.data
    
    def _print_statistics(self):
        """Print basic temperature statistics"""
        if self.data is None or len(self.data) == 0:
            return
            
        print("\nTemperature Statistics:")
        print(f"Total unique SMILES pairs: {len(self.data)}")
        print(f"Temperature range: {self.data['temperature'].min():.1f}°C to {self.data['temperature'].max():.1f}°C")
        print(f"Mean temperature: {self.data['temperature'].mean():.1f}°C")
        print(f"Median temperature: {self.data['temperature'].median():.1f}°C")
        
        # Find most common temperature
        temp_counts = self.data['temperature'].value_counts()
        most_common_temp = temp_counts.index[0]
        most_common_count = temp_counts.iloc[0]
        print(f"\nMost frequent temperature: {most_common_temp:.1f}°C (appears {most_common_count} times)")
        
        # Show top 3 temperatures by frequency
        print("\nTop 3 temperatures by frequency:")
        for temp, count in temp_counts.head(3).items():
            print(f"{temp:.1f}°C: {count} pairs")
        self.visualize()
    
    def visualize(self, save_path=None):
        """
        Create visualizations for temperature distribution
        Args:
            save_path (str, optional): Directory to save plots. Defaults to output_dir if set.
        """
        if self.data is None or len(self.data) == 0:
            print("No data available for visualization")
            return
            
        save_path = save_path or self.output_dir
        if not save_path:
            print("No save path specified. Plots will not be saved.")
            return
            
        # Temperature distribution plot
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.data, x='temperature', bins=20)
        plt.title('Distribution of Temperatures in Unique SMILES Pairs')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Count')
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'temperature_distribution.png'))
        plt.close()
        
        # Additional visualizations can be added here
