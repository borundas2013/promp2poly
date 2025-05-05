"""
Temperature analyzer module for analyzing temperature distributions in SMILES pairs.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import os
import sys
import numpy as np

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
        self.output_dir = None
        
    def set_output_directory(self, directory):
        """Set the output directory for saving analysis results"""
        self.output_dir = directory
        os.makedirs(directory, exist_ok=True)
        
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

   
    def analyze_model_temperature(self, input_files):
        """
        Analyze temperature distribution for a specific model
        
        Args:
            input_files (list): List of JSON files to analyze
            
        Returns:
            tuple: (unique_data DataFrame, duplicates_data DataFrame)
        """
        data = self.load_json_data(input_files)
        unique_pairs, duplicates = self.extract_pairs_d(data, include_duplicates=True)
        
        temperature_unique_data = []
        temperature_duplicates_data = []
        for pair in unique_pairs:
            if 'temperature' not in pair or 'smile1' not in pair or 'smile2' not in pair:
                continue
            temperature_unique_data.append({
                'temperature': pair['temperature'],
            })
        
        for pair in duplicates:
            if 'temperature' not in pair or 'smile1' not in pair or 'smile2' not in pair:
                continue
            temperature_duplicates_data.append({
                'temperature': pair['temperature']
            })

        if not temperature_unique_data:
            return None
        
        self._unique_data = pd.DataFrame(temperature_unique_data)
        self._duplicates_data = pd.DataFrame(temperature_duplicates_data)
        return self._unique_data, self._duplicates_data

    def create_model_comparison_plot(self, model_data_dict, output_dir):
        """Create comparative visualizations of temperature distributions for different models"""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set global font properties
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
        
        processed_data = {}
        stats_text = []  # For collecting statistics
        stats_text.append("Temperature Distribution Statistics\n")
        stats_text.append("=" * 35 + "\n\n")
        
        for model_name, data in model_data_dict.items():
            if data is not None:
                # Get temperature counts for unique and duplicate pairs
                unique_temps = data['unique']
                duplicate_temps = data['duplicate']
                
                # Calculate statistics
                unique_counts = unique_temps['temperature'].value_counts()
                duplicate_counts = duplicate_temps['temperature'].value_counts()
                
                # Add to processed data for plotting
                processed_data[model_name] = {
                    'unique': unique_counts,
                    'duplicate': duplicate_counts
                }
                
                # Add statistics to text
                stats_text.append(f"Model: {model_name}\n")
                stats_text.append("-" * (len(model_name) + 7) + "\n")
                
                stats_text.append("\nUnique SMILES Pairs:")
                stats_text.append(f"Total count: {len(unique_temps)}")
                stats_text.append("Temperature distribution:")
                for temp, count in unique_counts.items():
                    stats_text.append(f"  {temp}°C: {count} pairs")
                
                stats_text.append("\nDuplicate SMILES Pairs:")
                stats_text.append(f"Total count: {len(duplicate_temps)}")
                stats_text.append("Temperature distribution:")
                for temp, count in duplicate_counts.items():
                    stats_text.append(f"  {temp}°C: {count} pairs")
                stats_text.append("\n" + "=" * 35 + "\n")

        if not processed_data:
            print("No data available for plotting")
            return

        # Save statistics to file
        with open(os.path.join(output_dir, 'temperature_statistics.txt'), 'w') as f:
            f.write('\n'.join(str(line) for line in stats_text))

        # Create bar plots with higher DPI for better quality
        plt.rcParams['figure.dpi'] = 300
        fig = plt.figure(figsize=(15, 12))
        
        # Create subplot grid with proper spacing
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.4)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        
        # Get all temperatures
        all_temps = set()
        for data in processed_data.values():
            all_temps.update(data['unique'].index)
            all_temps.update(data['duplicate'].index)
        x_labels = sorted(all_temps)
        x = np.arange(len(x_labels))
        width = 0.2
        
        # Calculate offsets for bars
        n_models = len(processed_data)
        offsets = np.linspace(-width * (n_models-1)/2, width * (n_models-1)/2, n_models)
        
        # Plot unique pairs
        for (model_name, data), offset in zip(processed_data.items(), offsets):
            counts = [data['unique'].get(temp, 0) for temp in x_labels]
            total = sum(counts)
            ax1.bar(x + offset, counts, width, label=f'{model_name} (n={total})')
        
        ax1.set_title('Distribution of Unique SMILES Pairs by Temperature', pad=15)
        ax1.set_xlabel('Temperature (°C)', labelpad=10)
        ax1.set_ylabel('Count', labelpad=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels(x_labels, rotation=0)
        ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot duplicate pairs
        for (model_name, data), offset in zip(processed_data.items(), offsets):
            counts = [data['duplicate'].get(temp, 0) for temp in x_labels]
            total = sum(counts)
            ax2.bar(x + offset, counts, width, label=f'{model_name} (n={total})')
        
        ax2.set_title('Distribution of Duplicate SMILES Pairs by Temperature', pad=15)
        ax2.set_xlabel('Temperature (°C)', labelpad=10)
        ax2.set_ylabel('Count', labelpad=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels(x_labels, rotation=0)
        ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Add main title with proper spacing
        fig.suptitle('Temperature Distribution Comparison Across Models', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        # Save as both PNG and PDF with high quality
        plt.savefig(os.path.join(output_dir, 'model_temperature_comparison.png'), 
                   dpi=300, bbox_inches='tight', pad_inches=0.3)
        plt.savefig(os.path.join(output_dir, 'model_temperature_comparison.pdf'), 
                   bbox_inches='tight', pad_inches=0.3)
        plt.close()
        
        print(f"\nFiles saved in {output_dir}:")
        print("- model_temperature_comparison.png")
        print("- model_temperature_comparison.pdf")
        print("- temperature_statistics.txt")