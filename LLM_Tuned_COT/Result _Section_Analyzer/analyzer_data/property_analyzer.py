import json
import re
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from base_analyzer import BaseAnalyzer

# Add parent directory to Python path for property prediction
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Property_Prediction.predict import predict_property

class PropertyAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__()
        self.data = []
        self.total_pairs = 0
        self.pairs_with_properties = 0
        self.tg_absolute_errors = []
        self.er_absolute_errors = []
        self.tg_actual = []
        self.tg_predicted = []
        self.er_actual = []
        self.er_predicted = []
        
    def get_property_from_prompt(self, prompt):
        text = prompt.lower()
        
        tg_patterns = [
            r"(?:tg|tg:|tg=|glass transition temperature|Tg|Tg:|Tg=)\s*(?:of|:|=|around|near|approximately|~)?\s*(\d+\.?\d*)\s*(?:°c|°C|C|c|degrees|deg)",
            r"(?:tg|tg:|tg=|glass transition temperature|Tg|Tg:|Tg=)\s*(?:≈|~|about|around|approximately)?\s*(\d+\.?\d*)\s*(?:°c|°C|C|c|degrees|deg)",
            r"(\d+\.?\d*)\s*(?:°c|°C|C|c|degrees|deg)\s*(?:tg|Tg|glass transition temperature)",
            r"Tg\s*(?:≈|~|=|:|is)?\s*(\d+\.?\d*)\s*(?:°c|°C|C|c|degrees|deg)",
        ]

        er_patterns = [
            r"(?:er|er:|er=|elastic recovery|stress recovery|Er |Er=|Er:?)\s*(?:of|:|=|around|near|approximately|~)?\s*(\d+\.?\d*)\s*(?:mpa|MPa|mega\s*pascal|MP|mega pascal)",
            r"(?:er|er:|er=|elastic recovery|stress recovery|Er |Er=|Er:?)\s*(?:≈|~|about|around|approximately)?\s*(\d+\.?\d*)\s*(?:mpa|MPa|mega\s*pascal|MP|mega pascal)",
            r"(\d+\.?\d*)\s*(?:mpa|MPa|mega\s*pascal|MP|mega pascal)\s*(?:er|Er|elastic recovery|stress recovery)",
            r"Er\s*(?:≈|~|=|:|is)?\s*(\d+\.?\d*)\s*(?:mpa|MPa|mega\s*pascal|MP|mega pascal)",
        ]

        tg_value = None
        for pattern in tg_patterns:
            tg_match = re.search(pattern, text, re.IGNORECASE)
            if tg_match:
                try:
                    tg_value = float(tg_match.group(1))
                    break
                except (ValueError, IndexError):
                    continue

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

    def analyze(self, input_files):
        """Analyze property predictions from input files"""
        self.data = []
        self.tg_absolute_errors = []
        self.er_absolute_errors = []
        self.tg_actual = []
        self.tg_predicted = []
        self.er_actual = []
        self.er_predicted = []
        self.pairs_with_properties = 0

        data = self.load_json_data(input_files)
        all_pairs = self.extract_pairs(data)
        
        

        for pair in all_pairs:
            self.total_pairs += 1
            prompt = pair.get('prompt', '')
            smile1 = pair.get('smile1', '')
            smile2 = pair.get('smile2', '')
                        
            tg_value, er_value = self.get_property_from_prompt(prompt)
            if tg_value is None or er_value is None:
                continue
                            
            self.pairs_with_properties += 1
            scores = predict_property(smile1, smile2)
                        
            tg_error = scores['tg_score'] - tg_value
            er_error = scores['er_score'] - er_value
                        
            # Store actual and predicted values
            self.tg_actual.append(tg_value)
            self.tg_predicted.append(scores['tg_score'])
            self.er_actual.append(er_value)
            self.er_predicted.append(scores['er_score'])
                        
            # Store absolute errors
            self.tg_absolute_errors.append(abs(tg_error))
            self.er_absolute_errors.append(abs(er_error))
                        
            analysis_result = {
                            'prompt': prompt,
                            'smile1': smile1,
                            'smile2': smile2,
                            'tg_actual': tg_value,
                            'tg_predicted': scores['tg_score'],
                            'er_actual': er_value,
                            'er_predicted': scores['er_score'],
                            'tg_error': tg_error,
                            'er_error': er_error
                        }
            self.data.append(analysis_result)
                        
            # print(f"\nPrompt: {prompt}")
            # print(f"Target Tg: {tg_value:.2f}°C, Predicted: {scores['tg_score']:.2f}°C (Error: {tg_error:.2f}°C)")
            # print(f"Target Er: {er_value:.2f}MPa, Predicted: {scores['er_score']:.2f}MPa (Error: {er_error:.2f}MPa)")
            # print("-" * 80)

        # Only visualize if we have data
        if self.pairs_with_properties > 0:
            self.visualize(self.tg_absolute_errors, self.er_absolute_errors)
        else:
            print("\nNo property data to visualize.")

    def visualize(self, tg_errors, er_errors):
        """Create error distribution plots for Tg and Er"""
        plt.style.use('bmh')
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('Property Prediction Analysis', fontsize=16, y=1.02)
        
        # Create subplots with larger size
        gs = plt.GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])  # Tg scatter plot
        ax2 = fig.add_subplot(gs[0, 1])  # Er scatter plot
        ax3 = fig.add_subplot(gs[1, 0])  # Tg error distribution
        ax4 = fig.add_subplot(gs[1, 1])  # Er error distribution
        
        # Colors
        actual_color = '#2ecc71'    # Green for actual values
        predicted_color = '#e74c3c'  # Red for predicted values
        connect_color = '#95a5a6'    # Gray for connecting lines
        
        # Sort values by actual Tg/Er to make trends clearer
        tg_indices = np.argsort(self.tg_actual)
        er_indices = np.argsort(self.er_actual)
        
        # Plot Tg comparison
        x = np.arange(len(self.tg_actual))
        
        # Plot actual values
        ax1.plot(x, [self.tg_actual[i] for i in tg_indices], 
                color=actual_color, label='Actual Tg', 
                marker='o', markersize=4, linestyle='-', linewidth=1, alpha=0.8)
        
        # Plot predicted values
        ax1.plot(x, [self.tg_predicted[i] for i in tg_indices], 
                color=predicted_color, label='Predicted Tg', 
                marker='o', markersize=4, linestyle='-', linewidth=1, alpha=0.8)
        
        ax1.set_title('Tg: Actual vs Predicted Values', fontsize=14, pad=10)
        ax1.set_xlabel('Sample Index (sorted by Actual Tg)', fontsize=12)
        ax1.set_ylabel('Temperature (°C)', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot Er comparison
        # Plot actual values
        ax2.plot(x, [self.er_actual[i] for i in er_indices], 
                color=actual_color, label='Actual Er', 
                marker='o', markersize=4, linestyle='-', linewidth=1, alpha=0.8)
        
        # Plot predicted values
        ax2.plot(x, [self.er_predicted[i] for i in er_indices], 
                color=predicted_color, label='Predicted Er', 
                marker='o', markersize=4, linestyle='-', linewidth=1, alpha=0.8)
        
        ax2.set_title('Er: Actual vs Predicted Values', fontsize=14, pad=10)
        ax2.set_xlabel('Sample Index (sorted by Actual Er)', fontsize=12)
        ax2.set_ylabel('Er (MPa)', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Plot error distributions
        sns.histplot(tg_errors, bins=30, ax=ax3, color=predicted_color, alpha=0.7)
        ax3.set_title('Distribution of Tg Prediction Errors', fontsize=14, pad=10)
        ax3.set_xlabel('Error (°C)', fontsize=12)
        ax3.set_ylabel('Count', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        sns.histplot(er_errors, bins=30, ax=ax4, color=predicted_color, alpha=0.7)
        ax4.set_title('Distribution of Er Prediction Errors', fontsize=14, pad=10)
        ax4.set_xlabel('Error (MPa)', fontsize=12)
        ax4.set_ylabel('Count', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, 'property_analysis.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
