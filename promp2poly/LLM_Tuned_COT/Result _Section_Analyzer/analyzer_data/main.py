"""
Main script to run SMILES pair analysis
"""
import os
import sys
import argparse
import matplotlib
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from temperature_analyzer import TemperatureAnalyzer
from property_analyzer import PropertyAnalyzer
from group_analyzer import GroupAnalyzer
from smiles_pair_analyzer import SMILESPairAnalyzer

def print_comprehensive_stats(temp_analyzer, property_analyzer, group_analyzer, smiles_analyzer, model_name):
    print("\n=== Comprehensive Analysis Results ===\n")
    
    # Initialize report dictionary
    report = {
        "model_name": model_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "smiles_metrics": {},
        "property_metrics": {},
        "group_metrics": {}
    }
    
    # SMILES Validity and Uniqueness
    total_pairs = smiles_analyzer.total_pairs
    valid_pairs = sum(1 for entry in smiles_analyzer.data if entry.get('valid_smiles', False))
    unique_pairs = sum(1 for entry in smiles_analyzer.data if entry.get('is_unique', False))
    
    validity_rate = (valid_pairs / total_pairs * 100) if total_pairs > 0 else 0
    uniqueness_rate = (unique_pairs / total_pairs * 100) if total_pairs > 0 else 0
    
    # Store SMILES metrics
    report["smiles_metrics"] = {
        "total_pairs": total_pairs,
        "valid_pairs": valid_pairs,
        "unique_pairs": unique_pairs,
        "validity_rate": round(validity_rate, 2),
        "uniqueness_rate": round(uniqueness_rate, 2)
    }
    
    print(f"1. SMILES Quality Metrics:")
    print(f"   - Validity Rate: {validity_rate:.2f}%")
    print(f"   - Uniqueness Rate: {uniqueness_rate:.2f}%")
    if len(smiles_analyzer.data) > 0:
        unique_prompts = len(set(entry['prompt'] for entry in smiles_analyzer.data))
        avg_pairs = unique_pairs/unique_prompts if unique_prompts > 0 else 0
        report["smiles_metrics"]["unique_prompts"] = unique_prompts
        report["smiles_metrics"]["avg_pairs_per_query"] = round(avg_pairs, 2)
        print(f"   - Average Unique Pairs per Query: {avg_pairs:.2f}" if unique_prompts > 0 else "   - Average Unique Pairs per Query: 0.00")
    else:
        report["smiles_metrics"]["unique_prompts"] = 0
        report["smiles_metrics"]["avg_pairs_per_query"] = 0
        print("   - Average Unique Pairs per Query: 0.00")
    
    # Property Alignment
    if hasattr(property_analyzer, 'data') and property_analyzer.data:
        property_alignment = sum(1 for entry in property_analyzer.data if abs(entry.get('tg_error', float('inf'))) <= 10 
                               and abs(entry.get('er_error', float('inf'))) <= 10)
        alignment_rate = (property_alignment / len(property_analyzer.data) * 100)
        
        report["property_metrics"] = {
            "total_analyzed": len(property_analyzer.data),
            "property_alignment": property_alignment,
            "alignment_rate": round(alignment_rate, 2)
        }
        
        print(f"\n2. Property Alignment:")
        print(f"   - Property Match Rate (±10 units): {alignment_rate:.2f}%")
        if hasattr(property_analyzer, 'tg_absolute_errors') and property_analyzer.tg_absolute_errors:
            mean_tg_error = sum(property_analyzer.tg_absolute_errors)/len(property_analyzer.tg_absolute_errors)
            report["property_metrics"]["mean_tg_error"] = round(mean_tg_error, 2)
            print(f"   - Mean Absolute Tg Error: {mean_tg_error:.2f}°C")
        if hasattr(property_analyzer, 'er_absolute_errors') and property_analyzer.er_absolute_errors:
            mean_er_error = sum(property_analyzer.er_absolute_errors)/len(property_analyzer.er_absolute_errors)
            report["property_metrics"]["mean_er_error"] = round(mean_er_error, 2)
            print(f"   - Mean Absolute Er Error: {mean_er_error:.2f}MPa")
    else:
        report["property_metrics"] = {"available": False}
        print("\n2. Property Alignment:")
        print("   - No property data available")
    
    # Group Analysis
    if hasattr(group_analyzer, 'results') and group_analyzer.results:
        correct_assignments = sum(1 for result in group_analyzer.results if result.get('correct_group_assignment', False))
        reactive_pairs = sum(1 for result in group_analyzer.results if result.get('reaction_validity', False))
        total_analyzed = len(group_analyzer.results)
        
        report["group_metrics"] = {
            "total_analyzed": total_analyzed,
            "correct_assignments": correct_assignments,
            "reactive_pairs": reactive_pairs
        }
        
        print(f"\n3. Chemical Group Analysis:")
        if total_analyzed > 0:
            matching_accuracy = (correct_assignments/total_analyzed*100)
            reactive_rate = (reactive_pairs/total_analyzed*100)
            report["group_metrics"]["matching_accuracy"] = round(matching_accuracy, 2)
            report["group_metrics"]["reactive_rate"] = round(reactive_rate, 2)
            print(f"   - Group Matching Accuracy: {matching_accuracy:.2f}%")
            print(f"   - Reactive Group Success Rate: {reactive_rate:.2f}%")
        else:
            report["group_metrics"]["matching_accuracy"] = 0
            report["group_metrics"]["reactive_rate"] = 0
            print("   - No groups analyzed")
    else:
        report["group_metrics"] = {"available": False}
        print("\n3. Chemical Group Analysis:")
        print("   - No group data available")
    
    # Save report to JSON file
    output_dir = os.path.join('analysis_reports')
    os.makedirs(output_dir, exist_ok=True)
    report_file = os.path.join(output_dir, f'{model_name}_analysis_report.json')
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"\nAnalysis report saved to: {report_file}")
    return report

def run_analysis(input_dir, output_dir, model_name=''):
    """Run analysis on input files and save results to output directory"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set output directories
    analysis_dir = os.path.join(output_dir, 'Analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Initialize analyzers
    temp_analyzer = TemperatureAnalyzer()
    property_analyzer = PropertyAnalyzer()
    group_analyzer = GroupAnalyzer()
    smiles_analyzer = SMILESPairAnalyzer()
    
    # Set output directories for all analyzers
    temp_analyzer.set_output_directory(analysis_dir)
    property_analyzer.set_output_directory(analysis_dir)
    group_analyzer.set_output_directory(analysis_dir)
    smiles_analyzer.set_output_directory(analysis_dir)
    
    # Get all JSON files
    json_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                 if f.endswith('_s.json')]

    print(f"Running analysis for {model_name if model_name else 'input'} files...")
    
    # Run temperature analysis
    # print("\nRunning temperature analysis...")
    # temp_analyzer.analyze(json_files)
    
    
    # # Run property analysis
    print("\nRunning property analysis...")
    property_analyzer.analyze(json_files)
    
    # Run group analysis
    print("\nRunning group analysis...")

    group_analyzer.analyze(json_files)
    
    # Run SMILES pair analysis
    print("\nRunning SMILES pair analysis...")
    smiles_analyzer.analyze(json_files)

    # Print comprehensive statistics
    print_comprehensive_stats(temp_analyzer, property_analyzer, group_analyzer, smiles_analyzer, model_name)



def main2():
    """
    Main function to analyze and compare temperature distributions across models
    """
    # Model configurations
    models = {
        'GPT4o': ('GPT4o/Output', 'GPT4o/Output'),
        'Llama32': ('Llama32/Output', 'Llama32/Output'),
        'MistralAI': ('MistralAI/Output', 'MistralAI/Output'),
        'DeepSeek': ('DeepSeek/Output', 'DeepSeek/Output')
    }
    
    # Collect temperature data for each model
    temp_analyzer = TemperatureAnalyzer()
    temp_analyzer.set_output_directory("analysis_results")
    model_temp_data = {}
    
    for model_name, (input_dir, output_dir) in models.items():
        print(f"\nProcessing {model_name}...")
        # Get JSON files from directory
        json_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                     if f.endswith('_s.json')]
        if not json_files:
            print(f"No JSON files found in {input_dir}")
            continue
            
        unique_temp_data, duplicate_temp_data = temp_analyzer.analyze_model_temperature(json_files)
        model_temp_data[model_name] = {'unique': unique_temp_data, 'duplicate': duplicate_temp_data}

    #print(model_temp_data)
    
    # Generate comparative visualizations
    if model_temp_data:
        print("\nGenerating comparative plots...")
        #temp_analyzer.create_model_comparison_plot(model_temp_data, "analysis_results")
        temp_analyzer.create_alternative_plots(model_temp_data, "analysis_results")


def compare_model_properties():
    """Compare property statistics across different models"""
    # Model configurations
    models = {
        'GPT4o': ('GPT4o/Output/Analysis', 'GPT4o/Output'),
        'Llama32': ('Llama32/Output/Analysis', 'Llama32/Output'),
        'MistralAI': ('MistralAI/Output/Analysis', 'MistralAI/Output/'),
        'DeepSeek': ('DeepSeek/Output/Analysis', 'DeepSeek/Output')
    }

    # Collect property data for each model
    property_analyzer = PropertyAnalyzer()
    model_property_data = {}

    for model_name, (input_dir, output_dir) in models.items():
        print(f"\nProcessing {model_name} properties...")
        property_analyzer.set_output_directory(os.path.join(output_dir, 'Analysis'))
        
        # # Get JSON files from directory
        # json_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
        #              if f.endswith('_s.json')]
        # csv_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
        # if f.endswith('.csv')]
        csv_files = []
        
        csv_file = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
        if f.startswith('all_reactive')]
        csv_files.append(csv_file[0])
        
        csv_file = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
        if f.startswith('all_non_reactive')]
        csv_files.append(csv_file[0])
        print(csv_files)
        if not csv_files:
            print(f"No CSV files found in {input_dir}")
            continue

        # Analyze properties for this model
        #property_analyzer.analyze(json_files)
        property_analyzer.analyze_reactive(csv_files)
        
        # Store the property data
        model_property_data[model_name] = {
            'tg_actual': property_analyzer.tg_actual.copy(),
            'tg_predicted': property_analyzer.tg_predicted.copy(),
            'er_actual': property_analyzer.er_actual.copy(),
            'er_predicted': property_analyzer.er_predicted.copy(),
            'tg_errors': property_analyzer.tg_absolute_errors.copy(),
            'er_errors': property_analyzer.er_absolute_errors.copy()
        }
        print(f"\n=== Analyzing {model_name} Model ===")
        #analyze_high_property_samples(property_analyzer, model_name)

    # Create comparative visualizations
    if model_property_data:
        print("\nGenerating comparative property plots...")
        plt.style.use('bmh')
        sns.set_style("whitegrid")
        
        # Create a single figure with subplots
        fig = plt.figure(figsize=(10, 8))  # More compact width
        gs = plt.GridSpec(2, 1, figure=fig, height_ratios=[1, 0.8], hspace=0.3)
        fig.suptitle('Property Prediction Analysis by Model', 
                    fontsize=16, y=0.98, weight='bold')
        
        # Create subplots
        ax_box = fig.add_subplot(gs[0])  # Box plot
        ax_bar = fig.add_subplot(gs[1])  # Bar plot
        
        # Prepare data
        model_names = []
        tg_errors = []
        er_errors = []
        mean_tg_errors = []
        mean_er_errors = []
        
        for model_name, data in model_property_data.items():
            model_names.append(model_name)
            tg_errors.append(data['tg_errors'])
            er_errors.append(data['er_errors'])
            mean_tg_errors.append(np.mean(data['tg_errors']))
            mean_er_errors.append(np.mean(data['er_errors']))
        
        # Box plots
        positions = np.arange(len(model_names)) * 1.5  # Reduced spacing
        width = 0.6  # Reduced width
        
        # Define colors
        tg_color = '#ffd6a5'  # Lighter orange for Tg
        er_color = '#a8e6cf'  # Lighter teal for Er
        
        bp_tg = ax_box.boxplot(tg_errors, positions=positions-width/2, widths=width, 
                              patch_artist=True, 
                              medianprops=dict(color="black", linewidth=2),
                              flierprops=dict(marker='o', markerfacecolor='black', markersize=4),
                              whiskerprops=dict(linewidth=2, color='black'),
                              capprops=dict(linewidth=2, color='black'),
                              boxprops=dict(linewidth=2))
        bp_er = ax_box.boxplot(er_errors, positions=positions+width/2, widths=width, 
                              patch_artist=True, 
                              medianprops=dict(color="black", linewidth=2),
                              flierprops=dict(marker='o', markerfacecolor='black', markersize=4),
                              whiskerprops=dict(linewidth=2, color='black'),
                              capprops=dict(linewidth=2, color='black'),
                              boxprops=dict(linewidth=2))
        
        # Color the boxplots
        for box in bp_tg['boxes']:
            box.set_facecolor(tg_color)
            box.set_alpha(0.9)
            box.set_linewidth(2)
        for box in bp_er['boxes']:
            box.set_facecolor(er_color)
            box.set_alpha(0.9)
            box.set_linewidth(2)
        
        ax_box.set_xticks(positions)
        ax_box.set_xticklabels(model_names, rotation=0, ha='center', 
                              fontsize=12, weight='bold')
        ax_box.set_ylabel('Absolute Error', fontsize=14, weight='bold')
        ax_box.legend([bp_tg["boxes"][0], bp_er["boxes"][0]], ['Tg (°C)', 'Er (MPa)'],
                     bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12, frameon=True)
        
        # Set y-axis limits and grid for box plot
        ax_box.set_ylim(bottom=0)  # Let matplotlib auto-determine the bottom limit
        ax_box.yaxis.grid(True, linestyle='-', alpha=0.2)
        ax_box.set_axisbelow(True)
        ax_box.tick_params(axis='both', which='major', labelsize=12)
        
        # Bar plot
        x = np.arange(len(model_names))
        width = 0.3  # Reduced width
        
        ax_bar.bar(x - width/2, mean_tg_errors, width, label='Tg Error (°C)',
                  color='#ff4d4d', alpha=0.9)  # Bright red for Tg error
        ax_bar.bar(x + width/2, mean_er_errors, width, label='Er Error (MPa)',
                  color='#3366ff', alpha=0.9)  # Bright blue for Er error
        
        # Add value labels on bars
        for i, v in enumerate(mean_tg_errors):
            ax_bar.text(i - width/2, v + 0.5, f'{v:.1f}°C', ha='center', va='bottom',
                       fontsize=12, weight='bold')
        for i, v in enumerate(mean_er_errors):
            ax_bar.text(i + width/2, v + 0.5, f'{v:.1f}MPa', ha='center', va='bottom',
                       fontsize=12, weight='bold')
        
        ax_bar.set_ylabel('Mean Absolute Error', fontsize=14, weight='bold')
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(model_names, rotation=0, ha='center', 
                              fontsize=12, weight='bold')
        ax_bar.legend(fontsize=12, frameon=True, bbox_to_anchor=(1.02, 1), loc='upper left')
        
        # Set y-axis limits and grid for bar plot
        ax_bar.set_ylim(bottom=0)  # Let matplotlib auto-determine the bottom limit
        ax_bar.yaxis.grid(True, linestyle='-', alpha=0.2)
        ax_bar.set_axisbelow(True)
        ax_bar.tick_params(axis='both', which='major', labelsize=12)
        
        # Adjust margins and spacing
        plt.subplots_adjust(left=0.12, right=0.95, bottom=0.2, top=0.93)
        
        # Save plots
        output_dir = 'analysis_results'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as PNG
        plt.savefig(os.path.join(output_dir, 'property_analysis_combined_new_2.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        
        # Save as PDF
        plt.savefig(os.path.join(output_dir, 'property_analysis_combined_new_2.pdf'), 
                   format='pdf', bbox_inches='tight')
        
        plt.close()
        
        # Print summary statistics
        print("\nSummary Statistics:")
        for model_name, data in model_property_data.items():
            print(f"\n{model_name}:")
            print(f"  Tg Mean Absolute Error: {np.mean(data['tg_errors']):.2f}°C")
            print(f"  Er Mean Absolute Error: {np.mean(data['er_errors']):.2f}MPa")
            print(f"  Number of samples: {len(data['tg_actual'])}")

def analyze_high_property_samples(property_analyzer, model_name):
    """Analyze samples with:
    1. High Tg (>300 degrees)
    2. Er value higher than Tg value
    Save results to CSV files and return counts
    """
    if not property_analyzer.data:
        print(f"\n{model_name}: No data available for high property analysis.")
        return
    
    # Calculate max and min values
    max_tg = max(sample['tg_predicted'] for sample in property_analyzer.data)
    min_tg = min(sample['tg_predicted'] for sample in property_analyzer.data)
    max_er = max(sample['er_predicted'] for sample in property_analyzer.data)
    min_er = min(sample['er_predicted'] for sample in property_analyzer.data)
    
    # Print max and min values
    print(f"\n{model_name} - Property Ranges:")
    print(f"  Tg Range: {min_tg:.2f}°C to {max_tg:.2f}°C")
    print(f"  Er Range: {min_er:.2f}MPa to {max_er:.2f}MPa")
    
    # Find samples with high Tg (>300 degrees)
    high_tg_samples = [sample for sample in property_analyzer.data 
                      if sample['tg_predicted'] > 300]

    high_er_samples = [sample for sample in property_analyzer.data 
                      if sample['er_predicted'] > 200]
    
    # # Find samples where Er is higher than Tg
    # er_higher_than_tg = [sample for sample in property_analyzer.data 
    #                     if sample['er_predicted'] > sample['tg_predicted']]
    
    # Sort both lists by their respective key properties
    high_tg_samples = sorted(high_tg_samples, 
                           key=lambda x: x['tg_predicted'], 
                           reverse=True)
    high_er_samples = sorted(high_er_samples, 
                             key=lambda x: x['er_predicted'], 
                             reverse=True)
    
    # Create output directory if it doesn't exist
    output_dir = 'analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save high Tg samples to CSV
    if high_tg_samples:
        high_tg_df = pd.DataFrame(high_tg_samples)
        high_tg_df['model'] = model_name
        high_tg_csv = os.path.join(output_dir, f'high_tg_samples_new.csv')
        
        # Append to existing CSV if it exists, otherwise create new
        if os.path.exists(high_tg_csv):
            high_tg_df.to_csv(high_tg_csv, mode='a', header=False, index=False)
        else:
            high_tg_df.to_csv(high_tg_csv, index=False)
    
    # Save high Er samples to CSV
    if high_er_samples:
        high_er_df = pd.DataFrame(high_er_samples)
        high_er_df['model'] = model_name
        high_er_csv = os.path.join(output_dir, f'high_er_samples_new.csv')
        
        # Append to existing CSV if it exists, otherwise create new
        if os.path.exists(high_er_csv):
            high_er_df.to_csv(high_er_csv, mode='a', header=False, index=False)
        else:
            high_er_df.to_csv(high_er_csv, index=False)
    
    # # Save Er > Tg samples to CSV
    # if er_higher_than_tg:
    #     er_tg_df = pd.DataFrame(er_higher_than_tg)
    #     er_tg_df['model'] = model_name
    #     er_tg_df['er_tg_difference'] = er_tg_df['er_predicted'] - er_tg_df['tg_predicted']
    #     er_tg_csv = os.path.join(output_dir, f'er_higher_than_tg_samples.csv')
        
    #     # Append to existing CSV if it exists, otherwise create new
    #     if os.path.exists(er_tg_csv):
    #         er_tg_df.to_csv(er_tg_csv, mode='a', header=False, index=False)
    #     else:
    #         er_tg_df.to_csv(er_tg_csv, index=False)
    
    # Print only the counts
    print(f"\n{model_name}:")
    print(f"  Samples with Tg > 300°C: {len(high_tg_samples)}")
    print(f"  Samples with Er > 200 MPa: {len(high_er_samples)}")

def main():
    # # Keep existing analysis
    # input_dir = 'GPT4o/Output'
    # output_dir = 'GPT4o/Output'
    # model_name = 'GPT4o'
    # run_analysis(input_dir, output_dir, model_name)

    # input_dir = 'Llama32/Output'
    # output_dir = 'Llama32/Output'
    # model_name = 'Llama32'
    # run_analysis(input_dir, output_dir, model_name)

    # input_dir = 'MistralAI/Output'
    # output_dir = 'MistralAI/Output'
    # model_name = 'MistralAI'
    # run_analysis(input_dir, output_dir, model_name)

    # input_dir = 'DeepSeek/Output'
    # output_dir = 'DeepSeek/Output'
    # model_name = 'DeepSeek'
    # run_analysis(input_dir, output_dir, model_name)

    # Add comparative analysis
    compare_model_properties()

if __name__ == "__main__":
    main()
