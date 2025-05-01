"""
Main script to run SMILES pair analysis
"""
import os
import sys
import argparse
import matplotlib
import json
import datetime

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
    print("\nRunning temperature analysis...")
    temp_analyzer.analyze(json_files)
    
    
    # Run property analysis
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

def main():
    input_dir = 'GPT4o/Output'
    output_dir = 'GPT4o/Output'
    model_name = 'GPT4o'
    run_analysis(input_dir, output_dir, model_name)

    input_dir = 'Llama32/Output'
    output_dir = 'Llama32/Output'
    model_name = 'Llama32'
    run_analysis(input_dir, output_dir, model_name)

    input_dir = 'MistralAI/Output'
    output_dir = 'MistralAI/Output'
    model_name = 'MistralAI'
    run_analysis(input_dir, output_dir, model_name)

    input_dir = 'DeepSeek/Output'
    output_dir = 'DeepSeek/Output'
    model_name = 'DeepSeek'
    run_analysis(input_dir, output_dir, model_name)

if __name__ == "__main__":
    main()
