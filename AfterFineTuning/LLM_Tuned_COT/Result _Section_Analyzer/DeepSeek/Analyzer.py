import json
import pandas as pd
import numpy as np
import os
import sys
import re
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from dual_smile_process import *
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Property_Prediction.predict import predict_property

class TSMPAnalyzer:
    def __init__(self):
        # Load reference data
        self.monomer1, self.monomer2, self.er, self.tg = process_dual_monomer_data(
            '../Data/unique_smiles_Er.csv',
            '../Data/smiles.xlsx'
        )
        self.monomer1 = np.array(self.monomer1)
        self.monomer2 = np.array(self.monomer2)
        
    def load_tsmp_data(self, file_path):
        """Load TSMP data from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data_list = json.load(f)
            
            all_unique_pairs = []
            for data_dict in data_list:
                unique_pairs = data_dict.get('unique_pairs', [])
                for pair in unique_pairs:
                    extracted_pair = {
                        'smile1': pair.get('smile1', ''),
                        'smile2': pair.get('smile2', ''),
                        'reaction': pair.get('reaction', False),
                        'temperature': pair.get('temperature', 0),
                        'groups': pair.get('groups', []),
                        'prompt': pair.get('prompt', '')
                    }
                    all_unique_pairs.append(extracted_pair)
            
            return all_unique_pairs
        
        except FileNotFoundError:
            print(f"Error: File {file_path} not found")
            return []
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {file_path}")
            return []
        except Exception as e:
            print(f"Error occurred while loading data: {str(e)}")
            return []

    def filter_by_reaction(self, data, reaction_status=True):
        """Filter pairs based on reaction status"""
        return [item for item in data if item.get('reaction') == reaction_status]

    def analyze_and_save_tsmp_data(self, input_files):
        """Analyze TSMP data and save to CSV files"""
        all_reactive_pairs = []
        all_non_reactive_pairs = []
        
        for input_file in input_files:
            all_pairs = self.load_tsmp_data(input_file)
            reactive_pairs = self.filter_by_reaction(all_pairs, True)
            non_reactive_pairs = self.filter_by_reaction(all_pairs, False)
            
            # Process reactive pairs
            all_reactive_pairs.extend([{
                'SMILE1': pair['smile1'],
                'SMILE2': pair['smile2'],
                'Temperature': pair.get('temperature', 0),
                'Prompt': pair.get('prompt', ''),
                'Reactive_groups': pair.get('groups', []),
                'Source_File': input_file.split('/')[-1]
            } for pair in reactive_pairs])
            
            # Process non-reactive pairs
            all_non_reactive_pairs.extend([{
                'SMILE1': pair['smile1'],
                'SMILE2': pair['smile2'],
                'Temperature': pair.get('temperature', 0),
                'Prompt': pair.get('prompt', ''),
                'Non_Reactive_groups': pair.get('groups', []),
                'Source_File': input_file.split('/')[-1]
            } for pair in non_reactive_pairs])
            
            print(f"\nStatistics for {input_file.split('/')[-1]}:")
            print(f"Total pairs analyzed: {len(all_pairs)}")
            print(f"Reactive pairs found: {len(reactive_pairs)}")
            print(f"Non-reactive pairs found: {len(non_reactive_pairs)}")
        
        # Create DataFrames and add SL column
        reactive_df = pd.DataFrame(all_reactive_pairs)
        reactive_df.insert(0, 'SL', range(1, len(reactive_df) + 1))
        
        non_reactive_df = pd.DataFrame(all_non_reactive_pairs)
        non_reactive_df.insert(0, 'SL', range(1, len(non_reactive_df) + 1))
        
        # Save to CSV
        reactive_df.to_csv('DeepSeek/Output/small_model/all_reactive_pairs.csv', index=False)
        non_reactive_df.to_csv('DeepSeek/Output/small_model/all_non_reactive_pairs.csv', index=False)
        
        print("\nTotal Statistics:")
        print(f"Total reactive pairs: {len(all_reactive_pairs)}")
        print(f"Total non-reactive pairs: {len(all_non_reactive_pairs)}")


    def extract_chemical_groups(self, prompts):
        group_patterns = {
            'epoxy': Constants.EPOXY_SMARTS,
            'imine': Constants.IMINE_SMARTS,
            'vinyl': Constants.VINYL_SMARTS,
            'thiol': Constants.THIOL_SMARTS,
            'acrylic': Constants.ACRYL_SMARTS,
            'benzene': Constants.BEZEN_SMARTS,
            'hydroxyl': Constants.Hydroxyl_SMARTS
        }
        
        def find_groups(prompt):
            groups = []
            prompt_lower = prompt.lower()
            
            # Pattern to match explicit group mentions: group_name(formula)
            explicit_pattern = r'(\w+)\(([\w\d=()]+)\)'
            explicit_matches = re.findall(explicit_pattern, prompt_lower)
            
            # Add explicit matches
            for group_name, formula in explicit_matches:
                # Check if the group name matches any known variations
                for known_group, variations in Constants.GROUP_VARIATIONS.items():
                    if group_name in variations or any(var in group_name for var in variations):
                        # Normalize formula to match SMARTS pattern
                        formula = formula.upper()
                        groups.append((known_group, group_patterns[known_group]))
                        break
            
            # Look for implicit formula mentions - only if no explicit mentions found for that group
            found_groups = {g[0] for g in groups}  # Track which groups we've found
            for group_name, formula in group_patterns.items():
                if group_name not in found_groups:  # Only look for groups we haven't found yet
                    # Check both the formula and group name variations
                    variations = Constants.GROUP_VARIATIONS[group_name]
                    if any(var in prompt_lower for var in variations):
                        groups.append((group_name, formula))
            
            return groups
        
        results = []
        for prompt in prompts:
            groups = find_groups(prompt)
            if groups:
                results.append({
                    'prompt': prompt,
                    'groups': groups
                })
            
        return results

    def check_combinations_in_dataset(self):
        """Check combinations and individual SMILE matches"""
        reactive_df = pd.read_csv('DeepSeek/Output/small_model/all_reactive_pairs.csv')
        non_reactive_df = pd.read_csv('DeepSeek/Output/small_model/all_non_reactive_pairs.csv')
        
        # Initialize counters
        stats = {
            'reactive': {
                'matches': 0,
                'reverse_matches': 0,
                'smile1_matches': 0,
                'smile2_matches': 0
            },
            'non_reactive': {
                'matches': 0,
                'reverse_matches': 0,
                'smile1_matches': 0,
                'smile2_matches': 0
            }
        }
        
        # Check reactive pairs
        for _, row in reactive_df.iterrows():
            self._process_pair(row, stats['reactive'])
        
        # Check non-reactive pairs
        for _, row in non_reactive_df.iterrows():
            self._process_pair(row, stats['non_reactive'])
        
        self._print_statistics(reactive_df, non_reactive_df, stats)

    def _process_pair(self, row, stats):
        """Process individual pair and update statistics"""
        smile1, smile2 = row['SMILE1'], row['SMILE2']
        
        # Check combinations
        forward_match = np.any((self.monomer1 == smile1) & (self.monomer2 == smile2))
        reverse_match = np.any((self.monomer1 == smile2) & (self.monomer2 == smile1))
        
        if forward_match:
            stats['matches'] += 1
        if reverse_match:
            stats['reverse_matches'] += 1
            
        # Check individual smiles
        if np.any(self.monomer1 == smile1) or np.any(self.monomer2 == smile1):
            stats['smile1_matches'] += 1
        if np.any(self.monomer1 == smile2) or np.any(self.monomer2 == smile2):
            stats['smile2_matches'] += 1

    def _print_statistics(self, reactive_df, non_reactive_df, stats):
        """Print detailed statistics"""
        # Print reactive statistics
        print("\nMatching Statistics:")
        self._print_category_statistics("Reactive Pairs", reactive_df, stats['reactive'])
        self._print_category_statistics("Non-Reactive Pairs", non_reactive_df, stats['non_reactive'])
        
        # Print overall statistics
        total_pairs = len(reactive_df) + len(non_reactive_df)
        total_matches = (stats['reactive']['matches'] + stats['reactive']['reverse_matches'] +
                        stats['non_reactive']['matches'] + stats['non_reactive']['reverse_matches'])
        total_smile1_matches = stats['reactive']['smile1_matches'] + stats['non_reactive']['smile1_matches']
        total_smile2_matches = stats['reactive']['smile2_matches'] + stats['non_reactive']['smile2_matches']
        
        print("\nOverall Statistics:")
        print(f"Total pairs analyzed: {total_pairs}")
        print(f"Total pair matches found: {total_matches}")
        print(f"Overall percentage of pair matches: {(total_matches / total_pairs) * 100:.2f}%")
        print(f"Total individual SMILE1 matches: {total_smile1_matches} ({(total_smile1_matches / total_pairs) * 100:.2f}%)")
        print(f"Total individual SMILE2 matches: {total_smile2_matches} ({(total_smile2_matches / total_pairs) * 100:.2f}%)")

    def _print_category_statistics(self, category_name, df, stats):
        """Print statistics for a specific category"""
        print(f"\n{category_name}:")
        print(f"Total pairs checked: {len(df)}")
        print(f"Forward matches found: {stats['matches']}")
        print(f"Reverse matches found: {stats['reverse_matches']}")
        total_matches = stats['matches'] + stats['reverse_matches']
        print(f"Total matches (forward + reverse): {total_matches}")
        print(f"Percentage of matches: {(total_matches / len(df)) * 100:.2f}%")
        print("\nIndividual SMILE matches:")
        print(f"SMILE1 matches: {stats['smile1_matches']} ({(stats['smile1_matches'] / len(df)) * 100:.2f}%)")
        print(f"SMILE2 matches: {stats['smile2_matches']} ({(stats['smile2_matches'] / len(df)) * 100:.2f}%)")

    def draw_molecule_pairs(self, output_dir='DeepSeek/Output/small_model/Molecules'):
        
        # Create output directories if they don't exist
        reactive_dir = os.path.join(output_dir, 'reactive')
        non_reactive_dir = os.path.join(output_dir, 'non_reactive')
        os.makedirs(reactive_dir, exist_ok=True)
        os.makedirs(non_reactive_dir, exist_ok=True)

        # Load the CSV files
        reactive_df = pd.read_csv('DeepSeek/Output/small_model/all_reactive_pairs.csv')
        non_reactive_df = pd.read_csv('DeepSeek/Output/small_model/all_non_reactive_pairs.csv')

        # Process reactive pairs
        print("\nDrawing Reactive Pairs...")
        self._draw_pairs(reactive_df, reactive_dir, "reactive")

        # Process non-reactive pairs
        print("\nDrawing Non-Reactive Pairs...")
        self._draw_pairs(non_reactive_df, non_reactive_dir, "non_reactive")

    def _draw_pairs(self, df, output_dir, pair_type):
      
        for _, row in df.iterrows():
            try:
                # Convert SMILES to RDKit molecules
              
                mol1 = Chem.MolFromSmiles(row['SMILE1'])
                mol2 = Chem.MolFromSmiles(row['SMILE2'])

                if mol1 is None or mol2 is None:
                    print(f"Warning: Could not parse SMILES for pair {row['SL']}")
                    continue

                # Generate 2D coordinates for better visualization
                AllChem.Compute2DCoords(mol1)
                AllChem.Compute2DCoords(mol2)

                # Prepare legends based on pair type and reactive groups
                legend1 = f'SMILE1\nTemp: {row["Temperature"]}'
                
                # Handle reactive groups for both types
                if pair_type == "reactive":
                    groups = row.get("Reactive_groups", [])
                    legend2 = f'SMILE2\nReactive groups: {groups}' if groups and groups != '[]' else 'SMILE2'
                else:
                    groups = row.get("Non_Reactive_groups", [])
                    legend2 = f'SMILE2\nNon-reactive groups: {groups}' if groups and groups != '[]' else 'SMILE2'

                # Add SL to legends
                #legend1 = f'SL: {row["SL"]}\n' + legend1
                #legend2 = f'SL: {row["SL"]}\n' + legend2

                # Create a combined image
                img = Draw.MolsToGridImage(
                    [mol1, mol2],
                    legends=[legend1, legend2],
                    subImgSize=(400, 400),
                    returnPNG=False
                )

                # Save the image using SL in filename
                filename = f"{row['SL']}.png"  # Zero-padded 4-digit number
                img.save(os.path.join(output_dir, filename))

                if (row['SL']) % 100 == 0:
                    print(f"Processed {row['SL']} {pair_type} pairs")
              

            except Exception as e:
                print(f"Error processing pair SL{row['SL']}: {str(e)}")
    def check_smiles_for_group(self, smiles, smarts_pattern):
        """Check if a SMILES string contains a specific chemical group"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            pattern = Chem.MolFromSmarts(smarts_pattern)
            return len(mol.GetSubstructMatches(pattern)) > 0
        except:
            return False
    

    def analyze_group_matches(self, input_files):
        """
        Analyze JSON files to match prompts with chemical groups and verify SMILES pairs.
        This method:
        1. Reads JSON files and extracts prompts
        2. Extracts chemical groups from prompts
        3. Checks if SMILES pairs contain the mentioned groups
        """
        results = []
        
        # Process each JSON file
        for file_path in input_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Process each entry in the JSON
                for entry in data:
                    # Get all pairs (both unique and duplicates)
                    all_pairs = []
                    if 'unique_pairs' in entry:
                        all_pairs.extend(entry['unique_pairs'])
                    # if 'duplicates' in entry:
                    #     all_pairs.extend(entry['duplicates'])
                    
                    for pair in all_pairs:
                        if 'prompt' not in pair or 'smile1' not in pair or 'smile2' not in pair:
                            continue
                            
                        # Extract groups from prompt
                        prompt_groups = self.extract_chemical_groups([pair['prompt']])
                        if not prompt_groups:
                            continue
                            
                        groups = prompt_groups[0]['groups']
                        if len(groups) < 2:  # Need at least two groups for comparison
                            continue

                       
                            
                        # Check each SMILES for group matches
                        group1, group2 = groups[0], groups[1]  # Take first two groups
                        smile1, smile2 = pair['smile1'], pair['smile2']
                        reaction_validity = pair['reaction']
                        #reaction_validity, reactive_groups = check_reaction_validity(smile1, smile2)

                        # Check combinations of groups in SMILES
                        match_results = {
                            'prompt': pair['prompt'],
                            'group1': group1,
                            'group2': group2,
                            'smile1': smile1,
                            'smile2': smile2,
                            'smile1_contains_group1': self.check_smiles_for_group(smile1, group1[1]),
                            'smile1_contains_group2': self.check_smiles_for_group(smile1, group2[1]),
                            'smile2_contains_group1': self.check_smiles_for_group(smile2, group1[1]),
                            'smile2_contains_group2': self.check_smiles_for_group(smile2, group2[1]),
                            'reaction_validity': reaction_validity,
                        }
                        
                        # Calculate match score
                        correct_assignment = (
                            (match_results['smile1_contains_group1'] and 
                             match_results['smile2_contains_group2']) or
                            (match_results['smile1_contains_group2'] and 
                             match_results['smile2_contains_group1'])
                        )

                        #correct_assignment = (match_results['smile1_contains_group1'] and match_results['smile2_contains_group2'])
                        match_results['correct_group_assignment'] = correct_assignment
                        results.append(match_results)
                        
                        # Print detailed analysis
                        self._print_group_analysis(match_results)
                        
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
        
        self._print_group_summary(results)
        return results
    
    def _print_group_analysis(self, match_results):
        """Print detailed analysis for a single SMILES pair"""
        print(f"\nAnalyzing Prompt: {match_results['prompt']}")
        print(f"Groups mentioned: {match_results['group1'][0]} and {match_results['group2'][0]}")
        print(f"SMILE1: {match_results['smile1']}")
        print(f"- Contains {match_results['group1'][0]}: {match_results['smile1_contains_group1']}")
        print(f"- Contains {match_results['group2'][0]}: {match_results['smile1_contains_group2']}")
        print(f"SMILE2: {match_results['smile2']}")
        print(f"- Contains {match_results['group1'][0]}: {match_results['smile2_contains_group1']}")
        print(f"- Contains {match_results['group2'][0]}: {match_results['smile2_contains_group2']}")
        print(f"Correct group assignment: {match_results['correct_group_assignment']}")
        print("-" * 80)
        
    def _print_group_summary(self, results):
        """Print summary statistics for all analyzed pairs"""
        total = len(results)
        if total == 0:
            print("\nNo pairs were analyzed.")
            return
        
        correct = sum(1 for r in results if r['correct_group_assignment'])
        reactive = sum(1 for r in results if r['reaction_validity'])
        print(f"\nSummary Statistics:")
        print(f"Total pairs analyzed: {total}")
        print(f"Correct group assignments: {correct} ({(correct/total)*100:.2f}%)")
        print(f"Reaction validity: {reactive} ({(reactive/total)*100:.2f}%)")


    def get_property_from_prompt(self, prompt):
        text = prompt.lower()  # Convert to lowercase for case-insensitive matching
        
        tg_patterns = [
            # Standard format with °C
            r"(?:tg|tg:|tg=|glass transition temperature|Tg|Tg:|Tg=)\s*(?:of|:|=|around|near|approximately|~)?\s*(\d+\.?\d*)\s*(?:°c|°C|C|c|degrees|deg)",
            # Approximate format with °C
            r"(?:tg|tg:|tg=|glass transition temperature|Tg|Tg:|Tg=)\s*(?:≈|~|about|around|approximately)?\s*(\d+\.?\d*)\s*(?:°c|°C|C|c|degrees|deg)",
            # Number first format with °C
            r"(\d+\.?\d*)\s*(?:°c|°C|C|c|degrees|deg)\s*(?:tg|Tg|glass transition temperature)",
            # Very specific Tg pattern
            r"Tg\s*(?:≈|~|=|:|is)?\s*(\d+\.?\d*)\s*(?:°c|°C|C|c|degrees|deg)",
        ]

        # More flexible patterns for Er
        er_patterns = [
            # Standard format with MPa
            r"(?:er|er:|er=|elastic recovery|stress recovery|Er |Er=|Er:?)\s*(?:of|:|=|around|near|approximately|~)?\s*(\d+\.?\d*)\s*(?:mpa|MPa|mega\s*pascal|MP|mega pascal)",
            # Approximate format with MPa
            r"(?:er|er:|er=|elastic recovery|stress recovery|Er |Er=|Er:?)\s*(?:≈|~|about|around|approximately)?\s*(\d+\.?\d*)\s*(?:mpa|MPa|mega\s*pascal|MP|mega pascal)",
            # Number first format with MPa
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

        # Print the found values
        #print(f"Tg: {tg_value} °C" if tg_value is not None else "Tg: None °C")
        #print(f"Er: {er_value} MPa" if er_value is not None else "Er: None MPa")

        # Return the values
        return tg_value, er_value

    def analyze_property_match(self, input_files):
        # Lists to store errors
        tg_errors = []
        er_errors = []
        tg_absolute_errors = []
        er_absolute_errors = []
        total_pairs = 0
        pairs_with_properties = 0

        for file_path in input_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Process each entry in the JSON
                for entry in data:
                    # Get all pairs (both unique and duplicates)
                    all_pairs = []
                    if 'unique_pairs' in entry:
                        all_pairs.extend(entry['unique_pairs'])
                    if 'duplicates' in entry:
                        all_pairs.extend(entry['duplicates'])
                    
                    for pair in all_pairs:
                        if 'prompt' not in pair or 'smile1' not in pair or 'smile2' not in pair:
                            continue
                        total_pairs += 1
                        prompt = pair['prompt']
                        smile1 = pair['smile1']
                        smile2 = pair['smile2']

                        tg_value, er_value = self.get_property_from_prompt(prompt)

                        # Check if both properties are found
                        if tg_value is None or er_value is None:
                            print(f"Missing property values for prompt: {prompt}")
                            continue

                        pairs_with_properties += 1
                        scores = predict_property(smile1, smile2)

                        # Calculate errors
                        tg_error = scores['tg_score'] - tg_value
                        er_error = scores['er_score'] - er_value
                        
                        tg_errors.append(tg_error)
                        er_errors.append(er_error)
                        tg_absolute_errors.append(abs(tg_error))
                        er_absolute_errors.append(abs(er_error))

                        # Print individual results
                        print(f"Prompt: {prompt}")
                        print(f"Target Tg: {tg_value:.2f} °C, Predicted: {scores['tg_score']:.2f} °C (Error: {tg_error:.2f} °C)")
                        print(f"Target Er: {er_value:.2f} MPa, Predicted: {scores['er_score']:.2f} MPa (Error: {er_error:.2f} MPa)")
                        print("-" * 80)

            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")

        # Calculate and print summary statistics
        if pairs_with_properties > 0:
            mean_tg_error = sum(tg_errors) / len(tg_errors)
            mean_er_error = sum(er_errors) / len(er_errors)
            mean_tg_abs_error = sum(tg_absolute_errors) / len(tg_absolute_errors)
            mean_er_abs_error = sum(er_absolute_errors) / len(er_absolute_errors)
            
            print("\nSummary Statistics:")
            print(f"Total pairs analyzed: {total_pairs}")
            print(f"Pairs with valid properties: {pairs_with_properties} ({(pairs_with_properties/total_pairs)*100:.2f}%)")
            print("\nError Statistics:")
            print(f"Mean Tg Error: {mean_tg_error:.2f} °C")
            print(f"Mean Er Error: {mean_er_error:.2f} MPa")
            print(f"Mean Absolute Tg Error: {mean_tg_abs_error:.2f} °C")
            print(f"Mean Absolute Er Error: {mean_er_abs_error:.2f} MPa")

    def analyze_temperature_distribution(self, input_files):
        """
        Analyze and visualize temperature distribution of unique SMILES pairs
        """
        temperature_data = []
        
        for file_path in input_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                for entry in data:
                    all_pairs = []
                    if 'unique_pairs' in entry:
                        all_pairs.extend(entry['unique_pairs'])
                    
                    for pair in all_pairs:
                        if 'temperature' not in pair or 'smile1' not in pair or 'smile2' not in pair:
                            continue
                            
                        temperature_data.append({
                            'temperature': pair['temperature'],
                            'smile1': pair['smile1'],
                            'smile2': pair['smile2'],
                            'source_file': os.path.basename(file_path)
                        })
            
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
        
        if not temperature_data:
            print("No temperature data found in the input files.")
            return
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(temperature_data)
        
        # Basic statistics
        print("\nTemperature Statistics:")
        print(f"Total unique SMILES pairs: {len(df)}")
        print(f"Temperature range: {df['temperature'].min():.1f}°C to {df['temperature'].max():.1f}°C")
        print(f"Mean temperature: {df['temperature'].mean():.1f}°C")
        print(f"Median temperature: {df['temperature'].median():.1f}°C")
        
        # Create temperature distribution plot
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='temperature', bins=20)
        plt.title('Distribution of Temperatures in Unique SMILES Pairs')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Count')
        plt.savefig('temperature_distribution.png')
        plt.close()
        
       
        print("\nPairs per Source File:")
        source_counts = df['source_file'].value_counts()
        for source, count in source_counts.items():
            print(f"{source}: {count} pairs")
            
        return df

if __name__ == "__main__":
    analyzer = TSMPAnalyzer()

    file_1 = 'DeepSeek/Output/group_generated_responses_s.json'
    file_2 = 'DeepSeek/Output/properties_generated_responses_s.json'
    file_3 = 'DeepSeek/Output/mix_generated_responses_s.json'

