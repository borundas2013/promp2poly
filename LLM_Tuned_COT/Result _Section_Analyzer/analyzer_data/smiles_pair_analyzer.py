import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import os
from base_analyzer import BaseAnalyzer
import constants
from dual_smile_process import process_dual_monomer_data

class SMILESPairAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__()
        # Get paths relative to current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        data_dir = os.path.join(parent_dir, 'Data')
        
        # Initialize empty arrays
        self.monomer1 = np.array([])
        self.monomer2 = np.array([])
        self.er = np.array([])
        self.tg = np.array([])
        
        # Initialize analysis variables
        self.total_pairs = 0
        self.data = []
        self.reactive_pairs = []
        self.non_reactive_pairs = []
        
        # Try to load reference data if available
        try:
            if os.path.exists(data_dir):
                er_path = os.path.join(data_dir, 'unique_smiles_Er.csv')
                smiles_path = os.path.join(data_dir, 'smiles.xlsx')
                if os.path.exists(er_path) and os.path.exists(smiles_path):
                    self.monomer1, self.monomer2, self.er, self.tg = process_dual_monomer_data(er_path, smiles_path)
                    self.monomer1 = np.array(self.monomer1)
                    self.monomer2 = np.array(self.monomer2)
        except Exception as e:
            print(f"Warning: Could not load reference data: {str(e)}")
            print("Continuing without reference data...")
    
    def is_valid_smiles(self,smile1, smile2):
        if smile1 != "" and smile2 != "":
            mol1= Chem.MolFromSmiles(smile1)
            mol2= Chem.MolFromSmiles(smile2)
            if mol1 is not None and mol2 is not None:
                return True
        return False
        
    def filter_by_reaction(self, data, reaction_status=True):
        """Filter pairs based on reaction status"""
        return [item for item in data if item.get('reaction') == reaction_status]

    def load_tsmp_data(self, file_path):
        """Load TSMP data from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data_list = json.load(f)
            
            all_unique_pairs = []
            all_duplicate_pairs = []
            
            for data_dict in data_list:
                
                unique_pairs = data_dict.get('unique_pairs', [])
                duplicate_pairs = data_dict.get('duplicates', [])
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
                for pair in duplicate_pairs:
                    extracted_pair = {
                        'smile1': pair.get('smile1', ''),
                        'smile2': pair.get('smile2', ''),
                        'reaction': pair.get('reaction', False),
                        'temperature': pair.get('temperature', 0),
                        'groups': pair.get('groups', []),
                        'prompt': pair.get('prompt', '')
                    }
                    all_duplicate_pairs.append(extracted_pair)
                
                      
            return all_unique_pairs, all_duplicate_pairs
        
        except FileNotFoundError:
            print(f"Error: File {file_path} not found")
            return []
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {file_path}")
            return []
        except Exception as e:
            print(f"Error occurred while loading data: {str(e)}")
            return []

    def analyze_and_save_tsmp_data(self, input_files):
        self.reactive_pairs = []
        self.non_reactive_pairs = []
        
        # Create output directory
        os.makedirs(os.path.join(self.output_dir), exist_ok=True)
        
        for input_file in input_files:
            all_pairs, all_duplicate_pairs = self.load_tsmp_data(input_file)
            reactive_pairs = self.filter_by_reaction(all_pairs, True)
            non_reactive_pairs = self.filter_by_reaction(all_pairs, False)
            
            self.reactive_pairs.extend([{
                'SMILE1': pair['smile1'],
                'SMILE2': pair['smile2'],
                'Temperature': pair.get('temperature', 0),
                'Prompt': pair.get('prompt', ''),
                'Reactive_groups': pair.get('groups', []),
                'Source_File': input_file.split('/')[-1]
            } for pair in reactive_pairs])
            
            self.non_reactive_pairs.extend([{
                'SMILE1': pair['smile1'],
                'SMILE2': pair['smile2'],
                'Temperature': pair.get('temperature', 0),
                'Prompt': pair.get('prompt', ''),
                'Non_Reactive_groups': pair.get('groups', []),
                'Source_File': input_file.split('/')[-1]
            } for pair in non_reactive_pairs])
            
            print(f"\nStatistics for {input_file.split('/')[-1]}:")
            print(f"Total unique pairs analyzed: {len(all_pairs)}")
            print(f"Total duplicate pairs analyzed: {len(all_duplicate_pairs)}")
            print(f"Total pairs analyzed: {len(all_pairs) + len(all_duplicate_pairs)}")
            print(f"Reactive pairs found: {len(reactive_pairs)}")
            print(f"Non-reactive pairs found: {len(non_reactive_pairs)}")
        
        reactive_df = pd.DataFrame(self.reactive_pairs)
        reactive_df.insert(0, 'SL', range(1, len(reactive_df) + 1))
        
        non_reactive_df = pd.DataFrame(self.non_reactive_pairs)
        non_reactive_df.insert(0, 'SL', range(1, len(non_reactive_df) + 1))
        
        reactive_df.to_csv(os.path.join(self.output_dir, 'all_reactive_pairs.csv'), index=False)
        non_reactive_df.to_csv(os.path.join(self.output_dir, 'all_non_reactive_pairs.csv'), index=False)
        
        print("\nTotal Statistics:")
        print(f"Total reactive pairs: {len(self.reactive_pairs)}")
        print(f"Total non-reactive pairs: {len(self.non_reactive_pairs)}")

    def check_combinations_in_dataset(self):
        reactive_df = pd.read_csv(os.path.join(self.output_dir, 'all_reactive_pairs.csv'))
        non_reactive_df = pd.read_csv(os.path.join(self.output_dir, 'all_non_reactive_pairs.csv'))
        
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
        
        for _, row in reactive_df.iterrows():
            self._process_pair(row, stats['reactive'])
        
        for _, row in non_reactive_df.iterrows():
            self._process_pair(row, stats['non_reactive'])
        
        self._print_statistics(reactive_df, non_reactive_df, stats)
        return stats

    def _process_pair(self, row, stats):
        smile1, smile2 = row['SMILE1'], row['SMILE2']
        
        forward_match = np.any((self.monomer1 == smile1) & (self.monomer2 == smile2))
        reverse_match = np.any((self.monomer1 == smile2) & (self.monomer2 == smile1))
        
        if forward_match:
            stats['matches'] += 1
        if reverse_match:
            stats['reverse_matches'] += 1
            
        if np.any(self.monomer1 == smile1) or np.any(self.monomer2 == smile1):
            stats['smile1_matches'] += 1
        if np.any(self.monomer1 == smile2) or np.any(self.monomer2 == smile2):
            stats['smile2_matches'] += 1

    def _print_statistics(self, reactive_df, non_reactive_df, stats):
        print("\nMatching Statistics:")
        self._print_category_statistics("Reactive Pairs", reactive_df, stats['reactive'])
        self._print_category_statistics("Non-Reactive Pairs", non_reactive_df, stats['non_reactive'])
        
        total_pairs = len(reactive_df) + len(non_reactive_df)
        total_matches = (stats['reactive']['matches'] + stats['reactive']['reverse_matches'] +
                        stats['non_reactive']['matches'] + stats['non_reactive']['reverse_matches'])
        total_smile1_matches = stats['reactive']['smile1_matches'] + stats['non_reactive']['smile1_matches']
        total_smile2_matches = stats['reactive']['smile2_matches'] + stats['non_reactive']['smile2_matches']
        
        print("\nOverall Statistics:")
        print(f"Total pairs analyzed: {total_pairs}")
        print(f"Total pair matches found: {total_matches}")

    def _print_category_statistics(self, category_name, df, stats):
        print(f"\n{category_name}:")
        total_pairs = len(df)
        print(f"Total pairs checked: {total_pairs}")
        print(f"Forward matches found: {stats['matches']}")
        print(f"Reverse matches found: {stats['reverse_matches']}")
        total_matches = stats['matches'] + stats['reverse_matches']
        print(f"Total matches (forward + reverse): {total_matches}")
        if total_pairs > 0:
            print(f"Percentage of matches: {(total_matches / total_pairs) * 100:.2f}%")
        else:
            print("Percentage of matches: N/A (no pairs)")
        if total_pairs > 0:
            print("\nIndividual SMILE matches:")
            print(f"SMILE1 matches: {stats['smile1_matches']} ({(stats['smile1_matches'] / len(df)) * 100:.2f}%)")
            print(f"SMILE2 matches: {stats['smile2_matches']} ({(stats['smile2_matches'] / len(df)) * 100:.2f}%)")
        else:
            print("Individual SMILE matches: N/A (no pairs)")

    def draw_molecule_pairs(self):
        molecules_dir = os.path.join(self.output_dir, 'Molecules')
        reactive_dir = os.path.join(molecules_dir, 'reactive')
        non_reactive_dir = os.path.join(molecules_dir, 'non_reactive')
        os.makedirs(reactive_dir, exist_ok=True)
        os.makedirs(non_reactive_dir, exist_ok=True)

        reactive_df = pd.read_csv(os.path.join(self.output_dir, 'all_reactive_pairs.csv'))
        non_reactive_df = pd.read_csv(os.path.join(self.output_dir, 'all_non_reactive_pairs.csv'))

        print("\nDrawing Reactive Pairs...")
        self._draw_pairs(reactive_df, reactive_dir, "reactive")

        print("\nDrawing Non-Reactive Pairs...")
        self._draw_pairs(non_reactive_df, non_reactive_dir, "non_reactive")

    def _draw_pairs(self, df, output_dir, pair_type):
        for _, row in df.iterrows():
            try:
                mol1 = Chem.MolFromSmiles(row['SMILE1'])
                mol2 = Chem.MolFromSmiles(row['SMILE2'])

                if mol1 is None or mol2 is None:
                    print(f"Warning: Could not parse SMILES for pair {row['SL']}")
                    continue

                AllChem.Compute2DCoords(mol1)
                AllChem.Compute2DCoords(mol2)

                legend1 = f'SMILE1\nTemp: {row["Temperature"]}'
                
                if pair_type == "reactive":
                    groups = row.get("Reactive_groups", [])
                    legend2 = f'SMILE2\nReactive groups: {groups}' if groups and groups != '[]' else 'SMILE2'
                else:
                    groups = row.get("Non_Reactive_groups", [])
                    legend2 = f'SMILE2\nNon-reactive groups: {groups}' if groups and groups != '[]' else 'SMILE2'

                img = Draw.MolsToGridImage(
                    [mol1, mol2],
                    legends=[legend1, legend2],
                    subImgSize=(400, 400),
                    returnPNG=False
                )

                filename = f"{row['SL']}.png"
                img.save(os.path.join(output_dir, filename))

                if (row['SL']) % 100 == 0:
                    print(f"Processed {row['SL']} {pair_type} pairs")

            except Exception as e:
                print(f"Error processing pair SL{row['SL']}: {str(e)}")

    def analyze(self, input_files):
        """Analyze SMILES pairs from input files"""
        self.data = []
        self.total_pairs = 0
        
        for file_path in input_files:
            try:
                with open(file_path, 'r') as f:
                    entries = json.load(f)
                
                for entry in entries:
                    if 'unique_pairs' in entry:
                        for pair in entry['unique_pairs']:
                            self.total_pairs += 1
                            self.data.append({
                                'prompt': pair.get('prompt', ''),
                                'smile1': pair.get('smile1', ''),
                                'smile2': pair.get('smile2', ''),
                                'valid_smiles': self.is_valid_smiles(pair.get('smile1', ''), pair.get('smile2', '')),
                                'is_unique': True
                            })
                    
                    if 'duplicates' in entry:
                        for pair in entry['duplicates']:
                            self.total_pairs += 1
                            self.data.append({
                                'prompt': pair.get('prompt', ''),
                                'smile1': pair.get('smile1', ''),
                                'smile2': pair.get('smile2', ''),
                                'valid_smiles': self.is_valid_smiles(pair.get('smile1', ''), pair.get('smile2', '')),
                                'is_unique': False
                            })
                            
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
        
        # Run the original analysis
        self.analyze_and_save_tsmp_data(input_files)
        stats = self.check_combinations_in_dataset()
        self.draw_molecule_pairs()
        self.visualize(stats)

    def visualize(self, stats):
        """Create visualizations for SMILES pair analysis"""
        total_pairs = len(self.reactive_pairs) + len(self.non_reactive_pairs)
        if total_pairs == 0:
            print("No data available for visualization")
            return

        # Set style
        plt.style.use('bmh')  # Using a built-in style that's similar to seaborn
        
        # Create figure
        fig = plt.figure(figsize=(15, 7))
        fig.patch.set_facecolor('white')  # Set white background
        fig.suptitle('SMILES Pair Analysis Results', fontsize=14, y=1.02)
        
        # Plot pair distribution
        plt.subplot(121)
        pair_data = {
            'Reactive Pairs': len(self.reactive_pairs),
            'Non-Reactive Pairs': len(self.non_reactive_pairs)
        }
        colors = ['#3498db', '#e67e22']  # Professional blue and orange
        plt.pie(pair_data.values(), labels=pair_data.keys(), autopct='%1.1f%%', 
                labeldistance=1.2, pctdistance=0.8, colors=colors,
                wedgeprops={'edgecolor': 'white', 'linewidth': 1})
        plt.title('Distribution of SMILES Pairs', pad=20)

        # Plot match statistics
        plt.subplot(122)
        forward_matches = stats['reactive']['matches'] + stats['non_reactive']['matches']
        reverse_matches = stats['reactive']['reverse_matches'] + stats['non_reactive']['reverse_matches']
        no_matches = total_pairs - (forward_matches + reverse_matches)
        
        # Only include non-zero values in the pie chart
        match_data = {}
        if forward_matches > 0:
            match_data['Forward\nMatches'] = forward_matches
        if reverse_matches > 0:
            match_data['Reverse\nMatches'] = reverse_matches
        if no_matches > 0:
            match_data['No\nMatches'] = no_matches
            
        if len(match_data) == 1:
            # If only one type, use a single professional color
            colors = ['#2ecc71'] if 'No\nMatches' in match_data else ['#3498db']
        else:
            colors = ['#3498db', '#e74c3c', '#2ecc71'][:len(match_data)]  # Blue, Red, Green
            
        plt.pie(match_data.values(), labels=match_data.keys(), autopct='%1.1f%%',
                labeldistance=1.2, pctdistance=0.8, colors=colors,
                wedgeprops={'edgecolor': 'white', 'linewidth': 1})
        plt.title('Match Statistics', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'smiles_analysis.png'))
        plt.close()
