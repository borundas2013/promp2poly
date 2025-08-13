import json
import re
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ioff()
import seaborn as sns
import pandas as pd
from rdkit import Chem
from base_analyzer import BaseAnalyzer
import constants

class GroupAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__()
        self.results = []
        self.total_pairs = 0

    def extract_chemical_groups(self, prompts):
        group_patterns = {
            'epoxy': constants.EPOXY_SMARTS,
            'imine': constants.IMINE_SMARTS,
            'vinyl': constants.VINYL_SMARTS,
            'thiol': constants.THIOL_SMARTS,
            'acrylic': constants.ACRYL_SMARTS,
            'benzene': constants.BEZEN_SMARTS,
            'hydroxyl': constants.Hydroxyl_SMARTS
        }
        
        def find_groups(prompt):
            groups = []
            prompt_lower = prompt.lower()
            
            explicit_pattern = r'(\w+)\(([\w\d=()]+)\)'
            explicit_matches = re.findall(explicit_pattern, prompt_lower)
            
            for group_name, formula in explicit_matches:
                for known_group, variations in constants.GROUP_VARIATIONS.items():
                    if group_name in variations or any(var in group_name for var in variations):
                        formula = formula.upper()
                        groups.append((known_group, group_patterns[known_group]))
                        break
            
            found_groups = {g[0] for g in groups}
            for group_name, formula in group_patterns.items():
                if group_name not in found_groups:
                    variations = constants.GROUP_VARIATIONS[group_name]
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

    def check_smiles_for_group(self, smiles, smarts_pattern):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            pattern = Chem.MolFromSmarts(smarts_pattern)
            return len(mol.GetSubstructMatches(pattern)) > 0
        except:
            return False

    def analyze(self, input_files):
        self.results = []  # Reset results
        data = self.load_json_data(input_files)
        all_pairs = self.extract_pairs(data)
        
        for pair in all_pairs:
            if 'prompt' not in pair or 'smile1' not in pair or 'smile2' not in pair:
                continue
                            
            prompt_groups = self.extract_chemical_groups([pair['prompt']])
            if not prompt_groups:
                continue
                            
            groups = prompt_groups[0]['groups']
            if len(groups) < 2:
                continue

            group1, group2 = groups[0], groups[1]
            smile1, smile2 = pair['smile1'], pair['smile2']
            reaction_validity = pair['reaction']

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
                        
            correct_assignment = (
                            (match_results['smile1_contains_group1'] and 
                             match_results['smile2_contains_group2']) or
                            (match_results['smile1_contains_group2'] and 
                             match_results['smile2_contains_group1'])
                        )

            match_results['correct_group_assignment'] = correct_assignment
            self.results.append(match_results)  # Store in instance variable
                        
            #self._print_group_analysis(match_results)
                        
        self._print_group_summary(self.results)
        return self.results

    def _print_group_analysis(self, match_results):
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
        
        # Create visualization
        self.visualize(results)
        
    def visualize(self, results):
        """Create visualizations for group analysis results"""
        # Convert results to DataFrame for easier plotting
        df = pd.DataFrame(results)
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Group Assignment Success Rate
        success_data = {
            'Correct': df['correct_group_assignment'].sum(),
            'Incorrect': len(df) - df['correct_group_assignment'].sum()
        }
        plt.sca(ax1)
        plt.pie(success_data.values(), labels=success_data.keys(), autopct='%1.1f%%')
        plt.title('Group Assignment Success Rate')
        
        # Plot 2: Reaction Validity Distribution
        reaction_data = {
            'Reactive': df['reaction_validity'].sum(),
            'Non-reactive': len(df) - df['reaction_validity'].sum()
        }
        plt.sca(ax2)
        plt.pie(reaction_data.values(), labels=reaction_data.keys(), autopct='%1.1f%%')
        plt.title('Reaction Validity Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'group_analysis.png'))
        plt.close()
