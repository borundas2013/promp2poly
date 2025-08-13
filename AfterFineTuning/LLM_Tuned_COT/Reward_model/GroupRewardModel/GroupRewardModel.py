import sys
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
sys.path.append(str(parent_dir))
from pathlib import Path
from rdkit import Chem
from Reward_model.dual_smile_process import *

class GroupRewardModel:
    def __init__(self):
        pass

    def calculate_reactivity_reward(self, smiles1, smiles2, group1, group2):
        try:
            # Convert SMILES to molecules
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)
            
            if mol1 is None or mol2 is None:
                return False
            reactivity,group1_count,group2_count = check_reaction_validity_with_Fixed_groups(smiles1, smiles2, group1, group2)
            
            # Calculate reward based on reactivity and functional group counts
            if not reactivity:
                reward = 0.0
                
            # Base reward for having valid reactive groups
            reward = 1.0
            
            # Additional reward based on number of functional groups
            # Normalize by typical max count of 4 groups
            group_reward = (min(group1_count, 2) + min(group2_count, 2)) / 4.0
            print("Group Reward: ", group_reward,group1_count,group2_count)
            
            # Combine rewards
            total_reward = 0.7  * reward + 0.3 * group_reward
            
            return total_reward
                
           
        except Exception as e:
            print(f"Error checking groups: {e}")
            return False
