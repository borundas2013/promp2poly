
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
sys.path.append(str(parent_dir))

# Now we can import from parent directories
from Reward_model.dual_smile_process import process_dual_monomer_data
from Reward_model.Constants import *

class NoveltyRewardModel:
    def __init__(self, radius=2, nBits=2048):
        self.radius = radius
        self.nBits = nBits
        self.training_pairs = []
        self.min_length = 20

    def set_training_pairs(self, training_pairs):
        """Set the training pairs for comparison"""
        self.training_pairs = training_pairs

    def _smiles_to_fp(self, smiles):
        """Convert SMILES to molecular fingerprint"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=self.radius, nBits=self.nBits)

    def _monomer_pair_similarity(self, gen_pair, train_pair):
        """Calculate similarity between two monomer pairs"""
        # Each pair = (monomer1, monomer2)
        fp1_a = self._smiles_to_fp(gen_pair[0])
        fp2_a = self._smiles_to_fp(gen_pair[1])
        fp1_b = self._smiles_to_fp(train_pair[0])
        fp2_b = self._smiles_to_fp(train_pair[1])

        if None in [fp1_a, fp2_a, fp1_b, fp2_b]:
            return 0.0

        # Compare both possible alignments: (m1→m1, m2→m2) and (m1→m2, m2→m1)
        sim1 = (DataStructs.TanimotoSimilarity(fp1_a, fp1_b) +
                DataStructs.TanimotoSimilarity(fp2_a, fp2_b)) / 2
        sim2 = (DataStructs.TanimotoSimilarity(fp1_a, fp2_b) +
                DataStructs.TanimotoSimilarity(fp2_a, fp1_b)) / 2

        return max(sim1, sim2)

    def calculate_novelty_and_diversity_score(self, gen_pair):
        """Calculate novelty score for a generated monomer pair"""
        if not self.training_pairs:
            raise ValueError("Training pairs not set. Call set_training_pairs first.")
            
        max_similarity = 0.0
        max_duplicate_score = 0.0
        max_length_penalty = 0.0
        for i in  range(len(self.training_pairs[0])):
            train_pair = (self.training_pairs[0][i], self.training_pairs[1][i])
            sim = self._monomer_pair_similarity(gen_pair, train_pair)
           
            max_similarity = max(max_similarity, sim)

            length_penalty = self.calculate_length_penalty(gen_pair[0], gen_pair[1])
            max_length_penalty = max(max_length_penalty, length_penalty)

        novelty_score = 1.0 - max_similarity
        
        length_penalty_score = 1.0 - max_length_penalty
        final_diversity_score = novelty_score + length_penalty_score
        total_reward = novelty_score# + final_diversity_score

        print("Novelty score: ", novelty_score)
        print("Diversity score: ", final_diversity_score)
        print("Length penalty score: ", length_penalty_score)
        print("Total reward: ", total_reward)
            #print(sim,max_similarity)

        return total_reward#,novelty_score
    

    def calculate_length_penalty(self, smiles1, smiles2):
        """Calculate penalty for short SMILES strings"""
        len1 = len(smiles1)
        len2 = len(smiles2)
        
        # Calculate how much shorter than min_length each SMILES is
        penalty1 = max(0.0, 1.0 - (len1 / self.min_length))
        penalty2 = max(0.0, 1.0 - (len2 / self.min_length))
        
        # Average penalty
        length_penalty = (penalty1 + penalty2) / 2.0

        
        return length_penalty
    


# if __name__ == "__main__":
#     # Example usage
#     smiles1, smiles2, er_list, tg_list = process_dual_monomer_data('Data/unique_smiles_Er.csv', 'Data/smiles.xlsx')
    
#     # Create model instance
#     novelty_model = NoveltyRewardModel()
#     # Set training pairs
#     novelty_model.set_training_pairs([smiles1, smiles2])
#     demo_pair = (smiles1[100], smiles2[0])
#     print(demo_pair)
#     # Example calculation (assuming demo_pair is defined)
#     demo_score = novelty_model.calculate_novelty_score(demo_pair)
#     print(demo_score)



