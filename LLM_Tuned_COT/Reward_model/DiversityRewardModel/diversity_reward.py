# import tensorflow as tf
# import numpy as np
# from rdkit import Chem
# from rdkit.Chem import AllChem
# from rdkit import DataStructs
# import random

# class DiversityReward:
#     def __init__(self, training_data):
#         self.training_data = training_data
#         self.min_length = 20
        

#     def calculate_tanimoto_similarity(self, smiles1, smiles2, input_smiles1, input_smiles2):
#         """Calculate Tanimoto similarity between generated and input SMILES pairs"""
#         try:
#             # Convert SMILES to molecules
#             m1 = Chem.MolFromSmiles(smiles1)
#             m2 = Chem.MolFromSmiles(smiles2)
#             m1_input = Chem.MolFromSmiles(input_smiles1)
#             m2_input = Chem.MolFromSmiles(input_smiles2)
            
#             # Check for valid molecules
#             if not all([m1, m2, m1_input, m2_input]):
#                 return 0.0, 0.0
            
#             # Generate Morgan fingerprints
#             fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, 1024)
#             fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, 1024)
#             fp1_input = AllChem.GetMorganFingerprintAsBitVect(m1_input, 2, 1024)
#             fp2_input = AllChem.GetMorganFingerprintAsBitVect(m2_input, 2, 1024)
            
#             # Calculate similarities
#             sim1 = DataStructs.TanimotoSimilarity(fp1, fp1_input)
#             sim2 = DataStructs.TanimotoSimilarity(fp2, fp2_input)
            
#             return sim1, sim2
            
#         except Exception as e:
#             print(f"Error calculating similarity: {e}")
#             return 0.0, 0.0

#     def calculate_length_penalty(self, smiles1, smiles2):
#         """Calculate penalty for short SMILES strings"""
#         len1 = len(smiles1)
#         len2 = len(smiles2)
        
#         # Calculate how much shorter than min_length each SMILES is
#         penalty1 = max(0.0, 1.0 - (len1 / self.min_length))
#         penalty2 = max(0.0, 1.0 - (len2 / self.min_length))
        
#         # Average penalty
#         length_penalty = (penalty1 + penalty2) / 2.0
        
#         # print(f"SMILES lengths: {len1}, {len2}")
#         # print(f"Length penalty: {length_penalty}")
        
#         return length_penalty


#     def calculate_diversity_reward(self, generated_samples):
#         max_duplicate_score = 0.0
#         max_length_penalty = 0.0
#         for i in range(len(self.training_data[0])):
#             gen_smiles1 = generated_samples[0]
#             gen_smiles2 = generated_samples[1]
#             input_smiles1 = self.training_data[0][i]
#             input_smiles2 = self.training_data[1][i]
            
#             sim1, sim2 = self.calculate_tanimoto_similarity(gen_smiles1, gen_smiles2, input_smiles1, input_smiles2)
#             length_penalty = self.calculate_length_penalty(gen_smiles1, gen_smiles2)
#             max_duplicate_score = max(max_duplicate_score, (sim1 + sim2) / 2.0)
#             max_length_penalty = max(max_length_penalty, length_penalty)
        
#         diversity_score = 1.0 - max_duplicate_score
#         length_penalty_score = 1.0 - max_length_penalty
#         final_reward = diversity_score + length_penalty_score

#         return round(final_reward, 3)

#     def _print_reward_details(self, index, smiles1, smiles2, diversity_score,
#                             batch_diversity, length_penalty, base_reward, final_reward):
#         """Print detailed information about reward calculation"""
#         print(f"\nSample {index}:")
#         print(f"Generated SMILES: {smiles1}, {smiles2}")
#         print(f"Diversity score: {diversity_score:.3f}")
#         print(f"Batch diversity: {batch_diversity:.3f}")
#         print(f"Length penalty: {length_penalty:.3f}")
#         print(f"Base reward: {base_reward:.3f}")
#         print(f"Final reward: {final_reward:.3f}")
#         print("-" * 50)

# # Example usage:
# if __name__ == "__main__":
#     # Create reward calculator
#     reward_calculator = DiversityReward(
#         min_length=8,
#         diversity_weight=0.5,
#         batch_diversity_weight=0.3,
#         novelty_weight=0.2,
#         length_penalty_weight=0.5
#     )
    
#     # Test data
#     generated_samples = [
#         {'smiles1': "CCO", 'smiles2': "CCNC1OC1"},  # Short SMILES
#         {'smiles1': "CC1OC1CCCC", 'smiles2': "CCOCCNCC"}  # Longer SMILES
#     ]
    
#     input_data = [
#         {'smiles1': "CCOCC1OC1CCC1OC1C", 'smiles2': "CCOCCCNC1OC1CC1OC1CCC1OC1C"},
#         {'smiles1': "CC1OC1C", 'smiles2': "CCOC"}
#     ]
    
#     # Calculate rewards
#     rewards = reward_calculator.calculate_reward(generated_samples, input_data)
#     print("\nFinal rewards:", rewards)


