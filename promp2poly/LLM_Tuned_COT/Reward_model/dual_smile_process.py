import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
sys.path.append(str(parent_dir))
import Reward_model.Constants as Constants
import pandas as pd
import numpy as np
import tensorflow as tf
from rdkit import Chem
from Reward_model.Data_Process_with_prevocab import *
import random

def process_dual_monomer_data(excel_path, excel_path2):

    try:
        # Read Excel file
        df = pd.read_csv(excel_path)
        #df = df.sample(frac=1).reset_index(drop=True)
        #df2 = pd.read_excel(excel_path2)
        #df2 = df2.sample(frac=1).reset_index(drop=True)

        # #Check if required columns exist
        # required_cols = ['Smiles', 'Er', 'Tg']
        # for col in required_cols:
        #     if col not in df.columns:
        #         raise ValueError(f"Required column '{col}' not found in Excel file")
        
        # Initialize lists for storing data
        smiles1_list = []
        smiles2_list = []
        er_list = []
        tg_list = []
        
        # Process each row
        for _, row in df.iterrows():
            try:
                # Extract the two SMILES from the SMILES column
                smiles_pair = eval(row['Smiles'])  # Safely evaluate string representation of list
                if len(smiles_pair) == 2:
                    smiles1, smiles2 = smiles_pair[0], smiles_pair[1]
                    smiles1_list.append(smiles1)
                    smiles2_list.append(smiles2)
                    er_list.append(row['Er'])
                    tg_list.append(row['Tg'])
            except:
                print(f"Skipping malformed SMILES pair: {row['SMILES']}")
                continue

        # for _, row in df2.iterrows():
        #     try:
        #         smiles_pair = row['SMILES'].split(',')
        #         if len(smiles_pair) == 2:
        #             smiles1, smiles2 = smiles_pair[0], smiles_pair[1]
        #             smiles1_list.append(smiles1)
        #             smiles2_list.append(smiles2)
        #             er_list.append(None)
        #             tg_list.append(None)
        #     except:
        #         print(f"Skipping malformed SMILES pair: {row['SMILES']}")
        #         continue
                

        return smiles1_list, smiles2_list, er_list, tg_list
        
    except Exception as e:
        print(f"Error processing Excel file: {str(e)}")
        raise
def count_functional_groups(smiles, smarts_pattern):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return len(mol.GetSubstructMatches(Chem.MolFromSmarts(smarts_pattern)))
def encode_functional_groups(monomer1_list, monomer2_list):
    # SMARTS patterns for different functional groups
   
    all_groups = []
    
    for m1, m2 in zip(monomer1_list, monomer2_list):
        found_groups_m1 = []
        found_groups_m2 = []
        
        # Check for each group in monomer 1
        if count_functional_groups(m1, Constants.EPOXY_SMARTS) >= 2:
            found_groups_m1.append("C1OC1")
        if count_functional_groups(m1, Constants.IMINE_SMARTS) >= 2:
            found_groups_m1.append("NC")
        if count_functional_groups(m1, Constants.THIOL_SMARTS) >= 2:
            found_groups_m1.append("CCS")
        if count_functional_groups(m1, Constants.ACRYL_SMARTS) >= 2:
            found_groups_m1.append("C=C(C=O)")
        if count_functional_groups(m1, Constants.VINYL_SMARTS) >= 2:
            found_groups_m1.append("C=C")
        if not (count_functional_groups(m1, Constants.EPOXY_SMARTS) >= 2 or 
                count_functional_groups(m1, Constants.IMINE_SMARTS) >= 2 or
                count_functional_groups(m1, Constants.THIOL_SMARTS) >= 2 or 
                count_functional_groups(m1, Constants.ACRYL_SMARTS) >= 2 or
                count_functional_groups(m1, Constants.VINYL_SMARTS) >= 2):
            found_groups_m1.append("No group")
        # Check for each group in monomer 2
        if count_functional_groups(m2, Constants.EPOXY_SMARTS) >= 2:
            found_groups_m2.append("C1OC1")
        if count_functional_groups(m2, Constants.IMINE_SMARTS) >= 2:
            found_groups_m2.append("NC")
        if count_functional_groups(m2, Constants.THIOL_SMARTS) >= 2:
            found_groups_m2.append("CCS")
        if count_functional_groups(m2, Constants.ACRYL_SMARTS) >= 2:
            found_groups_m2.append("C=C(C=O)")
        if count_functional_groups(m2, Constants.VINYL_SMARTS) >= 2:
            found_groups_m2.append("C=C")
        if not (count_functional_groups(m2, Constants.EPOXY_SMARTS) >= 2 or 
                count_functional_groups(m2, Constants.IMINE_SMARTS) >= 2 or
                count_functional_groups(m2, Constants.THIOL_SMARTS) >= 2 or 
                count_functional_groups(m2, Constants.ACRYL_SMARTS) >= 2 or
                count_functional_groups(m2, Constants.VINYL_SMARTS) >= 2):
            found_groups_m2.append("No group")

        
        # Combine groups from both monomers
        combined_groups = found_groups_m1 + found_groups_m2
        if not combined_groups:
            combined_groups.append('No group')
        
        all_groups.append(combined_groups)
    
    # Encode groups using the vocabulary
    encoded_groups = [encode_groups(groups, Constants.GROUP_VOCAB) for groups in all_groups]
    
    return encoded_groups

def add_gaussian_noise(tokens, noise_level=1):
    """Add Gaussian noise to token embeddings"""
    noise = np.random.normal(0, noise_level, tokens.shape)
    noisy_tokens = tokens + noise
    return noisy_tokens

def add_dropout_noise(tokens, dropout_rate=0.1):
    """Randomly zero out some tokens"""
    mask = np.random.binomial(1, 1-dropout_rate, tokens.shape)
    return tokens * mask

def add_swap_noise(tokens, swap_rate=0.1):
    """Randomly swap adjacent tokens"""
    noisy_tokens = tokens.copy()
    for i in range(len(tokens)):
        for j in range(1, len(tokens[i])-1):  # Avoid swapping start/end tokens
            # if np.random.random() < swap_rate:
            #     noisy_tokens[i][j], noisy_tokens[i][j+1] = \
            #     noisy_tokens[i][j+1], noisy_tokens[i][j]
            noisy_tokens[i][j], noisy_tokens[i][j+1] = noisy_tokens[i][j+1], noisy_tokens[i][j]
    return noisy_tokens

def add_mask_noise(tokens, vocab, mask_rate=0.1):
    """Randomly mask tokens with MASK token"""
    mask_token = vocab.get('[MASK]', len(vocab)-1)  # Use last token if no mask token
    noisy_tokens = tokens.copy()
    mask = np.random.random(tokens.shape) < mask_rate
    noisy_tokens[mask] = mask_token
    return noisy_tokens

def reaction_valid_samples(smiles1,smiles2,er_list,tg_list):
 
    valid_reaction = []
    invalid_reaction = []
    for i in range(len(smiles1)):
        reaction_valid = filter_valid_groups(smiles1[i], smiles2[i])
        if reaction_valid:
            valid_reaction.append([smiles1[i],smiles2[i],er_list[i],tg_list[i]])
        else:
            invalid_reaction.append([smiles1[i],smiles2[i],er_list[i],tg_list[i]])

    print(len(valid_reaction))
    print(len(invalid_reaction))
    random_invalid_reaction = random.sample(invalid_reaction, 259)
    valid_reaction.extend(random_invalid_reaction) 
    print(len(valid_reaction))
    return valid_reaction

def check_reaction_validity_with_invalid_groups(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    pairs = [
        (Constants.VINYL_SMARTS, Constants.THIOL_SMARTS, ['C=C', 'CCS']),
        (Constants.THIOL_SMARTS, Constants.VINYL_SMARTS, ['CCS', 'C=C']),
        (Constants.VINYL_SMARTS, Constants.ACRYL_SMARTS, ['C=C', 'C=C(C=O)']),
        (Constants.ACRYL_SMARTS, Constants.VINYL_SMARTS, ['C=C(C=O)', 'C=C']),
        (Constants.EPOXY_SMARTS, Constants.IMINE_SMARTS, ['C1OC1', 'NC']),
        (Constants.IMINE_SMARTS, Constants.EPOXY_SMARTS, ['NC', 'C1OC1']),
        (Constants.VINYL_SMARTS, Constants.VINYL_SMARTS, ['C=C', 'C=C']),
        
    ]
    labels = ["No_group","No_group"]
    total_count = 0
    found = False
    for smarts1, smarts2, labels in pairs:
        count1 = count_functional_groups(smiles1, smarts1)
        count2 = count_functional_groups(smiles2, smarts2)
        total = count1 + count2
        if count1 >= 2 and count2 >= 2:
            labels[0] = smarts1
            labels[1] = smarts2
            total_count = total
            found = True
            break
        elif count1 > 0 and count2 > 0:
            labels[0] = smarts1
            labels[1] = smarts2
            total_count = total
            found = True
            break
        elif count1 > 0 and count2 == 0:
            labels[0] = smarts1
            labels[1] = "No_group"
            total_count = count1
            found = True
            break
        elif count1 == 0 and count2 > 0:
            labels[0] = "No_group"
            labels[1] = smarts2
            total_count = count2
            found = True
            break
        else:
            labels[0] = "No_group"
            labels[1] = "No_group"
            total_count = 0
            found = False
        
        
    
    if found:
        return labels, total_count
    else:
        return ["No_group", "No_group"], 0
    
def filter_valid_groups(smiles1, smiles2):
    pairs = [
        (Constants.VINYL_SMARTS, Constants.THIOL_SMARTS, ['C=C', 'CCS']),
        (Constants.THIOL_SMARTS, Constants.VINYL_SMARTS, ['CCS', 'C=C']),
        (Constants.VINYL_SMARTS, Constants.ACRYL_SMARTS, ['C=C', 'C=C(C=O)']),
        (Constants.ACRYL_SMARTS, Constants.VINYL_SMARTS, ['C=C(C=O)', 'C=C']),
        (Constants.EPOXY_SMARTS, Constants.IMINE_SMARTS, ['C1OC1', 'NC']),
        (Constants.IMINE_SMARTS, Constants.EPOXY_SMARTS, ['NC', 'C1OC1']),
        (Constants.VINYL_SMARTS, Constants.VINYL_SMARTS, ['C=C', 'C=C']),
        
    ]
    for smarts1, smarts2, labels in pairs:
        count1 = count_functional_groups(smiles1, smarts1)
        count2 = count_functional_groups(smiles2, smarts2)
        if count1 >= 2 and count2 >= 2:
            return True
       
        else:
            return False


def check_reaction_validity(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return False,[]
    if count_functional_groups(smiles1, Constants.EPOXY_SMARTS) >= 2 and count_functional_groups(smiles2, Constants.IMINE_SMARTS) >= 2:
        return True,['C1OC1','NC']
    if count_functional_groups(smiles1, Constants.IMINE_SMARTS) >= 2 and count_functional_groups(smiles2, Constants.EPOXY_SMARTS) >= 2:
        return True,['NC','C1OC1']
    if count_functional_groups(smiles1, Constants.VINYL_SMARTS) >= 2 and count_functional_groups(smiles2, Constants.THIOL_SMARTS) >= 2:
        return True,['C=C','CCS']
    if count_functional_groups(smiles1, Constants.THIOL_SMARTS) >= 2 and count_functional_groups(smiles2, Constants.VINYL_SMARTS) >= 2:
        return True,['CCS','C=C']
    if count_functional_groups(smiles1, Constants.VINYL_SMARTS) >= 2 and count_functional_groups(smiles2, Constants.ACRYL_SMARTS) >= 2:
        return True,['C=C','C=C(C=O)']
    if count_functional_groups(smiles1, Constants.ACRYL_SMARTS) >= 2 and count_functional_groups(smiles2, Constants.VINYL_SMARTS) >= 2:
        return True,['C=C(C=O)','C=C']  
    
    return False,[]


def check_reaction_validity_with_Fixed_groups(smiles1, smiles2, group1, group2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return False,0,0
    if group1 ==None and group2 == None:
        reactivity,groups = check_reaction_validity(smiles1, smiles2)
        if reactivity:
            group1_count = count_functional_groups(smiles1, groups[0])  
            group2_count = count_functional_groups(smiles2, groups[1])
            return True,group1_count,group2_count
        else:
            return False,0,0
    else:
        if group1 == "C1OC1" and group2 == "NC":
            if count_functional_groups(smiles1, Constants.EPOXY_SMARTS) >= 2 and count_functional_groups(smiles2, Constants.IMINE_SMARTS) >= 2:
                return True,count_functional_groups(smiles1, Constants.EPOXY_SMARTS) , count_functional_groups(smiles2, Constants.IMINE_SMARTS)
        if group1 == "NC" and group2 == "C1OC1":
            if count_functional_groups(smiles1, Constants.IMINE_SMARTS) >= 2 and count_functional_groups(smiles2, Constants.EPOXY_SMARTS) >= 2:
                return True,count_functional_groups(smiles1, Constants.IMINE_SMARTS) , count_functional_groups(smiles2, Constants.EPOXY_SMARTS)
        if group1 == "C=C" and group2 == "CCS":
            if count_functional_groups(smiles1, Constants.VINYL_SMARTS) >= 2 and count_functional_groups(smiles2, Constants.THIOL_SMARTS) >= 2:
                return True,count_functional_groups(smiles1, Constants.VINYL_SMARTS) , count_functional_groups(smiles2, Constants.THIOL_SMARTS)
        if group1 == "CCS" and group2 == "C=C":
            if count_functional_groups(smiles1, Constants.THIOL_SMARTS) >= 2 and count_functional_groups(smiles2, Constants.VINYL_SMARTS) >= 2:
                return True,count_functional_groups(smiles1, Constants.THIOL_SMARTS) , count_functional_groups(smiles2, Constants.VINYL_SMARTS)
        if group1 == "C=C" and group2 == "C=C(C=O)":
            if count_functional_groups(smiles1, Constants.VINYL_SMARTS) >= 2 and count_functional_groups(smiles2, Constants.ACRYL_SMARTS) >= 2:
                return True,count_functional_groups(smiles1, Constants.VINYL_SMARTS) , count_functional_groups(smiles2, Constants.ACRYL_SMARTS)
        if group1 == "C=C(C=O)" and group2 == "C=C":
            if count_functional_groups(smiles1, Constants.ACRYL_SMARTS) >= 2 and count_functional_groups(smiles2, Constants.VINYL_SMARTS) >= 2:
                return True,count_functional_groups(smiles1, Constants.ACRYL_SMARTS) , count_functional_groups(smiles2, Constants.VINYL_SMARTS)
    return False,0,0


