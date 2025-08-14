import json
import os
import os



from rdkit import Chem
from rdkit.Chem import AllChem
import csv
import pandas as pd
import pandas as pd
from rdkit import Chem
import sys
import os
# Add the parent directory to Python path to access Data_util
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
os.chdir("GPT4")



def hasEpoxyGroup(smile):
    mol = Chem.MolFromSmiles(smile)
    substructure = Chem.MolFromSmarts('C1OC1')
    matches = []
    return mol.HasSubstructMatch(substructure) if mol is not None else False

def has_imine(smiles):
    imine_pattern_1 = Chem.MolFromSmarts('NC')
    imine_pattern_2 = Chem.MolFromSmarts('Nc')
    mol = Chem.MolFromSmiles(smiles)
  
    return mol.HasSubstructMatch(imine_pattern_1) or mol.HasSubstructMatch(imine_pattern_2) if mol is not None else False


def has_vinyl_group(smiles):
    vinyl_pattern = Chem.MolFromSmarts('C=C')
    mol = Chem.MolFromSmiles(smiles)

    return mol.HasSubstructMatch(vinyl_pattern) if mol is not None else False


def has_thiol_group(smiles):
    thiol_substructure = Chem.MolFromSmarts('CCS')
    mol = Chem.MolFromSmiles(smiles)
    return mol.HasSubstructMatch(thiol_substructure) if mol is not None else False


def has_acrylate_group(smiles):
    mol = Chem.MolFromSmiles(smiles)
    acrylate_substructure = Chem.MolFromSmarts('C=C(C=O)')
    return mol.HasSubstructMatch(acrylate_substructure) if mol is not None else False

def has_hydroxyl_group(smiles):
    hydroxyl_substructure = Chem.MolFromSmarts('O')
    mol = Chem.MolFromSmiles(smiles)
    return mol.HasSubstructMatch(hydroxyl_substructure) if mol is not None else False

def remove_duplicate_monomer_pairs(csv_filepath, output_csv=None):
    
    try:
        if not os.path.exists(csv_filepath):
            raise FileNotFoundError(f"CSV file not found: {csv_filepath}")
            
        # Generate output CSV filename if not provided
        if output_csv is None:
            base_name = os.path.splitext(os.path.basename(csv_filepath))[0]
            output_csv = f"{base_name}_no_duplicates.csv"
            
        print(f"Reading CSV file: {csv_filepath}")
        print("=" * 60)
        
        # Try different encodings
        try:
            df = pd.read_csv(csv_filepath, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(csv_filepath, encoding='latin-1')
            except UnicodeDecodeError:
                df = pd.read_csv(csv_filepath, encoding='cp1252')
        
        print(f"Original file: {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns found: {list(df.columns)}")
        
        # Check if required columns exist
        required_columns = ['Fixed Monomer 1', 'Fixed Monomer 2']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Warning: Required columns not found: {missing_columns}")
            print("Available columns:")
            for col in df.columns:
                print(f"  - {col}")
            return None
        
        
        valid_df= df[(df['Fixed Monomer 1']!='Not found') & (df['Fixed Monomer 2']!='Not found')]
       
        #df=pd.DataFrame({'Fixed Monomer 1':monomer_1,'Fixed Monomer 2':monomer_2})
        
        valid_df=valid_df.drop_duplicates()
        
        
       
        number_of_group_smiles=0
        group1_match=0
        reaction_match=0
        for index, row in valid_df.iterrows() :
            if check_reaction_consistency(row['Fixed Monomer 1'], row['Fixed Monomer 2']):
                    reaction_match+=1
            if pd.isna(row['Group1']) and pd.isna(row['Group2']):
                continue
            else:
                number_of_group_smiles+=1
                group1=row['Group1']
                group2=row['Group2']

                if check_group_consistency(group1, group2, row['Fixed Monomer 1'], row['Fixed Monomer 2']):
                   group1_match+=1
                

        print(f"Count of group1: {group1_match}")
        print(f"Count of group1: {group1_match/number_of_group_smiles*100:.2f}%")
        print(f"Count of reaction: {reaction_match}")
        print(f"Count of reaction: {reaction_match/len(valid_df)*100:.2f}%")
       
        
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        return None

def count_functional_groups(smiles, smarts_pattern):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return len(mol.GetSubstructMatches(Chem.MolFromSmarts(smarts_pattern)))

def check_reaction_consistency(smiles1, smiles2):
    threshold=2
    if count_functional_groups(smiles1, 'C=C') >= threshold and count_functional_groups(smiles2, 'C=C') >= threshold:
        return True
    elif count_functional_groups(smiles1, 'C1OC1') >= threshold and count_functional_groups(smiles2, 'NC') >= threshold:
        return True
    elif count_functional_groups(smiles1, 'NC') >= threshold and count_functional_groups(smiles2, 'C1OC1') >= threshold:
        return True
    elif count_functional_groups(smiles1, 'CCS') >= threshold and count_functional_groups(smiles2, 'C=C') >= threshold:
        return True
    elif count_functional_groups(smiles1, 'C=C') >= threshold and count_functional_groups(smiles2, 'CCS') >= threshold:
        return True
    elif count_functional_groups(smiles1, 'C=C') >= threshold and count_functional_groups(smiles2, 'O') >= threshold:
        return True
    elif count_functional_groups(smiles1, 'O') >= threshold and count_functional_groups(smiles2, 'C=C') >= threshold:
        return True
    elif count_functional_groups(smiles1, 'C=C(C=O)') >= threshold and count_functional_groups(smiles2, 'C=C') >= threshold:
        return True
    elif count_functional_groups(smiles1, 'C=C') >= threshold and count_functional_groups(smiles2, 'C=C(C=O)') >= threshold:
        return True
    elif count_functional_groups(smiles1, 'C=O') >= threshold and count_functional_groups(smiles2, 'NC') >= threshold:
        return True
    elif count_functional_groups(smiles1, 'NC') >= threshold and count_functional_groups(smiles2, 'C=O') >= threshold:
        return True
    elif count_functional_groups(smiles1, 'C(=O)O') >= threshold and count_functional_groups(smiles2, 'C1OC1') >= threshold:
        return True
    elif count_functional_groups(smiles1, 'C1OC1') >= threshold and count_functional_groups(smiles2, 'C(=O)O') >= threshold:
        return True
    elif count_functional_groups(smiles1, 'CS') >= threshold and count_functional_groups(smiles2, 'C1OC1') >= threshold:
        return True
    elif count_functional_groups(smiles1, 'C1OC1') >= threshold and count_functional_groups(smiles2, 'CS') >= threshold:
        return True
    else:
        return False



def check_group_consistency(group1, group2, smiles1, smiles2):
    group1_present=False
    group2_present=False
    if 'vinyl'in group1:
        group1_present=has_vinyl_group(smiles1)
    elif 'epoxy'in group1:
        group1_present=hasEpoxyGroup(smiles1)
    elif 'hydroxyl'in group1:
        group1_present=has_hydroxyl_group(smiles1)
    elif 'thiol'in group1:
        group1_present=has_thiol_group(smiles1)
    elif 'acrylate'in group1:
        group1_present=has_acrylate_group(smiles1)
    elif 'imine'in group1:
        group1_present=has_imine(smiles1)
    else:
        group1_present=False
    
    if 'vinyl'in group2:
        group2_present=has_vinyl_group(smiles2)
    elif 'epoxy'in group2:
        group2_present=hasEpoxyGroup(smiles2)
    elif 'hydroxyl'in group2:
        group2_present=has_hydroxyl_group(smiles2)
    elif 'thiol'in group2:
        group2_present=has_thiol_group(smiles2)
    elif 'acrylate'in group2:
        group2_present=has_acrylate_group(smiles2)
    elif 'imine'in group2:
        group2_present=has_imine(smiles2)
    else:
        group2_present=False
    
    return group1_present and group2_present
     


def load_dataset_gpt4():
    try:
        # Read Excel file
        
        excel_path = os.path.join(os.path.dirname(__file__), '..', 'Dataset', 'unique_smiles_Er.csv')
        df = pd.read_csv(excel_path)
    

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

        return smiles1_list, smiles2_list, er_list, tg_list
    except Exception as e:
        print(f"Error processing Excel file: {str(e)}")
        raise




csv_files = [
    "Output/generation_results_gpt4o_mini_group_u1C.csv",
  "Output/generation_results_gpt4o_mini_mix_u1C.csv",
  "Output/generation_results_gpt4o_mini_property_u1C.csv"
]


#read_multiple_csv_with_validation(csv_files)
remove_duplicate_monomer_pairs("Output/Combined_both.csv")


