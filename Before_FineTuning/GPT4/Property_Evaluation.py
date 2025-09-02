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
#os.chdir("GPT4")



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
        
        #valid_df=valid_df.drop_duplicates()
        
        
       
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
        print("Valid df: ",len(valid_df))
        print("Number of group smiles: ",number_of_group_smiles)
       
        
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        return None

def count_functional_groups(smiles, smarts_pattern):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return len(mol.GetSubstructMatches(Chem.MolFromSmarts(smarts_pattern)))

vinyl_vinyl=0
epoxy_imini=0
vinyl_hydroxyl=0
vinyl_thiol=0
vinyl_acrylate=0
other=0


def check_reaction_consistency(smiles1, smiles2):
    global vinyl_vinyl, epoxy_imini, vinyl_hydroxyl, vinyl_thiol, vinyl_acrylate, other
    threshold=2
    if count_functional_groups(smiles1, 'C=C') >= threshold and count_functional_groups(smiles2, 'C=C') >= threshold:
        vinyl_vinyl+=1
        return True
    elif count_functional_groups(smiles1, 'C1OC1') >= threshold and count_functional_groups(smiles2, 'NC') >= threshold:
        epoxy_imini+=1
        return True
    elif count_functional_groups(smiles1, 'NC') >= threshold and count_functional_groups(smiles2, 'C1OC1') >= threshold:
        epoxy_imini+=1
        return True
    elif count_functional_groups(smiles1, 'CCS') >= threshold and count_functional_groups(smiles2, 'C=C') >= threshold:
        vinyl_thiol+=1
        return True
    elif count_functional_groups(smiles1, 'C=C') >= threshold and count_functional_groups(smiles2, 'CCS') >= threshold:
        vinyl_thiol+=1
        return True
    elif count_functional_groups(smiles1, 'C=C') >= threshold and count_functional_groups(smiles2, 'O') >= threshold:
        vinyl_hydroxyl+=1
        return True
    elif count_functional_groups(smiles1, 'O') >= threshold and count_functional_groups(smiles2, 'C=C') >= threshold:
        vinyl_hydroxyl+=1
        return True
    elif count_functional_groups(smiles1, 'C=C(C=O)') >= threshold and count_functional_groups(smiles2, 'C=C') >= threshold:
        vinyl_acrylate+=1
        return True
    elif count_functional_groups(smiles1, 'C=C') >= threshold and count_functional_groups(smiles2, 'C=C(C=O)') >= threshold:
        vinyl_acrylate+=1
        return True
    elif count_functional_groups(smiles1, 'C=O') >= threshold and count_functional_groups(smiles2, 'NC') >= threshold:  #vinyl_hydroxyl
        other+=1
        return True
    elif count_functional_groups(smiles1, 'NC') >= threshold and count_functional_groups(smiles2, 'C=O') >= threshold:  #vinyl_hydroxyl
        other+=1
        return True
    elif count_functional_groups(smiles1, 'C(=O)O') >= threshold and count_functional_groups(smiles2, 'C1OC1') >= threshold:
        other+=1
        return True
    elif count_functional_groups(smiles1, 'C1OC1') >= threshold and count_functional_groups(smiles2, 'C(=O)O') >= threshold:
        other+=1
        return True
    elif count_functional_groups(smiles1, 'CS') >= threshold and count_functional_groups(smiles2, 'C1OC1') >= threshold:
        other+=1
        return True
    elif count_functional_groups(smiles1, 'C1OC1') >= threshold and count_functional_groups(smiles2, 'CS') >= threshold:
        other+=1
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
     




#read_multiple_csv_with_validation(csv_files)
remove_duplicate_monomer_pairs("Output/Zeroshot/Combined_zeroshot.csv")
print("Vinyl-vinyl: ",vinyl_vinyl)
print("Epoxy-imine: ",epoxy_imini)
print("Vinyl-hydroxyl: ",vinyl_hydroxyl)
print("Vinyl-thiol: ",vinyl_thiol)
print("Vinyl-acrylate: ",vinyl_acrylate)
print("Other: ",other)


