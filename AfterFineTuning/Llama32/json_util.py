import json
import os
import os


#os.chdir("Llama32")
from rdkit import Chem
from rdkit.Chem import AllChem
import csv
import pandas as pd
import pandas as pd
from rdkit import Chem
import re
os.chdir("Llama32")

def extract_monomers(text: str):
    # If the input has literal "\n", convert to real newlines
    if "\\n" in text:
        text = text.replace("\\n", "\n")

    # Monomer 1: capture up to the next "Monomer 2:" or end of string
    m1 = re.search(r"Monomer\s*1\s*:\s*(.+?)(?=\s*(?:Monomer\s*2\s*:|$))",
                   text, flags=re.IGNORECASE | re.DOTALL)
    # Monomer 2: capture to end of line
    m2 = re.search(r"Monomer\s*2\s*:\s*([^\r\n\"']+)",
                   text, flags=re.IGNORECASE)

    monomer1 = m1.group(1).strip() if m1 else None
    monomer2 = m2.group(1).strip() if m2 else None
    return monomer1, monomer2

def save_monomers_to_csv(filepath, output_csv="extracted_monomers_gpt4o_mini_group1.csv"):
    """
    Extract monomer SMILES and save to CSV file
    
    Args:
        filepath (str): Path to the JSON file (with or without .json extension)
        output_csv (str): Name of the output CSV file
    """
    try:
        if not filepath.endswith('.json'):
            filepath += '.json'
            
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Prepare CSV data
        csv_data = []
        
        if isinstance(data, list):
            print(f"\nExtracting monomer SMILES from {len(data)} objects and saving to {output_csv}")
            
            for i, obj in enumerate(data, 1):
                output_text = obj.get('output', '')
                group1 = obj.get('Group1', 'N/A')
                group2 = obj.get('Group2', 'N/A')
                temperature = obj.get('temperature', 'N/A')
                tg = obj.get('Tg', 'N/A')
                er = obj.get('Er', 'N/A')
                prompt = obj.get('prompt_data', 'N/A')
                
                # Extract SMILES using common patterns
                monomer1, monomer2 = None, None #extract_monomers(output_text)
                
                
                # Add to CSV data
                csv_data.append({
                    'SL': i,
                    'Monomer 1': monomer1 if monomer1 else 'N/A',
                    'Monomer 2': monomer2 if monomer2 else 'N/A',
                    'Output_text': output_text,
                    'Group1': group1,
                    'Group2': group2,
                    'Temperature': temperature,
                    'Tg': tg,
                    'Er': er,
                    'Prompt': prompt
                })
        
        # Write to CSV
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['SL', 'Monomer 1', 'Monomer 2','Output_text','Group1','Group2','Temperature','Tg','Er','Prompt']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in csv_data:
                writer.writerow(row)
        
        print(f"\nSuccessfully saved {len(csv_data)} monomer pairs to {output_csv}")
                
    except Exception as e:
        print(f"Error saving monomers to CSV: {e}")
        raise

# Test the new method

def fix_smiles_parsing_issues(smiles):
    """
    Fix common SMILES parsing issues including invalid characters, unmatched parentheses, 
    and malformed ring structures.
    
    Args:
        smiles (str): Input SMILES string that may have parsing issues
        
    Returns:
        str: Fixed SMILES string, or original if no fixes possible
    """
    import re
    
    if not smiles or smiles in ['nan', 'Not found', 'N/A', '']:
        return smiles
    
    original_smiles = smiles
    fixed_smiles = smiles
    
    try:
        # 1. Remove invalid characters that are not part of SMILES syntax
        invalid_chars = ['?', '!',  '$', '%', '^', '&', '*', '+', '=', '|', '<', '>', '~','K','0','t','b']
        for char in invalid_chars:
            fixed_smiles = fixed_smiles.replace(char, '')
        
        # 2. Fix empty parentheses - remove them completely
        fixed_smiles = re.sub(r'\(\)', '', fixed_smiles)
        
        # 3. Fix consecutive parentheses
        fixed_smiles = re.sub(r'\(+', '(', fixed_smiles)
        fixed_smiles = re.sub(r'\)+', ')', fixed_smiles)
        
        # 4. Fix unmatched parentheses
        open_parens = fixed_smiles.count('(')
        close_parens = fixed_smiles.count(')')
        
        if open_parens > close_parens:
            # Add missing close parentheses
            fixed_smiles += ')' * (open_parens - close_parens)
        elif close_parens > open_parens:
            # Remove extra close parentheses from the end
            extra_close = close_parens - open_parens
            for _ in range(extra_close):
                if fixed_smiles.endswith(')'):
                    fixed_smiles = fixed_smiles[:-1]
        
        # 3. Fix ring closure issues
        # Find all ring numbers
        ring_numbers = re.findall(r'(\d+)', fixed_smiles)
        ring_counts = {}
        for num in ring_numbers:
            ring_counts[num] = ring_counts.get(num, 0) + 1
        
        # Remove dangling ring numbers (appear only once)
        for num, count in ring_counts.items():
            if count == 1:
                fixed_smiles = re.sub(rf'{num}', '', fixed_smiles, count=1)
        
        # 4. Fix common SMILES syntax issues
        # Remove consecutive dots (should be single dot)
        fixed_smiles = re.sub(r'\.+', '.', fixed_smiles)
        
        # Fix common atom notation issues
        fixed_smiles = re.sub(r'([A-Z])([A-Z])', r'\1\2', fixed_smiles)  # Ensure proper atom notation
        
        # 5. Try RDKit validation and canonicalization
        try:
            mol = Chem.MolFromSmiles(fixed_smiles)
            if mol is not None:
                canonical_smiles = Chem.MolToSmiles(mol)
                if canonical_smiles and canonical_smiles != fixed_smiles:
                    fixed_smiles = canonical_smiles
        except:
            pass
        
        # 6. Final validation - if still invalid, try more aggressive fixes
        try:
            mol = Chem.MolFromSmiles(fixed_smiles)
            if mol is None:
                # Try removing problematic patterns
                # Remove any remaining invalid characters
                fixed_smiles = re.sub(r'[^A-Za-z0-9()=#@+-\[\]\.]', '', fixed_smiles)
                
                # Try again
                mol = Chem.MolFromSmiles(fixed_smiles)
                if mol is not None:
                    fixed_smiles = Chem.MolToSmiles(mol)
        except:
            pass
        
        # If all fixes fail, return original
        if fixed_smiles == original_smiles or not fixed_smiles:
            return original_smiles
            
        return fixed_smiles
        
    except Exception as e:
        # If any error occurs during fixing, return original
        return original_smiles


def detect_and_fix_dangling_rings(smiles):

    import re
    
    # Find all ring closure numbers
    ring_numbers = re.findall(r'(\d+)', smiles)
    
    # Count occurrences of each ring number
    ring_counts = {}
    for num in ring_numbers:
        ring_counts[num] = ring_counts.get(num, 0) + 1
    
    # Find issues
    issues = []
    for num, count in ring_counts.items():
        if count == 1:
            issues.append(f"Ring {num} appears only once (dangling)")
        elif count > 2:
            issues.append(f"Ring {num} appears {count} times (invalid)")
    
    # Fix dangling rings by removing the ring closure numbers
    fixed_smiles = smiles
    for num, count in ring_counts.items():
        if count == 1:
            # Remove the specific dangling ring number
            fixed_smiles = re.sub(rf'{num}', '', fixed_smiles, count=1)
            issues.append(f"Removed dangling ring {num}")
    
    # Try RDKit canonicalization
    try:
        mol = Chem.MolFromSmiles(fixed_smiles)
        if mol is not None:
            rdkit_fixed = Chem.MolToSmiles(mol)
            if rdkit_fixed != fixed_smiles:
                fixed_smiles = rdkit_fixed
                issues.append("RDKit canonicalization applied")
        else:
            issues.append("RDKit could not process fixed SMILES")
    except:
        issues.append("RDKit processing failed")
    
    return fixed_smiles


def read_multiple_csv_with_validation(csv_filepaths):
    """
    Read multiple CSV files and validate monomer columns using pandas and RDKit.
    Returns validation statistics for each file and overall.
    """
    
    
    try:
        all_results = []
        total_valid_count = 0
        total_count = 0
        total_not_valid_count = 0
        total_without_SMILES_count = 0
        
        print(f"\nProcessing {len(csv_filepaths)} CSV files...")
        print("=" * 80)
        
        for file_idx, csv_filepath in enumerate(csv_filepaths, 1):
            print(f"\n--- File {file_idx}: {csv_filepath} ---")
            
            if not os.path.exists(csv_filepath):
                print(f"CSV file not found: {csv_filepath}")
                continue
            
            # Read CSV with pandas - try different encodings
            try:
                df = pd.read_csv(csv_filepath, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(csv_filepath, encoding='latin-1')
                except UnicodeDecodeError:
                    df = pd.read_csv(csv_filepath, encoding='cp1252')
            # Create fixed monomer columns
            df['Fixed Monomer 1'] = df['Monomer 1'].copy()
            df['Fixed Monomer 2'] = df['Monomer 2'].copy()
            
            # Validate monomers
            valid_count = 0
            not_valid_count = 0
            without_SMILES_count = 0
            total_rows = len(df)
            
            for idx, row in df.iterrows():
                monomer1 = str(row['Monomer 1']).strip()
                monomer2 = str(row['Monomer 2']).strip()
                
                
                # Skip if monomers are empty or invalid
                if monomer1 in ['nan', 'Not found', 'N/A', ''] or monomer2 in ['nan', 'Not found', 'N/A', '']:
                    df['Fixed Monomer 1'][idx] = 'Not found'
                    df['Fixed Monomer 2'][idx] = 'Not found'
                    not_valid_count += 1
                    without_SMILES_count += 1
                    continue
                
                # Try to create MOL files first
                mol1 = None
                mol2 = None
                
                try:
                    mol1 = Chem.MolFromSmiles(monomer1)
                except:
                    pass
                    
                try:
                    mol2 = Chem.MolFromSmiles(monomer2)
                except:
                    pass
                
                # Only apply fixes if parsing failed
                if mol1 is None:
                    monomer1 = fix_smiles_parsing_issues(monomer1)
                    monomer1 = detect_and_fix_dangling_rings(monomer1)
                    try:
                        mol1 = Chem.MolFromSmiles(monomer1)
                    except:
                        pass
                
                if mol2 is None:
                    monomer2 = fix_smiles_parsing_issues(monomer2)
                    monomer2 = detect_and_fix_dangling_rings(monomer2)
                    try:
                        mol2 = Chem.MolFromSmiles(monomer2)
                    except:
                        pass
                
                # Count as valid if both monomers are valid
                if mol1 is not None and  mol2 is not None:
                    df['Fixed Monomer 1'][idx] = monomer1
                    df['Fixed Monomer 2'][idx] = monomer2
                    valid_count += 1
                else:
                    df['Fixed Monomer 1'][idx] = 'Not found'
                    df['Fixed Monomer 2'][idx] = 'Not found'
                    not_valid_count += 1
            
            total_count += total_rows
            total_valid_count += valid_count
            total_not_valid_count += not_valid_count
            total_without_SMILES_count += without_SMILES_count
            
            # Store file results
            success_rate = (valid_count / total_rows) * 100 if total_rows > 0 else 0
            all_results.append({
                'file': csv_filepath,
                'valid_count': valid_count,
                'not_valid_count': not_valid_count,
                'total_count': total_rows,
                'success_rate': success_rate
            })
            
            print(f"Valid pairs: {valid_count}/{total_rows} ({success_rate:.1f}%)")
            print(f"Not found pairs: {not_valid_count}")
            print(f"Valid pairs (excluding 'Not found'): {valid_count}/{valid_count + not_valid_count} ({(valid_count / (valid_count + not_valid_count)) * 100:.1f}%)")
            #df.to_csv("Output/"+os.path.basename(csv_filepath).replace(".csv", "_fixed.csv"), index=False)
        
        # Overall summary
        print("\n" + "=" * 80)
        print("OVERALL SUMMARY")
        print("=" * 80)
        print(f"Total pairs: {total_count}")
        print(f"Total not valid pairs: {total_not_valid_count}")
        print(f"Total valid pairs: {total_valid_count}")
        print("Response with SMILES: ", total_count-total_without_SMILES_count)
        print(f"Total without SMILES pairs: {total_without_SMILES_count}")
        overall_success_rate = (total_valid_count / total_count) * 100 if total_count > 0 else 0
        print(f"Overall success rate: {overall_success_rate:.1f}%")

         
        
        # Save the fixed dataframe for the last processed file

           
        
        return {
            'file_results': all_results,
            'total_valid': total_valid_count,
            'total_count': total_count,
            'overall_success_rate': overall_success_rate
        }
                
    except Exception as e:
        print(f"Error processing CSV files: {e}")
        raise

# Test the new method with multiple CSV files


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
        total_monomer_pairs = len(df)
        monomer_1= df[df['Fixed Monomer 1']!='Not found']['Fixed Monomer 1'].tolist()
        monomer_2= df[df['Fixed Monomer 2']!='Not found']['Fixed Monomer 2'].tolist()
        df=pd.DataFrame({'Fixed Monomer 1':monomer_1,'Fixed Monomer 2':monomer_2})
        pd_1=df
        df=df.drop_duplicates()
        unique_monomer_pairs = len(df)
        
        
        print(f"Unique monomer pairs: {unique_monomer_pairs}")
        print(f"Total monomer pairs: {total_monomer_pairs}")
        print(f"Unique monomer pairs: {unique_monomer_pairs/total_monomer_pairs*100:.2f}%")
        print(f"Total monomer pairs: {total_monomer_pairs}")
        df.to_csv('Unique_monomer_pairs.csv', index=False)
        print(f"Valid df: {len(pd_1)}")
       
        # return df_no_duplicates
        
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        return None


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




#save_monomers_to_csv("Output/generation_results_lama32_group_2.json","Output/generation_results_lama32_group_2.csv")
#save_monomers_to_csv("Output/generation_results_lama32_property.json","Output/generation_results_lama32_property.csv")
#save_monomers_to_csv("Output/generation_results_lama32_mix.json","Output/generation_results_lama32_mix.csv")

# save_monomers_to_csv("Output/Fewshot/generation_results_lama32_group_fewshot.json","Output/Fewshot/generation_results_lama32_group_fewshot.csv")
# save_monomers_to_csv("Output/Fewshot/generation_results_lama32_property_fewshot.json","Output/Fewshot/generation_results_lama32_property_fewshot.csv")
# save_monomers_to_csv("Output/Fewshot/generation_results_lama32_mix_fewshot.json","Output/Fewshot/generation_results_lama32_mix_fewshot.csv")


# csv_files = [
#     "Output/old/extracted_monomers_gpt4o_mini_mix.csv",
#   "Output/old/extracted_monomers_gpt4o_mini_group.csv",
#   "Output/old/extracted_monomers_gpt4o_mini_property.csv"
# ]
csv_files = [
    "Output/generation_results_lama32_group_2.csv",
   "Output/generation_results_lama32_mix.csv",
 "Output/generation_results_lama32_property.csv"
]


#read_multiple_csv_with_validation(csv_files)
remove_duplicate_monomer_pairs("Output/Combined_Llama.csv")




# # Example
# txt = 'Monomer 1: CC(C)(c2ccc(OCCOCCOCCOCC1CO1)cc2)c4ccc(OCCOCCOCCOCC3CO3)cc4\\nMonomer 2: Nc2ccc(S(=O)(=O)c1ccc(N)cc1)cc2'
# txt="Based on your requirements, I suggest the following monomers:\nMonomer 1: C=C(C)C(=O)OCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOC(=O)C(=C)C\nMonomer 2: C=C(C)C(=O)OCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOC(=O)C(=C)C"
# print(extract_monomers(txt))














