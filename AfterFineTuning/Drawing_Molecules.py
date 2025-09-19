from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import os
import pandas as pd
from Property_Prediction.predict import predict_property

def load_trained_data():
    try:
        # Read Excel file
        
        excel_path = os.path.join(os.path.dirname(__file__),  'Dataset', 'unique_smiles_Er.csv')
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




def save_molecule_svg(smiles, output_path):
    """
    Save molecule as SVG image from SMILES string
    
    Args:
        smiles (str): SMILES string of molecule
        output_path (str): Path to save SVG file
    """
    # Create molecule from SMILES
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is not None:
        # Generate 2D coordinates for the molecule
        AllChem.Compute2DCoords(mol)
        
        # Draw molecule and save as SVG
        d2d = Draw.MolDraw2DSVG(400, 400)
        d2d.DrawMolecule(mol)
        d2d.FinishDrawing()
        
        # Write SVG to file
        with open(output_path, 'w') as f:
            f.write(d2d.GetDrawingText())


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
        
        pd_df=df[df['Fixed Monomer 1']!='Not found']
    
        df=pd.DataFrame({'Fixed Monomer 1':monomer_1,'Fixed Monomer 2':monomer_2})
        pd_1=df
        df=df.drop_duplicates()
        unique_monomer_pairs = len(df)
        pd_df=pd_df.drop_duplicates()

        smiles1, smiles2, er, tg = load_trained_data()
        training_smiles_list = list(zip(smiles1, smiles2))
        print(f"First training pair: {training_smiles_list[0]}")
        for index,row in pd_df.iterrows():
            smiles1 = row['Fixed Monomer 1']
            smiles2 = row['Fixed Monomer 2']
            samples = (smiles1, smiles2)
            
            if samples in training_smiles_list:
                 pd_df.drop(index, inplace=True)
        
      
        unique_monomer_pairs_1 = len(pd_df)

        pd_df.to_csv('Unique_monomer_gpt.csv', index=False)
        
        print(f"Unique monomer pairs: {unique_monomer_pairs}")
        print(f"Total monomer pairs: {total_monomer_pairs}")
        print(f"Unique monomer pairs: {unique_monomer_pairs/total_monomer_pairs*100:.2f}%")
        print(f"Total monomer pairs: {total_monomer_pairs}")
        df.to_csv('Unique_monomer_pairs.csv', index=False)
        print(f"Valid df: {len(pd_1)}")
        print(f"Unique monomer pairs after removing training data: {unique_monomer_pairs_1}")
       
        # return df_no_duplicates
        
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        return None

if __name__ == "__main__":

    # # smiles1, smiles2, er, tg = load_trained_data()
    # # print(len(smiles1))
    # # print(len(smiles2))
    # # print(len(er))
    # # print(len(tg))

    # # remove_duplicate_monomer_pairs('GPT4/Output/Combined_gpt.csv')
    # #remove_duplicate_monomer_pairs('DeepSeek/Output/Combined_deepseek.csv')
    # #remove_duplicate_monomer_pairs('LLama32/Output/Combined_llama.csv')
 

    # # # smiles1 = "C=C(C)C(=O)OCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOC(=O)C(=C)C"
    # # # smiles2 = "CCC(CS)CC(CS)CC(CS)COC(=O)CC(=O)OCC(COCC(COCC(COCC1CO1)COCC2CO2)COCC3CO3)COC(=O)CC(=O)OCC(COCC(COCC4CO4)COCC5CO5)COCC6COC6"
    # # # output_path = "molecules/smiles1_gpt.svg"
    # # # output_path2 = "molecules/smiles2_gpt.svg"
    # # # save_molecule_svg(smiles1, output_path)
    # # # save_molecule_svg(smiles2, output_path2)

    # smiles1='CC(C/C=C\\CC(=O)OCC(O)COc3ccc(C(C)(C)c2ccc(OCC1CO1)cc2)cc3)C(=O)OCC(O)COc6ccc(C(C)(C)c5ccc(OCC4CO4)cc5)cc6'
    # smiles2='Nc2ccc(SSc1ccc(N)cc1)cc2'
    # # output_path1 = "molecules/smiles1_deepseek.svg"
    # # output_path2 = "molecules/smiles2_deepseek.svg"
    # # save_molecule_svg(smiles1, output_path1)
    # # save_molecule_svg(smiles2, output_path2)

    

    # #smiles1='C=C(C)C(=O)OCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOC(=O)C(=C)C'
    # #smiles2='C=CCOCC(CO)(COCC=C)COCC=C'
    # # output_path1 = "molecules/smiles1_llama.svg"
    # # output_path2 = "molecules/smiles2_llama.svg"
    # # save_molecule_svg(smiles1, output_path1)
    # # save_molecule_svg(smiles2, output_path2)

    # #smiles1='CC(C)(c2ccc(OCC1CO1)cc2)c4ccc(OCC3CO3)cc4'
    # #smiles2='CC(C)(C)N'
    # # output_path1 = "molecules/smiles1_gpt_fail.svg"
    # # output_path2 = "molecules/smiles2_gpt_fail.svg"
    # # save_molecule_svg(smiles1, output_path1)
    # # save_molecule_svg(smiles2, output_path2)

    # er,tg=predict_property(smiles1, smiles2)
    # print(f"Er: {er}, Tg: {tg}")

    smiles1='C=C(C)C(=O)OCCOCCOCCOCCOCCOCCOCCOCCOC(=O)C(=C)C'
    smiles2='C=C(C)C(=O)OCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOC(=O)C(=C)C'
    output_path1 = "molecules/smiles1_deepseek_ex4.svg"
    output_path2 = "molecules/smiles2_deepseek_ex4.svg"
    save_molecule_svg(smiles1, output_path1)
    save_molecule_svg(smiles2, output_path2)

  