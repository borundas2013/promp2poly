from rdkit import Chem
from rdkit.Chem import Draw
import os
import pandas as pd

def two_smiles_to_svg(smiles1: str, smiles2: str, filename: str = "pair.svg"):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is None or mol2 is None:
        raise ValueError("One or both SMILES are invalid.")

    # Compute 2D coordinates
    Chem.rdDepictor.Compute2DCoords(mol1)
    Chem.rdDepictor.Compute2DCoords(mol2)

    # Create drawer for first molecule
    drawer = Draw.rdMolDraw2D.MolDraw2DSVG(512, 512)
    drawer.DrawMolecule(mol1)
    drawer.FinishDrawing()
    svg1 = drawer.GetDrawingText()

    # Write first SVG to file
    with open(filename, "w") as f:
        f.write(svg1)
    
    # Create drawer for second molecule
    drawer = Draw.rdMolDraw2D.MolDraw2DSVG(512, 512)
    drawer.DrawMolecule(mol2)
    drawer.FinishDrawing()
    svg2 = drawer.GetDrawingText()

    # Write second SVG to file
    with open(filename.replace(".svg", "_2.svg"), "w") as f:
        f.write(svg2)

# Example usage
# molecules = [
#     ['CCCC(COC3CO3)CC(C)(COC2CO2)C(=O)OCC4CO4', 'NCCOCCOCCN'],
#     ['O=C(OCC1CO1)C3CC2OC2CC3C(=O)OCC4CO4','NCc1cccc(CN)c1'],
#     ['C=CCn1c(=O)n(CC=C)\\c(=O)n(CC=C)c1=O','CCC(COC(=O)CCSC)(COC(=O)CCSC)COC(=O)CCSC'],
#     ['C=C(C)C(=O)OCCOCCOCCOCCOCCOCCOCCOCCOC(=O)C(C)=O','C=C(C)C(=O)OCCOCCOCCOCCOCCOC(=O)C(C)=O']
# ]

#molecules = [["CC(C)(C4CCC(OCC3CO3)CC4)C5CCC(OCC2CO2)CC5", "CC(N)COCC(C)OCC(C)OCC(C)OCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCC(C)OCC(C)OCC(C)N"]]
molecules =[["CCCCCCCCC1OC1CCCCCCCC(=O)OCC(COC(=O)CCCCCCCC2OC2CC3OC3CC4OC4CC)OC(=O)CCCCCCCC5OC5CC6OC6CC7OC7CC", "CCCCNCCCNCCCNCCCNCCCN(CCN)CCN"]]

data_path = 'analysis_results/high_tg_samples_new.csv'
df = pd.read_csv(data_path)
smiles = df[['smile1', 'smile2',  'model','tg_predicted','er_predicted']].values.tolist()

for i, (smiles1, smiles2, model, tg_predict, er_predict) in enumerate(smiles):
    two_smiles_to_svg(smiles1, smiles2, f"molecule_drawings/high_tg/pair_{model}_{i}_{tg_predict}_{er_predict}.svg")


# # Create output directory if it doesn't exist
# os.makedirs("molecule_drawings", exist_ok=True)

# # Generate SVGs for each pair
# for i, (smiles1, smiles2) in enumerate(molecules, 1):
#     two_smiles_to_svg(smiles1, smiles2, f"molecule_drawings/pair_m_mistral_new_{i}.svg")
