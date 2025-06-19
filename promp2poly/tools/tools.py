from rdkit import Chem
from Prediction_property.TgEr.predict import predict_property
from Prediction_property.Solubility.solubility_prediction import predict_solubility
from Prediction_property.Toxicity.predict_toxicity import predict_toxicity_for_smiles_pair


def remove_bond_by_smarts(smiles1: str, smiles2: str, bond_smarts: str, target_monomer: str = "1") -> str:
    # Select which molecule to modify
    target_smiles = smiles1 if target_monomer == "1" else smiles2
    other_smiles = smiles2 if target_monomer == "1" else smiles1
    

    # Convert SMILES to molecule
    mol = Chem.MolFromSmiles(target_smiles)
    if mol is None:
        return f"Invalid SMILES for monomer {target_monomer}"

    # Convert SMARTS pattern to molecule
    pattern = Chem.MolFromSmarts(bond_smarts)
    if pattern is None:
        return "Invalid SMARTS pattern"

    # Find where the pattern matches in the molecule
    matches = mol.GetSubstructMatches(pattern)
    if not matches:
        return f"Sorry, the group/bond '{bond_smarts}' is not found in monomer {target_monomer}"

    # Create editable molecule
    rw_mol = Chem.RWMol(mol)
    
    # Get all atoms in the pattern
    pattern_atoms = set()
    for match in matches:
        pattern_atoms.update(match)

    # Remove bonds first
    bonds_to_remove = []
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if begin_idx in pattern_atoms or end_idx in pattern_atoms:
            bonds_to_remove.append((begin_idx, end_idx))

    # Remove bonds in reverse order to maintain indices
    for begin_idx, end_idx in sorted(bonds_to_remove, reverse=True):
        rw_mol.RemoveBond(begin_idx, end_idx)

    # Now remove the atoms
    for idx in sorted(pattern_atoms, reverse=True):
        if idx < rw_mol.GetNumAtoms():
            rw_mol.RemoveAtom(idx)

    cleaned_mol = rw_mol.GetMol()

    # Get the remaining fragments
    frags = Chem.GetMolFrags(cleaned_mol, asMols=True, sanitizeFrags=False)

    smiles_list = []
    for frag in frags:
        try:
            Chem.SanitizeMol(frag)
            smiles_list.append(Chem.MolToSmiles(frag))
        except:
            continue

    if not smiles_list:
        return f"Sorry, after removing '{bond_smarts}', no valid fragments remain in monomer {target_monomer}"

    # Format output in the requested style
    modified_monomer = "".join(smiles_list)
    if target_monomer == "1":
        return f"Here is the revised output: \n -- monomer1 = {modified_monomer} \n -- monomer2 = {other_smiles}"
    else:
        return f"Here is the revised output: \n -- monomer1 = {other_smiles} \n -- monomer2 = {modified_monomer}"
    

def add_group_by_smarts(smiles1: str, smiles2: str, group_smarts: str, target_monomer: str = "1", attachment_atom_idx: int = 0) -> str:
   
    # Select which molecule to modify
    target_smiles = smiles1 if target_monomer == "1" else smiles2
    other_smiles = smiles2 if target_monomer == "1" else smiles1

    # Convert SMILES to molecule
    mol = Chem.MolFromSmiles(target_smiles)
    if mol is None:
        return f"Invalid SMILES for monomer {target_monomer}"

    # Make editable molecule
    rw_mol = Chem.RWMol(mol)

    # Parse group with attachment point [*]
    group = Chem.MolFromSmarts(group_smarts)
    if group is None:
        return "Invalid group SMARTS"

    # Find attachment atom (the [*] atom) in group
    attachment_points = [atom.GetIdx() for atom in group.GetAtoms() if atom.GetSymbol() == '*']
    if not attachment_points:
        return "Group SMARTS must contain [*] as attachment point"
    group_attachment_idx = attachment_points[0]

    # Remove [*] atom and get neighbors
    rw_group = Chem.RWMol(group)
    group_neighbor = list(rw_group.GetAtomWithIdx(group_attachment_idx).GetNeighbors())[0]
    rw_group.RemoveAtom(group_attachment_idx)
    group = rw_group.GetMol()

    # Combine both molecules
    combo = Chem.CombineMols(rw_mol, group)
    rw_combo = Chem.RWMol(combo)

    # Calculate new atom index after combining
    offset = mol.GetNumAtoms()
    new_atom_idx = group_neighbor.GetIdx() + offset

    # Add bond between attachment site and new group
    rw_combo.AddBond(attachment_atom_idx, new_atom_idx, Chem.BondType.SINGLE)

    # Sanitize and return
    try:
        Chem.SanitizeMol(rw_combo)
        modified_smiles = Chem.MolToSmiles(rw_combo)
        if target_monomer == "1":
            return f"Here is the revised output: \n -- monomer1 = {modified_smiles} \n -- monomer2 = {other_smiles}"
        else:
            return f"Here is the revised output: \n -- monomer1 = {other_smiles} \n -- monomer2 = {modified_smiles}"
    except:
        return "Failed to sanitize combined molecule"
    
def get_property_for_all(smiles1: str, smiles2: str, ratio_1: float, ratio_2: float) -> dict:
    scores = predict_property(smiles1, smiles2, ratio_1, ratio_2)
    solubility, solubility_logs = predict_solubility(smiles1, smiles2)

    Tg = scores["tg_score"] 
    Er = scores["er_score"]
    solubility = solubility['average_hydration_free_energy']
    solubility_esol = solubility_logs['average_logS']
    solubility_esol_solubility = solubility_logs['solubility']
    toxicity_result = predict_toxicity_for_smiles_pair(smiles1, smiles2)
    return Tg, Er, solubility, solubility_esol, solubility_esol_solubility, toxicity_result

def get_property_for_toxicity(smiles1: str, smiles2: str) -> dict:
    toxicity_result = predict_toxicity_for_smiles_pair(smiles1, smiles2)
    return toxicity_result

def get_property_for_physical(smiles1: str, smiles2: str, ratio_1: float, ratio_2: float) -> dict:
    scores = predict_property(smiles1, smiles2, ratio_1, ratio_2)
    Tg = scores["tg_score"] 
    Er = scores["er_score"]
    return Tg, Er

def get_property_for_solubility(smiles1: str, smiles2: str) -> dict:
    solubility, solubility_logs = predict_solubility(smiles1, smiles2)
    solubility_esol = solubility_logs['average_logS']
    solubility_esol_solubility = solubility_logs['solubility']
    return solubility['average_hydration_free_energy'], solubility_esol, solubility_esol_solubility


def get_all_properties(smiles1: str, smiles2: str, ratio_1: float, ratio_2: float, property_type: str) -> dict:
    print("property_type:", property_type)

    if property_type.lower() == "all":
        Tg, Er, solubility, solubility_esol, solubility_esol_solubility, toxicity_result = get_property_for_all(smiles1, smiles2, ratio_1, ratio_2)
        response = f"""**Comprehensive Property Analysis**

**Thermal and Mechanical Properties:**
• **Glass Transition Temperature (Tg):** {Tg:.2f} °C  
  _Temperature where the polymer transitions from rigid to flexible state._
• **Recovery Stress (Er):** {Er:.2f} MPa  
  _Material's shape recovery capability under stress._

**Solubility Assessment:**
• **ESOL Solubility (logS):**
  - LogS Value: {solubility_esol:.2f} (log mol/L)
  - Classification: {solubility_esol_solubility}
  _Higher logS values indicate better water solubility._

• **Hydration Energy Model:**
  - Value: {solubility:.2f} kcal/mol
  _Complementary measure of water interaction tendency._

**Toxicity Profile:**  
{toxicity_result['table']}

**Safety Summary:**  
This monomer combination shows {toxicity_result['summary']['overall_assessment']} with {toxicity_result['summary']['high_risk_count']} high-risk endpoints out of 12."""

    elif property_type.lower() == "toxicity":
        toxicity_result = get_property_for_toxicity(smiles1, smiles2)
        response = f"""**Toxicity Assessment**

{toxicity_result['table']}

**Understanding the Results:**  
Each endpoint is classified based on toxicity probability:
- High Risk: ≥ 0.7 probability
- Moderate Risk: 0.5-0.7 probability
- Low Risk: < 0.5 probability

**Overall Assessment:** {toxicity_result['summary']['overall_assessment']}
Number of High-Risk Endpoints: {toxicity_result['summary']['high_risk_count']} out of 12"""    

    elif property_type.lower() == "physical":
        Tg, Er = get_property_for_physical(smiles1, smiles2, ratio_1, ratio_2)
        response = f"""**Physical Properties Assessment**

**Thermal Properties:**
• **Glass Transition Temperature (Tg):** {Tg:.2f} °C  
  _Key temperature where polymer changes from glass-like to rubber-like._

**Mechanical Properties:**
• **Recovery Stress (Er):** {Er:.2f} MPa  
  _Indicates shape memory and recovery potential._
"""

    elif property_type.lower() == "solubility":
        solubility, solubility_esol, solubility_esol_solubility = get_property_for_solubility(smiles1, smiles2)
        print("solubility_esol_solubility:", solubility_esol_solubility)
        print("solubility_esol:", solubility_esol)
        print("solubility:", solubility)
        response = f"""**Solubility Analysis**

**Primary Solubility Measure (ESOL Model):**
• **LogS Value:** {solubility_esol:.2f} log mol/L
• **Interpretation:** {solubility_esol_solubility}
  _LogS is the standard measure of water solubility:_
  - Values > 0: Very soluble
  - Values -1 to 0: Soluble
  - Values -2 to -1: Moderately soluble
  - Values -3 to -2: Slightly soluble
  - Values < -3: Poorly soluble

**Supporting Measure:**
• **Hydration Energy:** {solubility:.2f} kcal/mol
  _Provides additional insight into water interaction potential._"""
    
    return response
