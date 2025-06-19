from rdkit import Chem

def remove_bond_by_smarts(smiles: str, bond_smarts: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid SMILES"

    pattern = Chem.MolFromSmarts(bond_smarts)
    if pattern is None:
        return "Invalid SMARTS pattern"

    matches = mol.GetSubstructMatches(pattern)
    if not matches:
        return "No matching bonds found."

    rw_mol = Chem.RWMol(mol)
    atoms_to_remove = set()
    for match in matches:
        atoms_to_remove.update(match)

    for idx in sorted(atoms_to_remove, reverse=True):
        if idx < rw_mol.GetNumAtoms():
            rw_mol.RemoveAtom(idx)

    cleaned_mol = rw_mol.GetMol()

    # Extract all fragments safely (avoid sanitization crash)
    frags = Chem.GetMolFrags(cleaned_mol, asMols=True, sanitizeFrags=False)

    smiles_list = []
    for frag in frags:
        try:
            Chem.SanitizeMol(frag)
            smiles_list.append(Chem.MolToSmiles(frag))
        except:
            # Skip invalid fragments (e.g., broken aromatics)
            continue

    if not smiles_list:
        return "No valid fragments remain."

    return "".join(smiles_list)

from rdkit import Chem
from rdkit.Chem import AllChem

def add_group_by_attachment_smarts(base_smiles: str, group_smarts: str, attachment_atom_idx: int = 0) -> str:
    """
    Add a functional group or substructure defined by SMARTS to a base SMILES molecule.

    Parameters:
    - base_smiles: The original molecule as SMILES.
    - group_smarts: The group to add, defined in SMARTS (must contain [*] as attachment point).
    - attachment_atom_idx: Atom index in base molecule where the group will attach (default = 0).

    Returns:
    - New SMILES with the group attached, or error message.
    """
    mol = Chem.MolFromSmiles(base_smiles)
    if mol is None:
        return "Invalid base SMILES"

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
        return Chem.MolToSmiles(rw_combo)
    except:
        return "Failed to sanitize combined molecule"
    

from rdkit import Chem
from rdkit.Chem import AllChem

from rdkit import Chem
from rdkit.Chem import AllChem

def connect_molecules_by_bond(smiles1: str, smiles2: str, 
                            atom_idx1: int = 0, atom_idx2: int = 0,
                            bond_order: str = "single") -> str:
    """
    Connect two molecules by a specified bond type at selected atom indices.

    Parameters:
    - smiles1, smiles2: SMILES strings of the molecules to connect
    - atom_idx1: Atom index in molecule 1
    - atom_idx2: Atom index in molecule 2 (relative to molecule 2)
    - bond_order: "single", "double", or "triple"

    Returns:
    - Combined SMILES or error
    """
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is None or mol2 is None:
        return "Invalid input SMILES"

    # Validate atom indices
    if atom_idx1 >= mol1.GetNumAtoms() or atom_idx2 >= mol2.GetNumAtoms():
        return "Invalid atom index"

    # Check valence states before connecting
    atom1 = mol1.GetAtomWithIdx(atom_idx1)
    atom2 = mol2.GetAtomWithIdx(atom_idx2)
    
    # Calculate current valence
    valence1 = sum(bond.GetBondTypeAsDouble() for bond in atom1.GetBonds())
    valence2 = sum(bond.GetBondTypeAsDouble() for bond in atom2.GetBonds())
    
    # Choose bond type and get its value
    bond_types = {
        "single": (Chem.BondType.SINGLE, 1.0),
        "double": (Chem.BondType.DOUBLE, 2.0),
        "triple": (Chem.BondType.TRIPLE, 3.0)
    }

    if bond_order not in bond_types:
        return "Unsupported bond type: choose 'single', 'double', or 'triple'"

    bond_type, bond_value = bond_types[bond_order]
    
    # Check if new bond would exceed maximum valence
    max_valence = {
        'C': 4, 'N': 3, 'O': 2, 'S': 6, 'P': 5, 'F': 1, 'Cl': 1, 'Br': 1, 'I': 1
    }
    
    if (valence1 + bond_value > max_valence.get(atom1.GetSymbol(), 4) or 
        valence2 + bond_value > max_valence.get(atom2.GetSymbol(), 4)):
        return f"Cannot create {bond_order} bond: would exceed maximum valence"

    combined = Chem.CombineMols(mol1, mol2)
    rw_mol = Chem.RWMol(combined)

    offset = mol1.GetNumAtoms()
    new_atom_idx2 = atom_idx2 + offset

    rw_mol.AddBond(atom_idx1, new_atom_idx2, bond_type)

    try:
        Chem.SanitizeMol(rw_mol)
        return Chem.MolToSmiles(rw_mol)
    except Exception as e:
        return f"Failed to sanitize combined molecule: {e}"



# Test cases
print("Remove bond test:")
print(remove_bond_by_smarts("CCCOOCC","O[O]"))
print(remove_bond_by_smarts("CCNC1OC1Cc1ccccc1CCCCBr", "COC"))

# Add carboxylic acid (-COOH) to benzene
print("\nAdd group test:")
print(add_group_by_attachment_smarts("c1ccccc1", "C(=O)O[*]"))  

# Connect molecules test
print("\nConnect molecules test:")
# # Connect ethanol and amine (valid)
# print(connect_molecules_by_bond("CCO", "N", 2, 0, bond_order="single"))

# # Connect ethene and carbon monoxide (valid)
# print(connect_molecules_by_bond("C=C", "C=O", 1, 0, bond_order="double"))

# # Connect alkyne and nitrile (valid)
# print(connect_molecules_by_bond("C#C", "C#N", 1, 0, bond_order="triple"))

# # Invalid valence example (will return error)
# print(connect_molecules_by_bond("C", "C", 0, 0, bond_order="triple"))