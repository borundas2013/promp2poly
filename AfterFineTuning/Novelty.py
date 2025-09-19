from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import os
import pandas as pd
from Property_Prediction.predict import predict_property

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np
from collections import Counter

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
# =========================
# Generative evaluation utils (RDKit)
# - Internal diversity (distributional)
# - Training-set overlap (fine-tuned only)
# =========================

from functools import lru_cache
from typing import List, Dict, Tuple, Optional
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

# ---------- Canonicalization & fingerprints (cached) ----------

@lru_cache(maxsize=200_000)
def canon_cached(smiles: str) -> Optional[str]:
    """Return canonical SMILES or None if invalid."""
    if smiles is None:
        return None
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m, canonical=True) if m is not None else None

@lru_cache(maxsize=200_000)
def morgan_fp_cached(canonical_smiles: str,
                     radius: int = 2,
                     nBits: int = 2048,
                     useChirality: bool = False) -> Optional[DataStructs.ExplicitBitVect]:
    """Morgan fingerprint from canonical SMILES (cached)."""
    if canonical_smiles is None:
        return None
    m = Chem.MolFromSmiles(canonical_smiles)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits, useChirality=useChirality)

def order_invariant_pair(ca: str, cb: str) -> Tuple[str, str]:
    """Sort two canonical SMILES for order-invariant pair representation."""
    return (ca, cb) if ca <= cb else (cb, ca)

def pair_key_and_canon(smiA: str, smiB: str) -> Optional[Tuple[str, str]]:
    """Order-invariant canonical pair key or None if invalid."""
    ca, cb = canon_cached(smiA), canon_cached(smiB)
    if (ca is None) or (cb is None):
        return None
    return order_invariant_pair(ca, cb)

def pair_fp_from_canonical(ca: str, cb: str,
                           radius: int = 2,
                           nBits: int = 2048,
                           useChirality: bool = False) -> Optional[DataStructs.ExplicitBitVect]:
    """Order-invariant OR-combined fingerprint for a canonical pair."""
    fa = morgan_fp_cached(ca, radius, nBits, useChirality)
    fb = morgan_fp_cached(cb, radius, nBits, useChirality)
    if (fa is None) or (fb is None):
        return None
    # Efficient OR-combine into a new bit vector
    bv = DataStructs.ExplicitBitVect(nBits)
    # Use bit positions (fast, avoids numpy/string roundtrip)
    for i in fa.GetOnBits():
        bv.SetBit(i)
    for i in fb.GetOnBits():
        bv.SetBit(i)
    return bv

# ---------- Internal diversity / uniqueness (distributional) ----------

def evaluate_internal(generated_pairs: List[Dict[str, str]],
                      radius: int = 2,
                      nBits: int = 2048,
                      useChirality: bool = False,
                      max_pairs_for_pairwise: Optional[int] = None) -> Dict[str, float]:
    """
    Evaluate internal diversity without referencing the training set.
    - Counts total, valid, unique
    - Uniqueness %, duplicate fraction
    - Pairwise Tanimoto mean/median over unique pairs
    - Nearest-neighbor similarity mean/median over unique pairs
    """
    # Canonicalize and collect valid order-invariant keys
    keys = []
    for d in generated_pairs:
        k = pair_key_and_canon(d.get("m1"), d.get("m2"))
        if k is not None:
            keys.append(k)

    n_total = len(generated_pairs)
    n_valid = len(keys)
    validity_pct = 100.0 * n_valid / max(1, n_total)

    unique_list = list(set(keys))
    n_unique = len(unique_list)
    uniqueness_pct = 100.0 * n_unique / max(1, n_valid)
    duplicate_frac = 1.0 - (uniqueness_pct / 100.0)

    # Optional subsampling for very large sets to keep O(n^2) reasonable
    if (max_pairs_for_pairwise is not None) and (n_unique > max_pairs_for_pairwise):
        rng = np.random.default_rng(17)
        unique_list = list(rng.choice(unique_list, size=max_pairs_for_pairwise, replace=False))
        n_unique = len(unique_list)

    # Build pair fingerprints for unique pairs
    fps = []
    for a, b in unique_list:
        fp = pair_fp_from_canonical(a, b, radius=radius, nBits=nBits, useChirality=useChirality)
        if fp is not None:
            fps.append(fp)

    n = len(fps)
    sims, nn_sims = [], []
    if n > 1:
        # Pairwise + nearest-neighbor similarities
        for i in range(n):
            fi = fps[i]
            # Bulk similarity to speed up inner loop
            bulk = DataStructs.BulkTanimotoSimilarity(fi, fps)
            # Remove self-similarity
            if len(bulk) > i:
                bulk[i] = 0.0
            best = max(bulk) if bulk else 0.0
            nn_sims.append(best)
            # Upper triangle accumulation
            for j in range(i+1, n):
                sims.append(bulk[j])

    mean_sim   = float(np.mean(sims))   if sims else float('nan')
    median_sim = float(np.median(sims)) if sims else float('nan')
    diversity  = 1.0 - mean_sim if sims else float('nan')
    mean_nn    = float(np.mean(nn_sims))   if nn_sims else float('nan')
    median_nn  = float(np.median(nn_sims)) if nn_sims else float('nan')

    return {
        "n_total": n_total,
        "n_valid_pairs": n_valid,
        "validity_pct": validity_pct,
        "n_unique_pairs": n_unique,
        "uniqueness_pct": uniqueness_pct,
        "duplicate_frac": duplicate_frac,
        "mean_pairwise_sim": mean_sim,
        "median_pairwise_sim": median_sim,
        "diversity": diversity,  # = 1 - mean_pairwise_sim
        "mean_NN_sim": mean_nn,
        "median_NN_sim": median_nn,
        "nn_sims": nn_sims,
        "pairwise_sims": sims,
    }

def plot_nn_similarity_box(nn_sims_base,
                           nn_sims_ft,
                           save_path=None):
    b = np.asarray(nn_sims_base, dtype=float)
    f = np.asarray(nn_sims_ft, dtype=float)
    b = b[~np.isnan(b)]
    f = f[~np.isnan(f)]
    b = np.clip(b, 0.0, 1.0)
    f = np.clip(f, 0.0, 1.0)

    plt.figure(figsize=(6.2,4.2))
    bp = plt.boxplot([b, f],
                     tick_labels=["Base", "Fine-tuned"],
                     showfliers=False,
                     widths=0.6,
                     patch_artist=True,
                     boxprops=dict(facecolor='#2E86AB', alpha=0.8),
                     medianprops=dict(color='#A23B72', linewidth=2),
                     whiskerprops=dict(color='#F18F01', linewidth=2),
                     capprops=dict(color='#F18F01', linewidth=2))
    
    plt.ylabel("Nearest-neighbor Tanimoto similarity", fontsize=14, fontweight='bold', color='#2C3E50')
    plt.title("Local clustering (nearest-neighbor similarity)", fontsize=16, fontweight='bold', color='#2C3E50')
    plt.xticks(fontsize=12, fontweight='bold', color='#2C3E50')
    plt.yticks(fontsize=12, fontweight='bold', color='#2C3E50')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches="tight")
    plt.show()

# ---------- Training-set overlap (fine-tuned only) ----------

def _unique_canonical_keys(pairs: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    out = []
    for d in pairs:
        k = pair_key_and_canon(d.get("m1"), d.get("m2"))
        if k is not None:
            out.append(k)
    return list(set(out))

def training_overlap(generated_pairs: List[Dict[str, str]],
                     training_pairs: List[Dict[str, str]],
                     radius: int = 2,
                     nBits: int = 2048,
                     useChirality: bool = False,
                     thresholds: Tuple[float, ...] = (0.8, 0.9),
                     unique_only: bool = True) -> Dict[str, float]:
    """
    Compute overlap of generated pairs with training set.
    - If unique_only=True (recommended), deduplicate generated pairs first.
    - Returns mean/median of max similarity to training and fractions over thresholds.
    """
    # if unique_only:
    #     gen_keys = _unique_canonical_keys(generated_pairs)
    # else:
    #     # keep all (including duplicates)
    gen_keys = []
    for d in generated_pairs:
        k = pair_key_and_canon(d.get("m1"), d.get("m2"))
        if k is not None:
            gen_keys.append(k)

    train_keys = _unique_canonical_keys(training_pairs)

    gen_fps, train_fps = [], []
    for a, b in gen_keys:
        fp = pair_fp_from_canonical(a, b, radius=radius, nBits=nBits, useChirality=useChirality)
        if fp is not None:
            gen_fps.append(fp)
    for a, b in train_keys:
        fp = pair_fp_from_canonical(a, b, radius=radius, nBits=nBits, useChirality=useChirality)
        if fp is not None:
            train_fps.append(fp)

    if not gen_fps or not train_fps:
        out = {"mean_max_sim": float('nan'), "median_max_sim": float('nan')}
        for thr in thresholds:
            out[f"frac_maxSim_ge_{thr:.2f}"] = float('nan')
        return out

    max_sims = []
    for g in gen_fps:
        sims = DataStructs.BulkTanimotoSimilarity(g, train_fps)
        max_sims.append(max(sims) if sims else 0.0)

    max_sims = np.asarray(max_sims, dtype=float)
    out = {
        "mean_max_sim": float(np.mean(max_sims)),
        "median_max_sim": float(np.median(max_sims)),
    }
    for thr in thresholds:
        out[f"frac_maxSim_ge_{thr:.2f}"] = float(np.mean(max_sims >= thr))
    return out




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

        df_n=pd.DataFrame({'Smiles1':smiles1_list,'Smiles2':smiles2_list,'Er':er_list,'Tg':tg_list})
  

        return df_n
    except Exception as e:
        print(f"Error processing Excel file: {str(e)}")
        raise

def read_csv_file(csv_filepath):
    try:
        df = pd.read_csv(csv_filepath, encoding='utf-8')
        return df
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return None

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
        for index,row in df.iterrows():
            smiles1 = row['Fixed Monomer 1']
            smiles2 = row['Fixed Monomer 2']
            samples = (smiles1, smiles2)
            
            if samples in training_smiles_list:
                 df.drop(index, inplace=True)
        
      
        unique_monomer_pairs_1 = len(df)

        df.to_csv('Unique_monomer_gpt_zeroshot.csv', index=False)
        
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
    
import numpy as np
import matplotlib.pyplot as plt

def plot_pairwise_similarity_hist(pairwise_sims_base,
                                  pairwise_sims_ft,
                                  bins=30,
                                  save_path=None):
    """
    Plot histograms of pairwise similarity distributions
    for base vs fine-tuned model outputs.

    Parameters
    ----------
    pairwise_sims_base : list or array
        Pairwise Tanimoto similarities for base model.
    pairwise_sims_ft : list or array
        Pairwise Tanimoto similarities for fine-tuned model.
    bins : int
        Number of histogram bins (default=30).
    save_path : str or None
        If provided, save the figure to this path.
    """
    # convert to numpy, drop NaNs, clip to [0,1]
    b = np.asarray(pairwise_sims_base, dtype=float)
    f = np.asarray(pairwise_sims_ft, dtype=float)
    b = b[~np.isnan(b)]
    f = f[~np.isnan(f)]
    b = np.clip(b, 0.0, 1.0)
    f = np.clip(f, 0.0, 1.0)

    plt.figure(figsize=(7,4.2))
    # density overlay histogram with deep colors
    plt.hist(b, bins=bins, range=(0,1), density=True, alpha=0.7, 
             label="Base", color='#2E86AB', edgecolor='#1A5F7A', linewidth=1.2)
    plt.hist(f, bins=bins, range=(0,1), density=True, alpha=0.7, 
             label="Fine-tuned", color='#A23B72', edgecolor='#6B2C5A', linewidth=1.2)

    # mean markers with deep colors
    if b.size:
        plt.axvline(b.mean(), linestyle="--", linewidth=2.5, color='#F18F01', 
                   label=f"Base mean = {b.mean():.2f}")
    if f.size:
        plt.axvline(f.mean(), linestyle=":", linewidth=2.5, color='#C73E1D', 
                   label=f"Fine-tuned mean = {f.mean():.2f}")

    plt.xlabel("Pairwise Tanimoto similarity", fontsize=14, fontweight='bold', color='#2C3E50')
    plt.ylabel("Density", fontsize=14, fontweight='bold', color='#2C3E50')
    plt.title("Global structural diversity (pairwise similarity)", fontsize=16, fontweight='bold', color='#2C3E50')
    plt.xticks(fontsize=12, fontweight='bold', color='#2C3E50')
    plt.yticks([0, 2, 4, 6, 8], fontsize=12, fontweight='bold', color='#2C3E50')
    plt.ylim(0, 8)
    plt.legend(fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches="tight")
    plt.show()


def plot_combined_similarity_analysis(nn_sims_base, nn_sims_ft, pairwise_sims_base, pairwise_sims_ft, save_path=None):
    """
    Plot both nearest-neighbor similarity boxplot and pairwise similarity histogram
    side by side in a single figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Nearest-neighbor similarity boxplot
    b_nn = np.asarray(nn_sims_base, dtype=float)
    f_nn = np.asarray(nn_sims_ft, dtype=float)
    b_nn = b_nn[~np.isnan(b_nn)]
    f_nn = f_nn[~np.isnan(f_nn)]
    b_nn = np.clip(b_nn, 0.0, 1.0)
    f_nn = np.clip(f_nn, 0.0, 1.0)
    
    bp = ax1.boxplot([b_nn, f_nn],
                     tick_labels=["Base", "Fine-tuned"],
                     showfliers=False,
                     widths=0.6,
                     patch_artist=True,
                     boxprops=dict(facecolor='#2E86AB', alpha=0.8),
                     medianprops=dict(color='#A23B72', linewidth=2),
                     whiskerprops=dict(color='#F18F01', linewidth=2),
                     capprops=dict(color='#F18F01', linewidth=2))
    
    ax1.set_ylabel("Nearest-neighbor Tanimoto similarity", fontsize=14, fontweight='bold', color='#2C3E50')
    ax1.set_title("(A) Local clustering (nearest-neighbor similarity)", fontsize=16, fontweight='bold', color='#2C3E50')
    ax1.tick_params(axis='both', labelsize=12, colors='#2C3E50')
    for label in ax1.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax1.get_yticklabels():
        label.set_fontweight('bold')
    
    # Right plot: Pairwise similarity histogram
    b_pair = np.asarray(pairwise_sims_base, dtype=float)
    f_pair = np.asarray(pairwise_sims_ft, dtype=float)
    b_pair = b_pair[~np.isnan(b_pair)]
    f_pair = f_pair[~np.isnan(f_pair)]
    b_pair = np.clip(b_pair, 0.0, 1.0)
    f_pair = np.clip(f_pair, 0.0, 1.0)
    
    ax2.hist(b_pair, bins=30, range=(0,1), density=True, alpha=0.7, 
             label="Base", color='#2E86AB', edgecolor='#1A5F7A', linewidth=1.2)
    ax2.hist(f_pair, bins=30, range=(0,1), density=True, alpha=0.7, 
             label="Fine-tuned", color='#A23B72', edgecolor='#6B2C5A', linewidth=1.2)
    
    # mean markers
    if b_pair.size:
        ax2.axvline(b_pair.mean(), linestyle="--", linewidth=2.5, color='#F18F01', 
                   label=f"Base mean = {b_pair.mean():.2f}")
    if f_pair.size:
        ax2.axvline(f_pair.mean(), linestyle=":", linewidth=2.5, color='#C73E1D', 
                   label=f"Fine-tuned mean = {f_pair.mean():.2f}")
    
    ax2.set_xlabel("Pairwise Tanimoto similarity", fontsize=14, fontweight='bold', color='#2C3E50')
    ax2.set_ylabel("Density", fontsize=14, fontweight='bold', color='#2C3E50')
    ax2.set_title("(B) Global structural diversity (pairwise similarity)", fontsize=16, fontweight='bold', color='#2C3E50')
    ax2.tick_params(axis='both', labelsize=12, colors='#2C3E50')
    ax2.set_yticks([0, 2, 4, 6, 8])
    ax2.set_ylim(0, 8)
    ax2.legend(fontsize=12)
    for label in ax2.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax2.get_yticklabels():
        label.set_fontweight('bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":


    # df_llama_fewshot=read_csv_file('Combined_fewshot_llama.csv')
    # df_gpt_fewshot=read_csv_file('Combined_fewshot_gpt.csv')
    # df_gpt_zeroshot=read_csv_file('Combined_zeroshot_gpt.csv')
    #remove_duplicate_monomer_pairs('Combined_zeroshot_gpt.csv')
    

    df_gpt = read_csv_file('Unique_monomer_gpt.csv')
    df_llama = read_csv_file('Unique_monomer_llama.csv')
    df_deepseek = read_csv_file('Unique_monomer_deepseek.csv')

    df_llama_fewshot=read_csv_file('Unique_monomer_llama_fewshot.csv')
    df_gpt_fewshot=read_csv_file('Unique_monomer_gpt_fewshot.csv')
    df_gpt_zeroshot=read_csv_file('Unique_monomer_gpt_zeroshot.csv')


    traininf_df = load_trained_data()
    ft_generated_pairs=[]
    training_pairs=[]
    base_generated_pairs=[]


    for _, row in df_llama_fewshot.iterrows():
        base_generated_pairs.append({"m1":row['Fixed Monomer 1'], "m2":row['Fixed Monomer 2']})
    for _, row in df_gpt_fewshot.iterrows():
        base_generated_pairs.append({"m1":row['Fixed Monomer 1'], "m2":row['Fixed Monomer 2']})
    for _, row in df_gpt_zeroshot.iterrows():
        base_generated_pairs.append({"m1":row['Fixed Monomer 1'], "m2":row['Fixed Monomer 2']})

    for _, row in df_gpt.iterrows():
        ft_generated_pairs.append({"m1":row['Fixed Monomer 1'], "m2":row['Fixed Monomer 2']})
    for _, row in df_llama.iterrows():
        ft_generated_pairs.append({"m1":row['Fixed Monomer 1'], "m2":row['Fixed Monomer 2']})
    for _, row in df_deepseek.iterrows():
        ft_generated_pairs.append({"m1":row['Fixed Monomer 1'], "m2":row['Fixed Monomer 2']})
    for _, row in traininf_df.iterrows():
        training_pairs.append({"m1":row['Smiles1'], "m2":row['Smiles2']})

    # 2) Compute internal metrics (both models)
    metrics_base = evaluate_internal(base_generated_pairs)
    metrics_ft   = evaluate_internal(ft_generated_pairs)

    # plot_nn_similarity_box(metrics_base['nn_sims'], metrics_ft['nn_sims'], save_path='nn_similarity_box.svg')
    # plot_pairwise_similarity_hist(metrics_base['pairwise_sims'], metrics_ft['pairwise_sims'], save_path='pairwise_similarity_box.svg')
    plot_combined_similarity_analysis(
    metrics_base['nn_sims'], 
    metrics_ft['nn_sims'], 
    metrics_base['pairwise_sims'], 
    metrics_ft['pairwise_sims'], 
    save_path='combined_similarity_analysis.svg'
)
    # # 3) Compute training overlap (fine-tuned only)
    # overlap_ft = training_overlap(ft_generated_pairs, training_pairs,
    #                             thresholds=(0.7, 0.8))

    # # 4) Print or assemble a table row
    # def pretty(d): 
    #     for k,v in d.items(): print(f"{k}: {v}")
    # #print("BASE (internal):"); pretty(metrics_base)
    # print("\nFINETUNED (internal):"); pretty(metrics_ft)
    # print("\nFINETUNED (train overlap):"); pretty(overlap_ft)


