import pandas as pd
import numpy as np
import os
import json
import joblib
import matplotlib.pyplot as plt
import re
import traceback
from datetime import datetime
from sklearn.inspection import PartialDependenceDisplay
from sklearn.preprocessing import MinMaxScaler
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDLogger
from src.utils import ConfigLoader, get_run_dir

# ==============================================================================
# 1. Global Configurations and Constants
# ==============================================================================

FP_LENGTH = 2048
FP_RADIUS = 2

# SMILES column indices (1-based)
TARGET_SMILES_COL_INDICES_1_BASED = [1, 3, 5, 7, 9, 11, 13]

# Target property column indices (1-based)
SMILES_FILE_TARGET_COL_MAP = {
    'emission': 15, 
    'lifetime': 16 
}

# Task configs
TASK_CONFIG = {
    'emission': {
        'data_path_key': 'training_em',
        'model_path_key': 'em_model',
    },
    'lifetime': {
        'data_path_key': 'training_life',
        'model_path_key': 'life_model',
    }
}

# Categorical columns that need One-Hot encoding
CATEGORICAL_COLS = ['Preparation_Method_Code', 'Step_1_Reaction_Code', 'Step_2_Reaction_Code']

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*') 

# ==============================================================================
# 2. Auxiliary Function Definitions
# ==============================================================================

def normalize_feature_name(name):
    return str(name).replace(" ", "").replace("_", "").lower()

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def preprocess_data_for_model(df):
    """Reproduced preprocessing logic from training step"""
    existing_cat_cols = [col for col in CATEGORICAL_COLS if col in df.columns]
    for col in existing_cat_cols:
        df[col] = df[col].astype(str)
    df_processed = pd.get_dummies(df, columns=existing_cat_cols, drop_first=True)
    scaler = MinMaxScaler()
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
    return df_processed

def clean_features_from_json(json_data):
    keys_imp = json_data.get("key_Importance", [])
    if not keys_imp: keys_imp = json_data.get("key_importance", [])
    keys_bits = json_data.get("key_bits_to_decode", [])
    raw_features = set(keys_imp + keys_bits)
    all_pdp_features = []
    bit_features_to_decode = []
    for feat in raw_features:
        feat = str(feat).strip()
        if not feat: continue
        all_pdp_features.append(feat)
        if feat.startswith("Bit_"):
            try:
                bit_id = int(feat.split('_')[1])
                bit_features_to_decode.append((feat, bit_id))
            except: pass
    return list(set(all_pdp_features)), sorted(list(set(bit_features_to_decode)))

# ==============================================================================
# 3. Core Analysis Logic Processing Function
# ==============================================================================

def run_pdp_analysis(model_path, data_path, features, output_dir, task_type):
    print(f"\n>>> [1/2] Starting PDP Analysis ({task_type})...")
    pdp_failures = [] 

    if not os.path.exists(model_path): return [f"Model file does not exist: {model_path}"]
    if not os.path.exists(data_path): return [f"Data file does not exist: {data_path}"]

    try:
        model = joblib.load(model_path)
    except Exception as e:
        return [f"Model load failed: {str(e)}"]

    try:
        if data_path.endswith('.csv'): df_raw = pd.read_csv(data_path)
        else: df_raw = pd.read_excel(data_path)
        df_raw.columns = df_raw.columns.astype(str).str.strip()
    except Exception as e:
        return [f"Data load failed: {str(e)}"]

    try:
        X_full = preprocess_data_for_model(df_raw)
    except Exception as e:
        return [f"Data preprocessing failed: {str(e)}"]

    model_feats = getattr(model, "feature_names_in_", None)
    if model_feats is None:
        X = X_full
    else:
        missing = [f for f in model_feats if f not in X_full.columns]
        if missing:
            missing_df = pd.DataFrame(0, index=X_full.index, columns=missing)
            X_full = pd.concat([X_full, missing_df], axis=1)
        X = X_full[[f for f in model_feats if f in X_full.columns]]

    valid_features = []
    df_cols_normalized = {normalize_feature_name(c): c for c in X.columns}
    
    for feat in features:
        if feat in X.columns:
            valid_features.append(feat)
        else:
            norm_feat = normalize_feature_name(feat)
            if norm_feat in df_cols_normalized:
                valid_features.append(df_cols_normalized[norm_feat])
            else:
                pdp_failures.append(f"Feature not found: {feat}")

    if not valid_features:
        return pdp_failures + ["No valid features capable of being drawn."]

    pdp_dir = os.path.join(output_dir, "PDP_Plots")
    ensure_dir(pdp_dir)
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial'] 
    plt.rcParams.update({'font.size': 14}) # Increase font size
    
    print(f"    Drawing {len(valid_features)} PDP charts...")
    
    for feat in valid_features:
        try:
            fig, ax = plt.subplots(figsize=(8, 6)) # Increase figsize
            display = PartialDependenceDisplay.from_estimator(
                estimator=model, X=X, features=[feat], ax=ax, kind="average",
                line_kw={'color': '#d62728', 'linewidth': 3} # Thicker lines
            )
            safe_name = normalize_feature_name(feat)
            
            # Tweak title/label fonts if needed
            ax.set_title(f"PDP: {feat}", fontsize=16, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=12)
            
            fig.savefig(os.path.join(pdp_dir, f"PDP_{safe_name}.png"), dpi=300, bbox_inches='tight')
            
            pd_result = display.pd_results[0]
            if 'grid_values' in pd_result: x_values = pd_result['grid_values'][0]
            elif 'values' in pd_result: x_values = pd_result['values'][0]
            else: x_values = [] 

            y_values = pd_result['average'][0]
            if len(x_values) > 0:
                pdp_data_df = pd.DataFrame({'Feature_Value': x_values, 'Partial_Dependence': y_values})
                pdp_data_df.to_csv(os.path.join(pdp_dir, f"PDP_data_{safe_name}.csv"), index=False)
            plt.close(fig)
        except Exception as e:
            pdp_failures.append(f"Plot failed ({feat}): {str(e)}")
            plt.close(fig)

    print(f"✅ PDP Analysis finished.")
    return pdp_failures

def decode_bit_structures(bit_list, smiles_file, target_col_idx_1based, output_dir, task_type):
    print(f"\n>>> [2/2] Starting Bit structure decode (Algo: Hashed Morgan Count)...")
    
    if not os.path.exists(smiles_file): 
        print(f"❌ SMILES file doesn't exist: {smiles_file}")
        return []

    try:
        if smiles_file.endswith('.csv'): df = pd.read_csv(smiles_file)
        else: df = pd.read_excel(smiles_file)
    except Exception as e:
        print(f"❌ Read SMILES data failed: {e}")
        return []

    smiles_indices = [i-1 for i in TARGET_SMILES_COL_INDICES_1_BASED]
    y_col_idx = target_col_idx_1based - 1
    if y_col_idx >= df.shape[1]: y_col_idx = -1

    bit_errors = []

    for bit_name, bit_id in bit_list:
        bit_dir = os.path.join(output_dir, f"Structure_{bit_name}")
        ensure_dir(bit_dir)
        
        found_sample = None 
        try: df_sorted = df.sort_values(by=df.columns[y_col_idx], ascending=False)
        except: df_sorted = df

        for idx, row in df_sorted.iterrows():
            if found_sample: break 
            for col_idx in smiles_indices:
                val = row.iloc[col_idx]
                smiles = str(val).strip()
                if not smiles or smiles.lower() in ['nan', '0', 'none']: continue
                
                mol = Chem.MolFromSmiles(smiles)
                if not mol: continue
                
                bi = {}
                fp = AllChem.GetHashedMorganFingerprint(mol, radius=FP_RADIUS, nBits=FP_LENGTH, bitInfo=bi)
                
                if bit_id in bi:
                    found_sample = {"mol": mol, "bi": bi, "smiles": smiles, "val": row.iloc[y_col_idx]}
                    break
        
        if not found_sample:
            continue

        try:
            mol = found_sample['mol']
            bi = found_sample['bi']
            atom_idx, radius = bi[bit_id][0]
            
            if radius == 0:
                submol = Chem.RWMol()
                new_idx = submol.AddAtom(mol.GetAtomWithIdx(atom_idx))
            else:
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
                if not env:
                    submol = Chem.RWMol()
                    submol.AddAtom(mol.GetAtomWithIdx(atom_idx))
                else:
                    submol = Chem.PathToSubmol(mol, env)
            
            for atom in submol.GetAtoms():
                symbol = atom.GetSymbol()
                if symbol == 'C':
                    atom.SetProp('atomLabel', 'C')
            
            try:
                AllChem.Compute2DCoords(submol)
            except: pass

            d = rdMolDraw2D.MolDraw2DSVG(500, 500) # Larger canvas
            opts = d.drawOptions()
            opts.explicitMethyl = True   
            opts.legendFontSize = 24 # Larger font
            opts.bondLineWidth = 3.0
            opts.padding = 0.1
            
            legend_text = f"Bit {bit_id} | Src: {found_sample['smiles'][:10]}..."
            d.DrawMolecule(submol, legend=legend_text)
            d.FinishDrawing()
            svg_text = d.GetDrawingText()
            
            with open(os.path.join(bit_dir, f"{bit_name}_substructure.svg"), 'w') as f:
                f.write(svg_text)
            
            substructure_smiles = Chem.MolToSmiles(submol)
            
            with open(os.path.join(bit_dir, "source_info.txt"), 'w') as f:
                f.write(f"Bit Name: {bit_name}\n")
                f.write(f"Bit ID: {bit_id}\n")
                f.write(f"Substructure SMILES (Target Fragment): {substructure_smiles}\n")
                f.write(f"Source Molecule SMILES (Context): {found_sample['smiles']}\n")
                f.write(f"Target Property Value: {found_sample['val']}\n")
                
        except Exception as e:
            err_msg = f"Plot error ({bit_name}): {str(e)}"
            bit_errors.append(err_msg)

    print(f"✅ Bit Structure decode finished for {task_type}.")
    return bit_errors

# ==============================================================================
# 4. Class Encapsulation
# ==============================================================================

class DeepAnalysisRunner:
    
    def __init__(self):
        self.config_loader = ConfigLoader.get_instance()
        # Data paths are now fetched via config loader
        
        # Use child directory "Deep_Analysis" in the dynamic unified run_dir
        self.output_base_dir = os.path.join(get_run_dir(), "Deep_Analysis")
        ensure_dir(self.output_base_dir)
        
        print(f"💡 DeepAnalysisRunner initialized successfully.")
        print(f"📂 This analysis results will be saved to: {self.output_base_dir}")

    def analyze(self, json_input):
        if isinstance(json_input, str):
            try: data = json.loads(json_input)
            except: return {"status": "error", "message": "Invalid JSON input."}
        else: data = json_input

        task_prop = data.get("target_property", "emission").lower()
        tasks_to_run = []
        if 'emission' in task_prop or 'em' in task_prop: tasks_to_run.append('emission')
        if 'lifetime' in task_prop or 'life' in task_prop or 'lpl' in task_prop: tasks_to_run.append('lifetime')
        
        if not tasks_to_run:
            return {"status": "error", "message": f"Could not determine task from '{task_prop}'"}

        print(f"Detected tasks: {tasks_to_run}")

        pdp_features, bit_features_tuples = clean_features_from_json(data)
        
        all_errors = []
        final_output_dirs = []

        for current_task in tasks_to_run:
            print(f"\n=== Running analysis for TARGET: {current_task} ===")
            
            task_conf = TASK_CONFIG.get(current_task)
            
            train_data_path = self.config_loader.get_data_path(task_conf['data_path_key'])
            model_path = self.config_loader.get_model_path(task_conf['model_path_key'])
            smiles_data_path = self.config_loader.get_data_path('smiles_raw')
            target_col_idx = SMILES_FILE_TARGET_COL_MAP.get(current_task, 15)
            
            task_out_dir = os.path.join(self.output_base_dir, current_task)
            ensure_dir(task_out_dir)
            final_output_dirs.append(task_out_dir)

            pdp_errors = run_pdp_analysis(model_path, train_data_path, pdp_features, task_out_dir, current_task)
            all_errors.extend([f"[{current_task}] PDP Error: {e}" for e in pdp_errors])

            bit_errors = decode_bit_structures(bit_features_tuples, smiles_data_path, target_col_idx, task_out_dir, current_task)
            all_errors.extend([f"[{current_task}] Bit Error: {e}" for e in bit_errors])

        if all_errors:
            return {"status": "partial_success", "errors": all_errors, "output_dirs": final_output_dirs}
        else:
            print(f"\n🎉 Deep analysis completed successfully.")
            return {"status": "success", "output_dirs": final_output_dirs}