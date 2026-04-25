# src/llm_agents/optimizer.py

import os
import json
import yaml
import joblib
import pandas as pd
import numpy as np
import random
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from src.utils import ConfigLoader, get_run_dir
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import RDLogger
import warnings

# ==============================================================================
# Global Configuration & Plot Style
# ==============================================================================

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.weight'] = 'normal'

mpl.rcParams['figure.dpi'] = 600
mpl.rcParams['savefig.dpi'] = 600

mpl.rcParams['axes.labelsize'] = 48
mpl.rcParams['xtick.labelsize'] = 40
mpl.rcParams['ytick.labelsize'] = 40
mpl.rcParams['legend.fontsize'] = 34
mpl.rcParams['font.size'] = 34

mpl.rcParams['axes.linewidth'] = 4.0
mpl.rcParams['xtick.major.width'] = 4.0
mpl.rcParams['ytick.major.width'] = 4.0
mpl.rcParams['xtick.major.size'] = 14
mpl.rcParams['ytick.major.size'] = 14
mpl.rcParams['grid.linewidth'] = 2.0

COMMON_MATRICES = [
    {'Name': 'Urea', 'SMILES': 'NC(=O)N', 'MW': 60.06},
    {'Name': 'Boric Acid', 'SMILES': 'OB(O)O', 'MW': 61.83},
    {'Name': 'Biuret', 'SMILES': 'NC(=O)NC(=O)N', 'MW': 103.08}
]

PREP_METHOD_MAP = {0: 'None', 1: 'One-step', 2: 'Two-step', 3: 'Multi-step'}
REACTION_CODE_MAP = {0: 'None', 1: 'Hydrothermal', 2: 'Solvothermal', 3: 'Calcination', 4: 'Microwave', 5: 'Solid-state'}

# ==============================================================================
# Optimizer Agent Class
# ==============================================================================

class OptimizerAgent:
    def __init__(self):
        self.config_loader = ConfigLoader.get_instance()
        self.config = self.config_loader.config
        em_model_path = self.config_loader.get_model_path('em_model')
        self.data_dir = os.path.dirname(em_model_path) if em_model_path else "data"
        self.log_dir = os.path.join(get_run_dir(), "logs")
        self.img_dir = os.path.join(get_run_dir(), "images")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)
        self._models = {}
        self._contexts = {}

    def _load_model_and_context(self, task_type):
        if task_type in self._contexts:
            return self._models[task_type], self._contexts[task_type]
        
        print(f"🔧 [Optimizer] Loading {task_type} model and context...")
        prefix = "em" if task_type == 'emission' else "life"
        model_path = os.path.join(self.data_dir, f"trained_{prefix}_model.pkl")
        scaler_path = os.path.join(self.data_dir, f"{prefix}_scaler.pkl")
        features_path = os.path.join(self.data_dir, f"{prefix}_feature_names.json")

        if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path)):
            print(f"❌ Error: Missing model files ({prefix})")
            print(f"   -> Model: {model_path} [{os.path.exists(model_path)}]")
            print(f"   -> Scaler: {scaler_path} [{os.path.exists(scaler_path)}]")
            print(f"   -> Features: {features_path} [{os.path.exists(features_path)}]")
            return None, None
        
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            with open(features_path, 'r') as f:
                feature_names = json.load(f)
        except Exception as e:
            print(f"❌ Loading failed: {e}")
            return None, None

        context = {
            "scaler": scaler,
            "feature_names": feature_names,
            "base_feature_set": set(feature_names)
        }
        self._models[task_type] = model
        self._contexts[task_type] = context
        return model, context

    # --------------------------------------------------------------------------
    # Feature Calculation Helpers
    # --------------------------------------------------------------------------
    def _calculate_weighted_fingerprint_vector(self, components):
        total_vector = np.zeros(2048, dtype=float)
        for comp in components:
            smiles = comp.get('smiles')
            moles = float(comp.get('moles', 0.0))
            if not smiles or moles <= 0:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                continue
            fp_gen = AllChem.GetHashedMorganFingerprint(mol, radius=2, nBits=2048)
            for bit, count in fp_gen.GetNonzeroElements().items():
                if bit < 2048:
                    total_vector[bit] += count * moles
        return total_vector

    def _get_raw_fp_vector(self, smiles):
        vec = np.zeros(2048, dtype=float)
        if not smiles:
            return vec
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return vec
        fp_gen = AllChem.GetHashedMorganFingerprint(mol, radius=2, nBits=2048)
        for bit, count in fp_gen.GetNonzeroElements().items():
            if bit < 2048:
                vec[bit] = count
        return vec

    def _calculate_mol_descriptors(self, smiles, moles):
        keys = ['Molecular Weight', 'C', 'H', 'O', 'N', 'Other_atoms',
                'TPSA', 'Amino', 'Amide Group', '-COOH', '-OH', 'C=O',
                '-C≡N', '-SO₃H', 'Log P', 'Melting Point', 'Boiling Point', 'Molar Amount']
        desc = {k: 0.0 for k in keys}
        desc['Molar Amount'] = moles
        
        if not smiles or moles <= 0:
            return desc
        
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return desc
        
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        atoms = {'C': 0, 'H': 0, 'O': 0, 'N': 0, 'Other': 0}
        mol_h = Chem.AddHs(mol)
        
        for atom in mol_h.GetAtoms():
            sym = atom.GetSymbol()
            if sym in atoms:
                atoms[sym] += 1
            else:
                atoms['Other'] += 1
        
        pats = {
            'Amino': '[NX3;H2,H1;!$(NC=O)]',
            'Amide Group': '[NX3][CX3](=[OX1])',
            '-COOH': 'C(=O)[OH]',
            '-OH': '[OX2H]',
            'C=O': '[CX3]=[OX1]',
            '-C≡N': 'C#N',
            '-SO₃H': 'S(=O)(=O)[OH]'
        }
        groups = {}
        for k, sm in pats.items():
            groups[k] = len(mol.GetSubstructMatches(Chem.MolFromSmarts(sm)))
        
        desc['Molecular Weight'] = mw
        desc['C'] = atoms['C'] * moles
        desc['H'] = atoms['H'] * moles
        desc['O'] = atoms['O'] * moles
        desc['N'] = atoms['N'] * moles
        desc['Other_atoms'] = atoms['Other'] * moles
        desc['TPSA'] = tpsa * moles
        desc['Log P'] = logp * moles
        
        for k, v in groups.items():
            desc[k] = v * moles
        
        return desc

    def _fill_feature_row(self, row_dict, prefix, desc_dict, train_cols_set):
        for k, v in desc_dict.items():
            col_name = f"{prefix}_{k}"
            if col_name in train_cols_set:
                row_dict[col_name] = v
                continue
            if k == '-C≡N' and f"{prefix}_-C≡N" in train_cols_set:
                row_dict[f"{prefix}_-C≡N"] = v
            elif k == '-SO₃H' and f"{prefix}_-SO₃H" in train_cols_set:
                row_dict[f"{prefix}_-SO₃H"] = v

    def _flatten_input_vector(self, batch_tuple):
        row, pre1, pre2, zeros, test = batch_tuple
        flat_dict = row.copy()
        prefixes = [('Pre1', pre1), ('Pre2', pre2), ('Pre3', zeros), ('Pre4', zeros),
                    ('Pre5', zeros), ('Step1', zeros), ('Step2', zeros)]
        
        for prefix, d in prefixes:
            for k, v in d.items():
                flat_dict[f"{prefix}_{k}"] = v
                if k == '-C≡N':
                    flat_dict[f"{prefix}_-C≡N"] = v
                elif k == '-SO₃H':
                    flat_dict[f"{prefix}_-SO₃H"] = v
        
        for k, v in test.items():
            flat_dict[f"Test_{k}"] = v
            if k == 'Amino':
                flat_dict['Test_Amino Group'] = v
            if k == 'Amide Group':
                flat_dict['Test_Amide'] = v
        
        return flat_dict

    # ==========================================================================
    # Plotting Functions
    # ==========================================================================
    def _batch_plot_all_surfaces(self, df_plot, task_type, best_meta):
        print(f"📊 [Optimizer] Analyzing data distribution and generating 3D views...")
        param_map = {
            'Step_1_Temperature': 'S1 Temp',
            'Step_1_Time': 'S1 Time',
            'Ratio': 'Ratio',
            'Step1_Carbon_Dots_Dosage': 'Inter-step Dosage',
            'Step_2_Temperature': 'S2 Temp',
            'Step 2_Time': 'S2 Time'
        }
        
        valid_cols = []
        for col, label in param_map.items():
            if col in df_plot.columns and df_plot[col].std() > 0.01:
                valid_cols.append(col)
        
        if len(valid_cols) < 2:
            print("⚠️ Insufficient variable parameters, skipping plotting.")
            return
        
        combinations = list(itertools.combinations(valid_cols, 2))
        print(f"   -> Active dimensions identified: {valid_cols}")
        
        for x_col, y_col in combinations:
            self._plot_single_surface(df_plot, x_col, y_col, param_map, task_type, best_meta)

    def _plot_single_surface(self, df, x_col, y_col, param_map, task_type, best_meta):
        try:
            from scipy.interpolate import griddata
            has_scipy = True
        except ImportError:
            has_scipy = False

        try:
            x_label = param_map.get(x_col, x_col)
            y_label = param_map.get(y_col, y_col)
            fname_x = x_col.replace(' ', '').replace('_', '')
            fname_y = y_col.replace(' ', '').replace('_', '')
            save_name = f"Opt_3D_{fname_x}_vs_{fname_y}.png"
            
            x = df[x_col].values
            y = df[y_col].values
            z = df['Score'].values
            
            fig = plt.figure(figsize=(16, 14))
            ax = fig.add_subplot(111, projection='3d')
            ax.dist = 12
            ax.set_box_aspect((1, 1, 0.7))

            surf = None
            if has_scipy:
                xi = np.linspace(min(x), max(x), 100)
                yi = np.linspace(min(y), max(y), 100)
                Xi, Yi = np.meshgrid(xi, yi)
                Zi = griddata((x, y), z, (Xi, Yi), method='linear')
                Zi = np.nan_to_num(Zi, nan=np.nanmin(z))
                surf = ax.plot_surface(Xi, Yi, Zi, cmap=cm.Spectral_r, rstride=1, cstride=1,
                                     linewidth=0, antialiased=True, shade=False, alpha=0.9)
            else:
                surf = ax.plot_trisurf(x, y, z, cmap=cm.Spectral_r, edgecolor='none',
                                     linewidth=0, antialiased=True, shade=False)

            max_idx = np.argmax(z)
            x_peak, y_peak, z_peak = x[max_idx], y[max_idx], z[max_idx]
            z_min = np.min(z)
            z_rng = np.max(z) - z_min
            z_txt = z_peak + (z_rng * 0.25 if z_rng > 0.01 else z_peak * 0.05)

            ax.scatter([x_peak], [y_peak], [z_peak], c='gold', s=600, marker='*',
                     zorder=50, edgecolors='#d48806', linewidth=2.5)
            
            ax.plot([x_peak, x_peak], [y_peak, y_peak], [z_min, z_peak],
                  color='#d48806', linestyle='--', lw=4.5, alpha=0.8)
            
            x_fmt = ".0f" if abs(x_peak) > 10 else ".1f"
            y_fmt = ".0f" if abs(y_peak) > 10 else ".1f"
            
            lbl = f"Best: {z_peak:.2f}\n{x_peak:{x_fmt}} / {y_peak:{y_fmt}}"
            ax.text(x_peak, y_peak, z_txt, lbl, ha='center', va='bottom',
                  fontsize=38, family='Arial', weight='bold',
                  bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', lw=3, alpha=0.9), zorder=100)

            label_font = {'family': 'Arial', 'weight': 'bold', 'size': 48}
            ax.set_xlabel(x_label, fontdict=label_font, labelpad=60)
            ax.set_ylabel(y_label, fontdict=label_font, labelpad=60)
            ax.set_zlabel('Score', fontdict=label_font, labelpad=45)
            
            ax.tick_params(axis='x', labelsize=40, pad=15)
            ax.tick_params(axis='y', labelsize=40, pad=15)
            ax.tick_params(axis='z', labelsize=40, pad=15)

            for t in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
                t.set_fontname('Arial')
                t.set_fontweight('normal')

            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.grid(True, linestyle=':', alpha=0.4, linewidth=2.0)
            
            cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=15, pad=0.15)
            cbar.ax.tick_params(labelsize=34, width=3, length=10)

            ax.view_init(elev=35, azim=-55)
            plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

            save_path = os.path.join(self.img_dir, save_name)
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.5, transparent=True)
            plt.close()
            print(f"✅ [Vis] 3D plot saved: {save_path}")
        except Exception as e:
            print(f"⚠️ Plotting error: {e}")

    # ==========================================================================
    # Main Optimization Function
    # ==========================================================================
    def optimize(self, summary_report, candidates):
        target_prop = str(summary_report.get("target_property", "")).lower()
        has_em = ('emission' in target_prop or 'em' in target_prop)
        has_life = ('lifetime' in target_prop or 'life' in target_prop)
        
        if not has_em and not has_life:
            has_em = True
        
        is_dual_task = has_em and has_life
        primary_task = 'emission' if has_em else 'lifetime'

        models = {}
        if has_em:
            m, c = self._load_model_and_context('emission')
            if m:
                models['emission'] = {'model': m, 'ctx': c}
        if has_life:
            m, c = self._load_model_and_context('lifetime')
            if m:
                models['lifetime'] = {'model': m, 'ctx': c}
        
        if not models:
            return {"Error": "No models loaded"}

        N_ITER = 300
        RATIOS = [10.0, 20.0, 30.0, 50.0, 80.0, 100.0, 150.0, 200.0]
        BIT_NAMES = [f"Bit_{i}" for i in range(2048)]
        DESC_TEST_KEYS = ['C','H','O','N','Other_atoms','TPSA','Amino','Amide Group','-COOH','-OH','C=O','-C≡N','-SO₃H','Log P','Melting Point','Boiling Point']
        
        REACTION_TEMP_RANGES = {
            1: (180, 280),
            2: (180, 300),
            3: (300, 600),
            4: (150, 250),
            5: (250, 500),
        }
        REACTION_TIME_RANGES = {
            1: (4.0, 24.0),
            2: (4.0, 24.0),
            3: (1.0, 6.0),
            4: (0.1, 1.0),
            5: (1.0, 8.0),
        }
        
        S1_VALID_RXNS = [1, 2, 4]
        S2_VALID_RXNS = [3, 5]
        
        all_flat_inputs_list = []
        all_predictions_list = []
        all_meta_list = []

        best_plot_records = []
        global_best_score = -9999
        optimized_candidates = []
        best_plot_meta = {}

        desc_zeros = self._calculate_mol_descriptors(None, 0)

        mat_fp_cache = {}
        mat_desc_cache = {}
        for mat in COMMON_MATRICES:
            ms = mat['SMILES']
            mat_fp_cache[ms] = self._get_raw_fp_vector(ms)
            desc_unit = self._calculate_mol_descriptors(ms, 1.0)
            for r in RATIOS:
                d = {}
                for k, v in desc_unit.items():
                    if k == 'Molecular Weight':
                        d[k] = v
                    elif k == 'Molar Amount':
                        d[k] = r
                    else:
                        d[k] = v * r
                mat_desc_cache[(ms, r)] = d

        print(f"🚀 Starting optimization search (Total: {len(candidates) * len(COMMON_MATRICES) * N_ITER} iterations)...")
        print(f"   📋 Matrix ratio range: {RATIOS}")
        print(f"   🌡️ S1 Temp ranges: Hydrothermal{REACTION_TEMP_RANGES[1]}, Solvothermal{REACTION_TEMP_RANGES[2]}, Microwave{REACTION_TEMP_RANGES[4]}")
        print(f"   🔥 S2 Temp ranges: Calcination{REACTION_TEMP_RANGES[3]}, Solid-state{REACTION_TEMP_RANGES[5]}")
        
        for mol_data in candidates:
            c_smiles = mol_data.get('SMILES')
            c_name = mol_data.get('Name')
            if not c_smiles:
                continue
            
            desc_pre1_base = self._calculate_mol_descriptors(c_smiles, 1.0)
            fp_raw_candidate = self._get_raw_fp_vector(c_smiles)
            
            for mat in COMMON_MATRICES:
                mat_smiles = mat['SMILES']
                mat_name = mat['Name']
                fp_raw_matrix = mat_fp_cache[mat_smiles]
                batch_rows = []
                batch_meta = []

                v_ratios = np.random.choice(RATIOS, size=N_ITER)
                v_methods = np.random.choice([1, 2], size=N_ITER)
                v_s1_rxns = np.random.choice(S1_VALID_RXNS, size=N_ITER)
                v_dosages = np.round(np.random.uniform(0.5, 10, N_ITER), 2)
                v_s2_rxns = np.random.choice(S2_VALID_RXNS, size=N_ITER)
                
                v_s1_temps = np.zeros(N_ITER)
                v_s1_times = np.zeros(N_ITER)
                v_s2_temps = np.zeros(N_ITER)
                v_s2_times = np.zeros(N_ITER)
                
                for _i in range(N_ITER):
                    t_lo, t_hi = REACTION_TEMP_RANGES[int(v_s1_rxns[_i])]
                    v_s1_temps[_i] = round(np.random.uniform(t_lo, t_hi), 1)
                    tm_lo, tm_hi = REACTION_TIME_RANGES[int(v_s1_rxns[_i])]
                    v_s1_times[_i] = round(np.random.uniform(tm_lo, tm_hi), 1)
                    t2_lo, t2_hi = REACTION_TEMP_RANGES[int(v_s2_rxns[_i])]
                    v_s2_temps[_i] = round(np.random.uniform(t2_lo, t2_hi), 1)
                    tm2_lo, tm2_hi = REACTION_TIME_RANGES[int(v_s2_rxns[_i])]
                    v_s2_times[_i] = round(np.random.uniform(tm2_lo, tm2_hi), 1)

                for iter_idx in range(N_ITER):
                    ratio = float(v_ratios[iter_idx])
                    method = int(v_methods[iter_idx])
                    s1_temp = float(v_s1_temps[iter_idx])
                    s1_time = float(v_s1_times[iter_idx])
                    s1_rxn = int(v_s1_rxns[iter_idx])
                    
                    if method == 1:
                        dosage = 0.0
                        s2_rxn = 0
                        s2_temp = 0.0
                        s2_time = 0.0
                    else:
                        dosage = float(v_dosages[iter_idx])
                        s2_rxn = int(v_s2_rxns[iter_idx])
                        s2_temp = float(v_s2_temps[iter_idx])
                        s2_time = float(v_s2_times[iter_idx])

                    desc_pre2 = mat_desc_cache[(mat_smiles, ratio)]
                    fp_vec = fp_raw_candidate + ratio * fp_raw_matrix
                    desc_test = {k: desc_pre1_base.get(k, 0) + desc_pre2.get(k, 0) for k in DESC_TEST_KEYS}

                    row = dict(zip(BIT_NAMES, fp_vec))
                    row.update({
                        'Step_1_Temperature': s1_temp,
                        'Step_1_Time': s1_time,
                        'Step1_Carbon_Dots_Dosage': dosage,
                        'Step_2_Temperature': s2_temp,
                        'Step 2_Time': s2_time,
                        'Test_Temperature': 293.0,
                        'Ex': 365.0
                    })
                    
                    row[f'Preparation_Method_Code_{int(method)}'] = 1.0
                    row[f'Step_1_Reaction_Code_{int(s1_rxn)}'] = 1.0
                    row[f'Step_2_Reaction_Code_{int(s2_rxn)}'] = 1.0
                    
                    batch_tuple = (row, desc_pre1_base, desc_pre2, desc_zeros, desc_test)
                    batch_rows.append(batch_tuple)
                    all_flat_inputs_list.append(self._flatten_input_vector(batch_tuple))

                    batch_meta.append({
                        'Precursor': c_name,
                        'Matrix': mat_name,
                        'Ratio': ratio,
                        'Method_Code': method,
                        'Dosage': dosage,
                        'Step_1_Temperature': s1_temp,
                        'Step_1_Time': s1_time,
                        'Step_2_Temperature': s2_temp,
                        'Step 2_Time': s2_time
                    })

                if not batch_rows:
                    continue
                
                preds_em = self._predict_batch(batch_rows, models['emission']['model'], models['emission']['ctx']) if 'emission' in models else None
                preds_life = self._predict_batch(batch_rows, models['lifetime']['model'], models['lifetime']['ctx']) if 'lifetime' in models else None

                current_batch_records = []
                for idx in range(len(batch_rows)):
                    em_val = preds_em[idx] if preds_em is not None else -1
                    life_val = preds_life[idx] if preds_life is not None else -1
                    
                    all_predictions_list.append({'Predicted_Emission': em_val, 'Predicted_Lifetime': life_val})
                    
                    rec = batch_meta[idx].copy()
                    rec['Predicted_Emission'] = em_val
                    rec['Predicted_Lifetime'] = life_val
                    
                    raw_score = 0
                    if is_dual_task:
                        raw_score = 0.6 * (em_val/700.0) + 0.4 * (life_val/1000.0) if em_val>0 and life_val>0 else 0
                    elif has_em:
                        raw_score = em_val
                    elif has_life:
                        raw_score = life_val
                    
                    penalty = 1.0
                    s1_t = rec.get('Step_1_Temperature', 0)
                    ratio_val = rec.get('Ratio', 10)
                    
                    if s1_t < 180:
                        penalty *= 0.1
                    if ratio_val < 10:
                        penalty *= 0.2
                    if 30 <= ratio_val <= 100:
                        penalty *= 1.2
                    
                    score = raw_score * penalty
                    rec['Score'] = score
                    
                    current_batch_records.append(rec)
                    all_meta_list.append(rec)

                scores = [r['Score'] for r in current_batch_records]
                local_max = np.max(scores)
                
                if local_max > global_best_score:
                    global_best_score = local_max
                    best_plot_records = current_batch_records
                    best_plot_meta = {'Matrix': mat_name, 'Molecule': c_name}

                if local_max > 0.1:
                    best_r = current_batch_records[np.argmax(scores)]
                    mol_res = mol_data.copy()
                    m_str = "One-step" if best_r['Method_Code'] == 1 else "Two-step"
                    mol_res['Optimized_Recipe'] = f"{m_str} | {best_r['Ratio']}x {mat_name} @ {best_r['Step_1_Temperature']}C"
                    mol_res['Optimized_Temp'] = f"{best_r['Step_1_Temperature']}°C"
                    mol_res['Predicted_Performance'] = f"Score: {local_max:.2f}"
                    mol_res['_sort_score'] = local_max
                    optimized_candidates.append(mol_res)

        print("💾 Building full model input dataset...")
        if all_flat_inputs_list:
            std_features = []
            if 'emission' in models:
                std_features = models['emission']['ctx']['feature_names']
            elif 'lifetime' in models:
                std_features = models['lifetime']['ctx']['feature_names']
            
            if std_features:
                try:
                    df_features = pd.DataFrame(all_flat_inputs_list)
                    df_aligned = df_features.reindex(columns=std_features, fill_value=0.0)
                    df_preds = pd.DataFrame(all_predictions_list)
                    df_meta = pd.DataFrame(all_meta_list)[['Precursor', 'Matrix', 'Method_Code', 'Ratio', 'Step_1_Temperature']]
                    df_final = pd.concat([df_meta, df_preds, df_aligned], axis=1)
                    
                    save_path = os.path.join(self.log_dir, "Total_Model_Input_Features.csv")
                    df_final.to_csv(save_path, index=False)
                    print(f"✅ Full model inputs saved: {save_path}")
                    print(f"   -> Dimensions: {df_final.shape}")
                except Exception as e:
                    print(f"⚠️ Failed to save CSV: {e}")
            else:
                print("⚠️ Could not retrieve feature list, skipping save.")

        if all_meta_list:
            pd.DataFrame(all_meta_list).to_csv(os.path.join(self.log_dir, "Total_Search_History.csv"), index=False)

        print("\n" + "="*50)
        print("🔝 Top Optimized Candidates:")
        final_list = []
        top_recipe, top_temp = "N/A", "N/A"
        
        if optimized_candidates:
            df_res = pd.DataFrame(optimized_candidates).sort_values('_sort_score', ascending=False).drop_duplicates('SMILES').head(10)
            for idx, row in df_res.iterrows():
                print(f"[{idx+1}] {row.get('Name')} | {row.get('Optimized_Recipe')} | {row.get('Predicted_Performance')}")
            
            final_list = df_res.drop(columns=['_sort_score']).to_dict(orient='records')
            top_recipe = df_res.iloc[0]['Optimized_Recipe']
            top_temp = df_res.iloc[0]['Optimized_Temp']
        else:
            final_list = candidates

        if best_plot_records:
            df_plot_all = pd.DataFrame(best_plot_records)
            best_idx = df_plot_all['Score'].idxmax()
            best_method = df_plot_all.loc[best_idx, 'Method_Code']
            df_filtered = df_plot_all[df_plot_all['Method_Code'] == best_method].copy()
            self._batch_plot_all_surfaces(df_filtered, primary_task, best_plot_meta)

        return {
            "Recipe_Strategy": top_recipe,
            "Temperature": top_temp,
            "Time": "See generated plots",
            "Molecules_With_Params": final_list
        }

    def _predict_batch(self, batch_data_tuple, model, ctx):
        feature_names = ctx['feature_names']
        final_rows = []
        
        for i, (row, pre1, pre2, zeros, test) in enumerate(batch_data_tuple):
            full_row = row.copy()
            self._fill_feature_row(full_row, 'Pre1', pre1, ctx['base_feature_set'])
            self._fill_feature_row(full_row, 'Pre2', pre2, ctx['base_feature_set'])
            
            for p in ['Pre3', 'Pre4', 'Pre5', 'Step1', 'Step2']:
                self._fill_feature_row(full_row, p, zeros, ctx['base_feature_set'])
            
            for k, v in test.items():
                if f"Test_{k}" in ctx['base_feature_set']:
                    full_row[f"Test_{k}"] = v
            
            final_rows.append(full_row)
        
        df_batch = pd.DataFrame(final_rows)
        df_input = pd.DataFrame(0.0, index=df_batch.index, columns=feature_names)
        df_input.update(df_batch[df_batch.columns.intersection(df_input.columns)])
        
        try:
            return model.predict(ctx['scaler'].transform(df_input))
        except:
            return None
