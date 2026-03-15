# src/llm_agents/scout.py

import os
import pandas as pd
import json
import re
import time
import numpy as np
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from openai import OpenAI
from dotenv import load_dotenv
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import random

from src.utils import ConfigLoader, get_run_dir, get_prompt, get_llm_client


mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.weight'] = 'normal'

# Resolution
mpl.rcParams['figure.dpi'] = 600
mpl.rcParams['savefig.dpi'] = 600

# --- [Modification] Significantly increased font sizes ---
mpl.rcParams['axes.labelsize'] = 48    # Axis labels (very large)
mpl.rcParams['xtick.labelsize'] = 40   # Tick labels
mpl.rcParams['ytick.labelsize'] = 40
mpl.rcParams['legend.fontsize'] = 34   # Legend
mpl.rcParams['font.size'] = 34         # Global default

# --- [Modification] Thicker lines ---
mpl.rcParams['axes.linewidth'] = 4.0      # Thicker frame
mpl.rcParams['xtick.major.width'] = 4.0   # Thicker tick marks
mpl.rcParams['ytick.major.width'] = 4.0
mpl.rcParams['xtick.major.size'] = 14     # Longer tick marks
mpl.rcParams['ytick.major.size'] = 14


def _worker_process_chunk(chunk_data):
    """
    Independent process worker unit: processes a chunk of data.
    """
    df_chunk, constraints, smarts_tuples = chunk_data

    # --- [Task Info] Task logic is unified, no separate printing to avoid spam ---
    pass

    
    candidates = []
    bg_samples = []
    
    # Pre-compile SMARTS
    patterns = []
    for s, bid, d in smarts_tuples:
        p = Chem.MolFromSmarts(s)
        if p: patterns.append((p, bid, d))
    
    min_mw = constraints.get('min_mw', 0)
    max_mw = constraints.get('max_mw', 9999)
    
    for _, row in df_chunk.iterrows():
        try:
            cid = row.get('ID') or row.get(0)
            smi = row.get('SMILES') or row.get(1)
            
            if not isinstance(smi, str): continue

            # A. Molecular weight filtering
            mol = Chem.MolFromSmiles(smi)
            if not mol: continue

            mw = Descriptors.MolWt(mol)
            if not (min_mw <= mw <= max_mw):
                continue
            
            # B. Background sampling (0.5%)
            if random.random() < 0.005:
                bg_samples.append({'SMILES': smi})

            # C. Structural feature scoring
            score = 0
            details = []
            
            for pat, bit_id, desc in patterns:
                if mol.HasSubstructMatch(pat):
                    score += 1
                    details.append(f"[{desc} (Bit_{bit_id})]")
            
            # D. Save
            if score > 0:
                candidates.append({
                    'ID': cid,
                    'SMILES': smi,
                    'MW': mw,
                    'Total_Score': score,
                    'Matched_Details': "; ".join(details)
                })
        except Exception:
            continue
            
    return candidates, bg_samples


# ==========================================
#  Main Class Definition: ScoutAgent
# ==========================================
class ScoutAgent:
    def __init__(self):
        self.config_loader = ConfigLoader.get_instance()
        self.client, self.model, self.temperature = get_llm_client()
        self.run_dir = get_run_dir()
        self.img_dir = os.path.join(self.run_dir, "images")
        os.makedirs(self.img_dir, exist_ok=True)

    def _extract_constraints(self, summary_report):
        """Extract molecular weight constraints"""
        constraints = {"min_mw": 0, "max_mw": 9999}
        rules = []
        if "design_guidelines" in summary_report:
            rules.extend(summary_report["design_guidelines"].get("structural_rules", []))
            rules.extend(summary_report["design_guidelines"].get("process_rules", []))
        if "parametric_rules" in summary_report:
            rules.extend(summary_report["parametric_rules"])
            
        for r in rules:
            r_lower = r.lower()
            if "mw" in r_lower or "weight" in r_lower or "molecular weight" in r_lower:
                greater_match = re.search(r'>\s*(\d+)', r_lower)
                if greater_match:
                    constraints["min_mw"] = float(greater_match.group(1))
                    print(f"  🔒 [Summary Constraint] Molecular Weight > {constraints['min_mw']}")
                
                less_match = re.search(r'<\s*(\d+)', r_lower)
                if less_match:
                    constraints["max_mw"] = float(less_match.group(1))
                    print(f"  🔒 [Summary Constraint] Molecular Weight < {constraints['max_mw']}")
        return constraints

    def _get_smarts_from_llm(self, desc):
        """Call LLM to get SMARTS"""
        prompt_template = get_prompt('scout_smarts_prompt')
        prompt = prompt_template.format(desc=desc) if prompt_template else f"What is the simplest RDKit SMARTS string for the chemical group '{desc}'? Only return the string."

        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = resp.choices[0].message.content.strip()
                content = re.sub(r"```.*?```", "", content, flags=re.DOTALL)
                content = re.sub(r"`.*?`", "", content)
                lines = [l.strip() for l in content.split('\n') if l.strip()]
                candidate = lines[-1].split(' ')[0] 
                return candidate
            except Exception as e:
                time.sleep(1)
        return None

    def _compute_chemical_space(self, df_bg, df_top):
        """Compute PCA"""
        print("🧪 [Vis] Computing chemical space coordinates (PCA)...")
        sample_size = min(2000, len(df_bg))
        bg_sample = df_bg.sample(n=sample_size, random_state=42).copy() if len(df_bg) > 0 else pd.DataFrame()
        
        if bg_sample.empty and df_top.empty:
            return None, None

        combined_smiles = bg_sample['SMILES'].tolist() + df_top['SMILES'].tolist()
        
        fps = []
        valid_indices = []
        
        for i, smi in enumerate(combined_smiles):
            m = Chem.MolFromSmiles(smi)
            if m:
                fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
                fps.append(np.array(fp))
                valid_indices.append(i)
                
        if not fps: return None, None

        X = np.array(fps)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        n_bg_total = len(bg_sample)
        n_bg_valid = sum(1 for i in valid_indices if i < n_bg_total)
        
        bg_pca = X_pca[:n_bg_valid]
        top_pca = X_pca[n_bg_valid:]
        
        return bg_pca, top_pca

    def _plot_chemical_space_trajectory(self, bg_pca, top_pca, save_name="chem_space.png"):
        """
        Plot and save the chemical space diagram (No title, extremely large font).
        Contains two versions: with annotation and without annotation.
        """
        if bg_pca is None or top_pca is None: return

        print("🎨 [Vis] Generating chemical space diagram (Big Fonts)...")
        
        # --- Save raw data ---
        base_name = os.path.splitext(save_name)[0]
        data_prefix = os.path.join(self.img_dir, base_name)
        
        if len(bg_pca) > 0:
            df_bg_save = pd.DataFrame(bg_pca, columns=['PC1', 'PC2'])
            df_bg_save.to_csv(f"{data_prefix}_background_points.csv", index=False)

        if len(top_pca) > 0:
            df_top_save = pd.DataFrame(top_pca, columns=['PC1', 'PC2'])
            df_top_save['Step'] = range(1, len(top_pca) + 1)
            df_top_save.to_csv(f"{data_prefix}_top_candidates.csv", index=False)

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 1. Background density (Contour)
        if len(bg_pca) > 5:
            x_bg, y_bg = bg_pca[:, 0], bg_pca[:, 1]
            try:
                xy = np.vstack([x_bg, y_bg])
                kde = gaussian_kde(xy)
                
                xmin, xmax = x_bg.min() - 1, x_bg.max() + 1
                ymin, ymax = y_bg.min() - 1, y_bg.max() + 1
                X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                positions = np.vstack([X.ravel(), Y.ravel()])
                Z = np.reshape(kde(positions).T, X.shape)
                
                # Save KDE data
                df_kde_save = pd.DataFrame({'Grid_X': X.ravel(), 'Grid_Y': Y.ravel(), 'Density_Z': Z.ravel()})
                df_kde_save.to_csv(f"{data_prefix}_kde_density.csv", index=False)
                
                ax.contourf(X, Y, Z, levels=10, cmap='Blues', alpha=0.6, zorder=1)
            except Exception as e:
                print(f"⚠️ KDE Error: {e}")
                ax.scatter(x_bg, y_bg, color='#B0C4DE', alpha=0.5, s=80) # Larger background points
        
        # 2. Top candidate molecules (initialize variables)
        x_top, y_top = None, None
        
        if len(top_pca) > 0:
            x_top, y_top = top_pca[:, 0], top_pca[:, 1]
            
            # --- [Modification] Thicker trajectory line ---
            ax.plot(x_top, y_top, color='#d48806', linewidth=4.5, linestyle='--', alpha=0.8, zorder=2, label='Search Path')
            
            # --- [Modification] Super large marker points (s=600) ---
            ax.scatter(x_top, y_top, 
                       marker='*', 
                       s=500, 
                       color='gold', 
                       edgecolors='#d48806', 
                       linewidth=2.5, 
                       zorder=3, 
                       label='Top Candidates')
            
            # [Note] Original annotation code moved back to save unannotated version first.

        # --- [Modification] Manually assign massive font size for axis labels ---
        ax.set_xlabel('Principal Component 1', family='Arial', weight='bold', fontsize=48)
        ax.set_ylabel('Principal Component 2', family='Arial', weight='bold', fontsize=48)
        
        # Border thickness
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(4.0)
            
        # Ticks
        ax.tick_params(width=4.0, length=14, labelsize=40)
        
        # Legend
        ax.legend(loc='upper right', 
                  frameon=True, fancybox=False, shadow=False, framealpha=0.9,
                  facecolor='white', edgecolor='black', fontsize=34,
                  handlelength=2.5, borderpad=0.8, labelspacing=0.8)
        
        ax.grid(False)
        
        # =========== [NEW] Save version without annotations ===========
        # Before adding annotation, apply tight layout and save a copy
        plt.tight_layout()
        base_name_no_ext = os.path.splitext(save_name)[0]
        no_anno_path = os.path.join(self.img_dir, f"{base_name_no_ext}_no_annotation.png")
        plt.savefig(no_anno_path, bbox_inches='tight')
        print(f"✅ [Vis] Unannotated image saved: {no_anno_path}")
        # =========================================

        # =========== [Moved/Modified] Add Annotations ===========
        if x_top is not None and y_top is not None:
             # --- [Modification] Larger annotation text, position moved downwards ---
            ax.annotate('Best Candidate',
                xy=(x_top[0], y_top[0]), 
                xycoords='data',
                # Modify here: downward offset (y - 1.5), and horizontally centered (ha='center')
                xytext=(x_top[0], y_top[0] - 1.5), 
                textcoords='data',
                ha='center', 
                arrowprops=dict(facecolor='black', shrink=0.05, width=3, headwidth=12), # Thicker arrow
                fontsize=38, # Massive annotation text
                family='Arial',
                weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=3, alpha=0.9))
        # =========================================
        
        # Save final version with annotations
        full_save_path = os.path.join(self.img_dir, save_name)
        # plt.tight_layout() # Already called above, not repeating to avoid jump
        plt.savefig(full_save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ [Vis] Image saved: {full_save_path}")

    # ==========================================
    #  Core entrypoint: search_molecules (Logic unchanged)
    # ==========================================
    def search_molecules(self, summary_report, limit=None, max_mw=None):
        print(f"🚀 [Scout] Starting high-performance parallel search engine (Multi-core)...")
        target_path = self.config_loader.get_data_path('cid_smiles')
        if not target_path or not os.path.exists(target_path):
            print(f"❌ [Scout] Data file not found: {target_path}")
            return []

        constraints = self._extract_constraints(summary_report)
        if max_mw is not None:
            constraints['max_mw'] = min(constraints['max_mw'], max_mw)

        print("⚙️  [Pre-compile] Parsing feature SMARTS...")
        critical_structs = summary_report.get("critical_structures", [])
        if not critical_structs:
            critical_structs = summary_report.get("critical_features_analysis", [])

        smarts_tuples = []
        for struct in critical_structs:
            bit_id = struct.get('feature_name') or struct.get('bit_id', 'Unknown')
            desc = struct.get('chemical_meaning') or struct.get('chemical_desc', '')
            if not desc or "undecoded" in desc.lower(): continue
            
            smarts = self._get_smarts_from_llm(desc)
            if smarts:
                print(f"  -> {bit_id}: {desc} => {smarts}")
                smarts_tuples.append((smarts, bit_id, desc))

        CHUNK_SIZE = 10000 
        # Cap max_workers to 10 to prevent OOM on Windows
        max_workers = min(10, max(1, multiprocessing.cpu_count() - 2))
        print(f"🔥 [Parallel] Starting {max_workers} parallel worker processes (Executing func: {_worker_process_chunk.__name__})...")
        print(f"📋 [Task Details] All parallel processes will execute the same filtering task:")
        print(f"   1. Molecular Weight Limits: {constraints.get('min_mw', 0)} - {constraints.get('max_mw', 'Inf')}")
        feat_list = '\n'.join([f"      -> {t[2]}" for t in smarts_tuples]) if smarts_tuples else "      -> (None)"
        print(f"   2. Target Structural Features ({len(smarts_tuples)}):\n{feat_list}")
        print(f"📄 [Data] Streaming file: {target_path}")

        global_candidates = []
        global_bg_samples = [] 

        estimated_total = limit if limit else 100_000_000 
        pbar = tqdm(total=estimated_total, unit="mol", desc="Mining")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            reader = pd.read_csv(target_path, sep='\t', header=None, names=['ID', 'SMILES'], chunksize=CHUNK_SIZE, iterator=True)

            lines_read = 0
            try:
                for chunk in reader:
                    if limit and lines_read >= limit: break
                    future = executor.submit(_worker_process_chunk, (chunk, constraints, smarts_tuples))
                    futures.append(future)
                    lines_read += len(chunk)
                    
                    if len(futures) > max_workers * 2:
                        done_indices = []
                        for i, f in enumerate(futures):
                            if f.done():
                                try:
                                    res_cand, res_bg = f.result()
                                    global_candidates.extend(res_cand)
                                    global_bg_samples.extend(res_bg)
                                    pbar.update(CHUNK_SIZE)
                                except Exception as e: print(f"Worker Error: {e}")
                                done_indices.append(i)
                        futures = [f for i, f in enumerate(futures) if i not in done_indices]
                        if len(global_candidates) > 5000:
                            temp_df = pd.DataFrame(global_candidates)
                            temp_df = temp_df.sort_values(by='Total_Score', ascending=False).head(2000)
                            global_candidates = temp_df.to_dict('records')

                for f in as_completed(futures):
                    try:
                        res_cand, res_bg = f.result()
                        global_candidates.extend(res_cand)
                        global_bg_samples.extend(res_bg)
                        pbar.update(CHUNK_SIZE)
                    except Exception as e: print(f"Worker Error: {e}")

            except StopIteration: pass
            except Exception as e: print(f"❌ File streaming interrupted: {e}")
            finally: pbar.close()

        print(f"\n✅ Scan completed. Found {len(global_candidates)} potential candidate molecules.")
        if not global_candidates: return []

        df_final = pd.DataFrame(global_candidates)
        df_final = df_final.sort_values(by=['Total_Score', 'MW'], ascending=[False, True])
        df_save = df_final.head(100).copy()
        df_save['Selection_Reason'] = df_save.apply(lambda row: f"Score: {row['Total_Score']} | Hits: {row['Matched_Details']} | MW: {row['MW']:.1f}", axis=1)

        try:
            df_bg = pd.DataFrame(global_bg_samples)
            if not df_bg.empty:
                bg_pca, top_pca = self._compute_chemical_space(df_bg, df_save.head(20))
                timestamp = int(time.time())
                self._plot_chemical_space_trajectory(bg_pca, top_pca, save_name=f"chem_space_{timestamp}.png")
            else:
                print("⚠️ [Vis] Skipping plot: No background molecules sampled.")
        except Exception as e:
            print(f"⚠️ Plotting Error: {e}")

        timestamp = int(time.time())
        save_path = os.path.join(self.run_dir, f"scout_candidates_{timestamp}.csv")
        df_save.to_csv(save_path, index=False)
        print(f"💾 Results saved to: {save_path}")
        if not df_save.empty:
            print(f"🏆 Top scoring molecule: {df_save.iloc[0]['SMILES']} (Score: {df_save.iloc[0]['Total_Score']})")

        return df_save.to_dict(orient='records')