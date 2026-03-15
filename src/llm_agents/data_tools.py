import pandas as pd
import numpy as np
import os
import yaml
import random

# =====================================================================
# [Configuration Area] Column Name Mappings
# =====================================================================
COLUMN_MAPPING = {
    "EMISSION_PEAK": 'Emission Wavelength (nm)',       
    "LIFETIME_MS":   'Afterglow Lifetime (ms)',      
    "FEATURE_COL":   'Feature',       
    "IMPORTANCE_COL":'Importance',    
    "ID_COL":        'Experiment_ID'  
}

from src.utils import ConfigLoader

class DataToolkit:
    def __init__(self):
        self.config_loader = ConfigLoader.get_instance()
        self.config = self.config_loader.config
        
        self.data_store = {}
        self._load_data()

    def _load_data(self):
        """Load data resources"""
        print(">>> [DataToolkit] Loading data resources...")
        
        def read_file(key_name):
            path = self.config_loader.get_data_path(key_name)
            if not path or not os.path.exists(path): 
                if path: print(f"⚠️ File does not exist: {path}")
                return None
            for encoding in ['utf-8', 'gbk', 'gb18030']:
                try:
                    if path.endswith('.csv'): df = pd.read_csv(path, encoding=encoding)
                    else: df = pd.read_excel(path)
                    df.columns = df.columns.astype(str).str.strip()
                    return df
                except: continue
            return None

        self.data_store['life_imp'] = read_file('life_importance')
        self.data_store['em_imp']   = read_file('em_importance')
        self.data_store['exp']      = read_file('experiments')
        
        # Preprocessing
        exp_df = self.data_store.get('exp')
        if exp_df is not None:
            em_col = COLUMN_MAPPING["EMISSION_PEAK"]
            life_col = COLUMN_MAPPING["LIFETIME_MS"]
            if COLUMN_MAPPING["ID_COL"] not in exp_df.columns:
                exp_df[COLUMN_MAPPING["ID_COL"]] = [f"Exp_{i}" for i in range(len(exp_df))]
            if em_col in exp_df.columns: exp_df[em_col] = pd.to_numeric(exp_df[em_col], errors='coerce')
            if life_col in exp_df.columns: exp_df[life_col] = pd.to_numeric(exp_df[life_col], errors='coerce')
            if em_col in exp_df.columns and life_col in exp_df.columns:
                exp_df.dropna(subset=[em_col, life_col], how='all', inplace=True)
            self.data_store['exp'] = exp_df
        
        self.data_store['em_shap_img'] = self.config_loader.get_data_path('em_shap')
        self.data_store['life_shap_img'] = self.config_loader.get_data_path('life_shap')

        print(">>> [DataToolkit] Load complete.")

    def get_experiment_data_with_sampling(self, target_type: str, min_val: float = None, max_val: float = None):
        """Get experimental data (sampling logic unchanged)"""
        df = self.data_store.get('exp')
        if df is None: return {"error": "No experimental data"}

        em_col = COLUMN_MAPPING["EMISSION_PEAK"]
        life_col = COLUMN_MAPPING["LIFETIME_MS"]
        id_col = COLUMN_MAPPING["ID_COL"]
        col_name = em_col if target_type == 'emission' else life_col
        
        if col_name not in df.columns: return {"error": f"Column name '{col_name}' does not exist"}

        df_clean = df.dropna(subset=[col_name]).sort_values(by=col_name)
        if df_clean.empty: return {"warning": "No valid data for this column"}

        final_rows = []
        
        # Emission: Global 30nm sampling + Local ±50nm sampling
        if target_type == 'emission':
            # Global
            global_min, global_max = df_clean[col_name].min(), df_clean[col_name].max()
            curr = global_min
            while curr < global_max:
                chunk = df_clean[(df_clean[col_name] >= curr) & (df_clean[col_name] < (curr + 30))]
                if not chunk.empty:
                    # Priority: take up to 5 if available
                    n_samples = min(5, len(chunk)) 
                    final_rows.append(chunk.sample(n_samples))
                curr += 30
            # Local
            center = min_val if min_val is not None else (global_min + global_max) / 2
            chunk_focus = df_clean[(df_clean[col_name] >= center - 50) & (df_clean[col_name] <= center + 50)]
            if not chunk_focus.empty:
                final_rows.append(chunk_focus.sample(min(5, len(chunk_focus))))

        # Lifetime: Optimized sampling strategy (Top 20 Best + Gradient Sampling)
        elif target_type == 'lifetime':
            # 1. Selection range
            lower = min_val if min_val is not None else 0
            upper = max_val if max_val is not None else float('inf')
            target_df = df_clean[(df_clean[col_name] >= lower) & (df_clean[col_name] <= upper)]
            
            # Sort by lifetime descending
            target_df = target_df.sort_values(by=col_name, ascending=False)
            
            if target_df.empty:
                return {"warning": "No data after filtering"}

            # 2. Get Top 20 (Best Performance)
            top_20 = target_df.head(20)
            final_rows.append(top_20)
            
            # 3. Gradient sampling for the remaining data
            remaining_df = target_df.iloc[20:].copy()
            
            if not remaining_df.empty:
                # Divide remaining data into 10 gradients (qcut)
                try:
                    # Handle case where qcut creates duplicate edges
                    num_bins = min(10, len(remaining_df))
                    if num_bins > 0:
                        remaining_df['bin'] = pd.qcut(remaining_df[col_name], q=num_bins, duplicates='drop')
                        
                        # Take 4 rows per gradient
                        for bin_label, group in remaining_df.groupby('bin'):
                            n_sample = min(4, len(group))
                            final_rows.append(group.sample(n_sample))
                except Exception as e:
                    # If binning fails, fallback to random sampling
                    print(f"Gradient sampling failed: {e}, falling back to random sample")
                    fallback_n = min(40, len(remaining_df))
                    final_rows.append(remaining_df.sample(fallback_n))
            
            print(f"✅ Lifetime Sampling: Top 20 + Gradient Samples (Total chunks: {len(final_rows)})")

        if not final_rows: return {"warning": "No data matching sampling conditions found"}

        final_df = pd.concat(final_rows).drop_duplicates(subset=[id_col])
        if 'bin' in final_df.columns:
            final_df = final_df.drop(columns=['bin'])
        
        clean_records = []
        for r in final_df.to_dict(orient='records'):
            clean_r = {}
            for k, v in r.items():
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)): clean_r[k] = None
                else: clean_r[k] = v
            clean_records.append(clean_r)

        return {
            "summary": f"Target: {target_type}",
            "count": len(final_df),
            "data": clean_records
        }

    def query_feature_importance(self, target: str, top_n: int = 8):
        """
        [Modification] default top_n = 5, retrieve top 5 important features.
        """
        feat_col = COLUMN_MAPPING["FEATURE_COL"]
        imp_col = COLUMN_MAPPING["IMPORTANCE_COL"]
        
        # 1. CSV
        df = self.data_store.get('life_imp') if target == 'lifetime' else self.data_store.get('em_imp')
        csv_result = {}
        
        if df is not None and feat_col in df.columns and imp_col in df.columns:
            # Sorting
            df_sorted = df.sort_values(by=imp_col, ascending=False)
            
            # Extract Bit features (Top N)
            bits = [str(x) for x in df_sorted[feat_col] if str(x).startswith("Bit_")]
            top_bits = bits[:top_n]
            
            # Extract Non-Bit features (Top N)
            others = [str(x) for x in df_sorted[feat_col] if not str(x).startswith("Bit_")]
            top_others = others[:top_n]
            
            csv_result = {
                "top_bits_from_csv": top_bits,
                "top_conditions_from_csv": top_others
            }
        else:
            csv_result = {"warning": "CSV data missing."}

        # 2. Image
        img_rel = self.data_store.get('life_shap_img') if target == 'lifetime' else self.data_store.get('em_shap_img')
        abs_path = os.path.abspath(img_rel) if img_rel and os.path.exists(img_rel) else None

        return {
            "target": target,
            **csv_result,
            "shap_image_available": bool(abs_path),
            "shap_image_path": abs_path
        }