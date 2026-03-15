import os
import sys
import shutil
import json
import joblib
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

# Ensure src module is importable
sys.path.append(os.getcwd())

from src.llm_agents.optimizer import OptimizerAgent
from src.utils import ConfigLoader

# ==========================================
# 1. Define Fake Models and Scalers (Mock Objects)
# ==========================================
class DummyModel:
    """A fake regression model, generating pseudo-random predictions based on input features"""
    def predict(self, X):
        # To make 3D plots look nice, we need to create some "patterns"
        # Assuming X is numpy array or DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Simple simulation: Score depends on 10th column (Temperature) and 2nd column (Ratio)
        # Adds some random noise
        n_samples = X.shape[0]
        # Randomly select a few features to simulate "chemical reaction"
        # Ensure returned values are between 0~100 (Emission) or 0~2000 (Lifetime)
        feature_a = X[:, -5] # Let's assume 5th column from end is temp
        feature_b = X[:, -4] # Let's assume 4th column from end is time
        
        # Generate a surface data with peaks
        base = np.sin(feature_a / 100.0) * np.cos(feature_b / 10.0)
        score = 500 + 200 * base + np.random.normal(0, 10, n_samples)
        return np.abs(score)

class DummyScaler:
    """Fake normalizer, returns as is without scaling"""
    def transform(self, X):
        return X

# ==========================================
# 2. Test Environment Setup Functions
# ==========================================
TEST_DIR = "test_env_optimizer"
DATA_DIR = os.path.join(TEST_DIR, "data")
RUN_DIR = os.path.join(TEST_DIR, "run")

def setup_test_environment():
    """Create temp directory and save fake .pkl model files"""
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(DATA_DIR)
    os.makedirs(RUN_DIR)

    print(f"🛠️  Building test environment: {TEST_DIR}")

    # 1. Save fake Feature Names (2199 dimensions, simulate real scenario)
    feature_names = [f"Bit_{i}" for i in range(2048)]
    feature_names.extend([
        "Step_1_Temperature", "Step_1_Time", "Step1_Carbon_Dots_Dosage", 
        "Step_2_Temperature", "Step 2_Time", "Ratio", 
        "Preparation_Method_Code_1", "Preparation_Method_Code_2",
        "Step_1_Reaction_Code_1", "Step_1_Reaction_Code_2",
        "Pre1_C", "Pre1_H", "Pre1_O", "Pre1_N", 
        "Test_Temperature", "Ex"
    ])
    
    # --- [Modification] Fix filenames to match agent expectations (em_xxx, life_xxx) ---
    with open(os.path.join(DATA_DIR, "em_feature_names.json"), "w") as f:
        json.dump(feature_names, f)
    
    with open(os.path.join(DATA_DIR, "life_feature_names.json"), "w") as f:
        json.dump(feature_names, f)
    # ----------------------------------------------------------------

    # 2. Save fake Model and Scaler (.pkl)
    dummy_model = DummyModel()
    dummy_scaler = DummyScaler()

    joblib.dump(dummy_model, os.path.join(DATA_DIR, "trained_em_model.pkl"))
    joblib.dump(dummy_scaler, os.path.join(DATA_DIR, "em_scaler.pkl"))
    
    joblib.dump(dummy_model, os.path.join(DATA_DIR, "trained_life_model.pkl"))
    joblib.dump(dummy_scaler, os.path.join(DATA_DIR, "life_scaler.pkl"))

    print("✅ Fake model files generated (Mock Models Saved)")

# ==========================================
# 3. Main Test Logic
# ==========================================
def run_test():
    setup_test_environment()

    # --- Patch ConfigLoader & get_run_dir ---
    # We need to hijack path retrieval functions in utils to point to our test folder
    
    with patch('src.utils.ConfigLoader.get_model_path') as mock_get_path, \
         patch('src.utils.get_run_dir') as mock_get_run_dir, \
         patch('src.llm_agents.optimizer.get_run_dir') as mock_agent_run_dir:

        # 1. Set Mock Return Values
        # When code requests 'em_model' path, return the path to the mocked file
        mock_get_path.side_effect = lambda x: os.path.join(DATA_DIR, "trained_em_model.pkl")
        # Set run path to test folder
        mock_get_run_dir.return_value = RUN_DIR
        mock_agent_run_dir.return_value = RUN_DIR

        # 2. Initialize Agent
        print("\n🤖 Init OptimizerAgent...")
        agent = OptimizerAgent()
        
        # Override internal data directory for insurance
        agent.data_dir = DATA_DIR

        # 3. Prepare Test Data Inputs
        summary_report = {
            "target_property": "emission wavelength and lifetime",
            "critical_features": ["Amide", "C=O"]
        }

        candidates = [
            {"Name": "Test_Mol_A", "SMILES": "CC(=O)O"},      # Acetic Acid
            {"Name": "Test_Mol_B", "SMILES": "c1ccccc1N"},    # Aniline
            {"Name": "Test_Mol_C", "SMILES": "NCC(=O)O"}      # Glycine
        ]

        # 4. Run Optimization
        print("🚀 Running optimize()...")
        # To accelerate test speed, N_ITER=300 is hardcoded in agent,
        # Real tests will take a bit because it computes RDKit descriptors
        result = agent.optimize(summary_report, candidates)

        # 5. Verify Results
        print("\n" + "="*40)
        print("🧪 Test Result Verification")
        print("="*40)

        # Check if CSV is generated
        csv_path = os.path.join(RUN_DIR, "logs", "Total_Model_Input_Features.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"✅ CSV file created successfully: {csv_path}")
            print(f"   -> Data shape: {df.shape}")
        else:
            print(f"❌ CSV file not found!")

        # Check if Images are generated (High-Resolution Style)
        img_dir = os.path.join(RUN_DIR, "images")
        images = [f for f in os.listdir(img_dir) if f.endswith(".png")]
        if images:
            print(f"✅ Created {len(images)} 3D plots successfully:")
            for img in images:
                print(f"   -> {img}")
            print("🎨 Check generated figures styles at test_env_optimizer/run/images !")
        else:
            print("⚠️ Warning: No images generated. This could be because random feature variance is too small, filtered out by _batch_plot_all_surfaces' std checker.")
            print("   (This is possible when using DummyModel, just retry or adjust DummyModel variance)")

        # Output Recommendations Yielded
        print("\n📋 Final Output Content (Top Recommendation):")
        print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    try:
        run_test()
    except Exception as e:
        print(f"\n❌ Test encounted exception: {e}")
        import traceback
        traceback.print_exc()