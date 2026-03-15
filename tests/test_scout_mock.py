# test_scout_mock.py

import os
import sys
import pandas as pd
import unittest
from unittest.mock import MagicMock, patch

# Add current directory to path, to allow src import
sys.path.append(os.getcwd())

# --- Mock src.utils dependency ---
# Must mock before importing scout, or it will throw error if utils.py misses locally
sys.modules['src.utils'] = MagicMock()
from src.utils import ConfigLoader, get_run_dir, get_prompt, get_llm_client

# Now we can safely import scout
from src.llm_agents.scout import ScoutAgent

class TestScoutAgent(unittest.TestCase):
    
    def setUp(self):
        """Pre-test setup: create dummy files and directories"""
        print("\n[Test] Setting up environment...")
        self.test_dir = "test_results"
        os.makedirs(self.test_dir, exist_ok=True)
        
        # 1. Create a mock molecules data file (Tab separated)
        self.data_path = "mock_cid_smiles.tsv"
        mock_data = [
            [1, "c1ccccc1"],                  # Benzene (MW ~78)
            [2, "CC(=O)O"],                   # Acetic Acid (MW ~60)
            [3, "c1ccccc1C(=O)O"],            # Benzoic Acid (Contains both, should be Top Candidate)
            [4, "CCO"],                       # Ethanol (No features)
            [5, "c1ccccc1N"],                 # Aniline
            [6, "O=C(O)c1ccccc1O"],           # Salicylic Acid
        ]
        
        # --- [Key Modification] Hugely inflate background noise data amount ---
        # Before we only had 100 entries, a 0.5% sampling rate wouldn't catch anything.
        # Now increased to 5000, expect to sample ~25 background points to be enough to plot.
        print("[Test] Generating 5000 mock molecules to ensure background sampling...")
        for i in range(7, 5007):
            # Generate simple carbon chains as background noise
            # To prevent RDKit errors, generate valid SMILES: C, CC, CCC...
            chain_len = (i % 20) + 1
            mock_data.append([i, "C" * chain_len]) 
            
        pd.DataFrame(mock_data).to_csv(self.data_path, sep='\t', header=False, index=False)

    def tearDown(self):
        """Post-test cleanup"""
        # Manual cleanup if necessary upon successful tests
        # if os.path.exists(self.data_path): os.remove(self.data_path)
        pass

    @patch('src.llm_agents.scout.ConfigLoader')
    @patch('src.llm_agents.scout.get_llm_client')
    @patch('src.llm_agents.scout.get_run_dir')
    def test_search_logic(self, mock_get_run_dir, mock_get_client, mock_config_loader):
        """Test core logic - Search & Print Plots"""
        print("[Test] Starting logic verification...")

        # --- 1. Mock Configuration ---
        mock_get_run_dir.return_value = self.test_dir
        
        # Mock LLM Client
        mock_client = MagicMock()
        mock_get_client.return_value = (mock_client, "gpt-4", 0.7)
        
        # Mock ConfigLoader returning mock data directory
        mock_instance = MagicMock()
        mock_instance.get_data_path.return_value = self.data_path
        mock_config_loader.get_instance.return_value = mock_instance

        # --- 2. Initialize Agent ---
        agent = ScoutAgent()
        
        # --- 3. Mock LLM Response (SMARTS string) ---
        # Mock LLM processing to translate Natural lang query to smt
        agent._get_smarts_from_llm = MagicMock(side_effect=lambda desc: 
            "c1ccccc1" if "Benzene Ring" in desc else ("C(=O)O" if "Carboxyl Group" in desc else None)
        )

        # --- 4. Fake Summary Report Compilation ---
        fake_summary = {
            "critical_structures": [
                {"feature_name": "Ring", "chemical_meaning": "Benzene Ring"},
                {"feature_name": "Acid", "chemical_meaning": "Carboxyl Group"}
            ],
            "design_guidelines": {
                # Setup simple Molecular weight checks for filtering
                "structural_rules": ["MW > 10"]
            }
        }

        # --- 5. Start running search query ---
        print("[Test] Running agent.search_molecules...")
        # Since we generated 5k rows, provide sufficiently large Limit
        results = agent.search_molecules(fake_summary, limit=6000)

        # --- 6. Output Valdation Verification Checks ---
        # A. Molecules should be found!
        self.assertTrue(len(results) > 0, "Agent returned empty list!")
        print(f"[Test] Found {len(results)} candidates.")
        
        # Verify if Top 1 returned correctly matched params
        top_mol = results[0]
        # We expect Benzoic acid or Salicylic acid (ID 3 / 6) to receive Highest Score rating (2 Points)
        print(f"[Test] Top Candidate: {top_mol['SMILES']} (Score: {top_mol['Total_Score']})")
        self.assertTrue(top_mol['Total_Score'] >= 1, "Scoring failed")

        # B. Output IO Verification Directory Outputs Validation
        images_dir = os.path.join(self.test_dir, "images")
        self.assertTrue(os.path.exists(images_dir), "Image directory not created")
        
        files = os.listdir(images_dir)
        print(f"[Test] Files generated in {images_dir}: {files}")
        
        # Check generated plot presence (If it created picture!)
        png_files = [f for f in files if f.endswith('.png')]
        self.assertTrue(len(png_files) > 0, 
                        f"No PNG image generated! (Files found: {files}). "
                        "Possible reason: Background sampling yielded 0 molecules due to small dataset.")
        
        # Determine presence of relevant metrics (.csv logic: bg, top, kde files check)
        base_name = os.path.splitext(png_files[0])[0]
        
        csv_bg = f"{base_name}_background_points.csv"
        csv_top = f"{base_name}_top_candidates.csv"
        csv_kde = f"{base_name}_kde_density.csv"
        
        self.assertIn(csv_bg, files, "Missing background points data")
        self.assertIn(csv_top, files, "Missing top candidates data")
        self.assertIn(csv_kde, files, "Missing KDE density data")
        
        print("\n✅ Test Passed: Code logic, plotting, and data saving are correct.")

if __name__ == '__main__':
    unittest.main()