# Data Directory

This directory contains the necessary data files for the CD-LPL Autonomous Discovery System. 

## Included Files (Version Controlled):
- `EM_feature_importance.csv` / `Life_feature_importance.csv`: Feature importances corresponding to the XGBoost models.
- `EM_shap_summary_plot.png` / `Life_shap_summary_plot.png`: SHAP summary plots used by the Planner agent.
- `models/`: Pre-trained model weights (`.pkl`) and feature definitions (`.json`).

## Files to be Downloaded Manually (Not in Git):
Due to GitHub's file size limits, the following large datasets are excluded from this repository. Before running the system, please ensure you place them in this folder.

1. **`CID-SMILES`** (approx. 8.7 GB): The raw PubChem CID and SMILES database for the Molecule Scout. You can obtain a subset or the full version from [PubChem FTP](https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/), or use the Zenodo link provided for this paper's specific snapshot.
2. **`Total_data_em.xlsx`** & **`Total_data_life.xlsx`**: The raw experimental training data used to generate the reference points and run Deep Analysis.

**Note**: Update `config/config.yaml` to match the exact filenames of any data instances you download.
