# CIT-ANFIS: Electric Load Forecasting

This repository contains the source code for the paper submitted to *Nature Communications*.

The **cTSK** framework combines causal discovery (WGCNA + FCI) with interpretable fuzzy neural networks to analyze gene regulatory networks.

## 📂 Repository Structure
CIT-ANFIS/  
├── Step1_Master_Baseline.py       # Benchmark comparison  
├── Step2_Ablation.py              # Ablation study  
├── Step3_Hyperparam_Sensitivity.py # Hyperparameter analysis  
├── Step4_Robustness.py            # Robustness testing  
├── Step5_Visualization.py         # Generate all figures (paper_figures/)   
├── dataprepare.py                 # Data loading & preprocessing    
├── dl_models/                     # Deep learning models  
├── tanfis_lib/                    # Standard ANFIS  
├── xganfis/                       # CIT-ANFIS core library  
├── ISO-NE (2023-2024).csv    # New England ISO dataset  
├── NEW-Malaysia.csv          # Malaysia dataset  
└── ods001.csv                # Belgium grid data  

## 🚀 Usage
# Basic dependencies
Python >= 3.8  
torch >= 1.10  
scikit-learn  
pandas >= 1.3  
numpy >= 1.21  
matplotlib >= 3.5  
seaborn  
xgboost  
holidays  # For holiday feature extraction  
tqdm      # Progress bar display  

### Option 1: Reproduce Figures (Quick Start)
If you only want to reproduce the figures and results shown in the paper:
python Step5_Visualization.py  
This will generate all publication-quality figures in the paper_figures/ directory using pre-computed results.

### Option 2: Retrain Model (Full Workflow)
To retrain all models from scratch and regenerate all results, run the scripts in the following order:
1. Benchmark Model Comparison
python Step1_Master_Baseline.py

2. Ablation Study
python Step2_Ablation.py

3. Hyperparameter Sensitivity Analysis
python Step3_Hyperparam_Sensitivity.py

4. Robustness Testing
python Step4_Robustness.py

5. Generate All Figures
python Step5_Visualization.py
