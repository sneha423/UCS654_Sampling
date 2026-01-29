# Imbalanced Credit Card Sampling Assignment

This project explores how different probability sampling techniques affect the performance of machine learning models on a highly imbalanced credit card dataset.

## Objectives

- Handle a highly imbalanced binary classification dataset.
- Apply five probability sampling techniques:
  - Simple random sampling
  - Stratified sampling
  - Cluster sampling
  - Systematic sampling
  - K-fold cross-validation
- Train five ML models on each sampled dataset.
- Detect overfitting (train vs test performance).
- For each sampling technique, identify the best model by accuracy and summarize results in a final table.

## Dataset

- Source: `Creditcard_data.csv` (provided in the assignment GitHub link).
- Target variable: fraud / non‑fraud indicator (last column, binary).
- The original class distribution is highly imbalanced.

Place `Creditcard_data.csv` in the project root, next to `main.py`.

## Project Structure

```text
.
├── main.py                 # Orchestrates full pipeline
├── data_loader.py          # Loading, SMOTE balancing, scaling, splits
├── sampling_techniques.py  # 5 probability sampling methods
├── models.py               # 5 ML models + evaluation & overfitting check
├── requirements.txt        # Python dependencies
└── results.csv             # Generated summary table (output)
```

Methods
Sampling Techniques
Simple random sampling – uniform random subset of the (balanced) training data.

Stratified sampling – preserves class proportions in the sample.

Cluster sampling – partitions data into clusters, then samples selected clusters.

Systematic sampling – selects every k‑th record after a random start.

K-fold cross-validation – 5 folds; models evaluated by mean test accuracy over folds.

Machine Learning Models
Five supervised models are used (M1–M5), for example:

Logistic Regression

Random Forest

Support Vector Machine

XGBoost

LightGBM

All models are configured with appropriate class balancing options and regularization where relevant.

Overfitting Detection
For each model and sample:

Compute train accuracy and test accuracy.

Generalization gap = train accuracy − test accuracy.

Overfitting status:

Low (gap < 0.08)

Moderate (0.08–0.15)

High (> 0.15)

How to Run
bash

# 1. Create and activate virtual environment (optional but recommended)

python -m venv venv

# Windows: venv\Scripts\activate

# Linux/Mac: source venv/bin/activate

# 2. Install dependencies

pip install -r requirements.txt

# 3. Ensure Creditcard_data.csv is in the project root

# 4. Run the pipeline

python main.py
On completion, results.csv will contain one row per sampling technique.

Output
The main deliverable is a table with, for each sampling technique:

Sample_ID

Sampling_Technique

Best_Model

Accuracy (test accuracy of best model)

Generalization_Gap

Overfitting_Status
This table is saved as results.csv in the project root.
