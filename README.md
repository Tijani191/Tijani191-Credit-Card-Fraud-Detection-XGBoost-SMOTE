Credit Card Fraud Detection
XGBoost + SMOTE 

A complete end-to-end machine learning pipeline for detecting fraudulent credit card transactions. Covers exploratory data analysis, class imbalance handling with SMOTE/ADASYN, XGBoost classification, model evaluation.

Results
Metric	Score
AUC-ROC	0.9723
F1-Score (Fraud)	0.6631
Recall (Fraud)	0.83
Precision (Fraud)	0.55

Note: Evaluated on a held-out test set that was never balanced — ensuring realistic performance estimates



Methodology
1. Exploratory Data Analysis
•	Dataset: Kaggle Credit Card Fraud Detection — 284,807 transactions, 492 fraudulent (0.17%)
•	Features: Time, Amount, and 28 PCA-transformed components V1-V28
•	No missing values; all features are numerical
2. Data Splitting (Pre-balancing)
•	70/30 stratified train/test split to preserve class ratio
•	Important: balancing is applied only to the training set
3. Class Imbalance — SMOTE & ADASYN
•	SMOTE (Synthetic Minority Oversampling Technique): generates synthetic fraud samples using k-nearest neighbours (k=5), bringing both classes to 199,020 samples
•	ADASYN (Adaptive Synthetic Sampling): alternative that weights synthetic samples by classification difficulty.

4. XGBoost Classifier

n_estimators      = 100
max_depth         = 6
learning_rate     = 0.1
subsample         = 0.8
colsample_bytree  = 0.8
early_stopping_rounds = 10
eval_metric       = logloss

5. Evaluation
•	Primary metrics: F1-Score and AUC-ROC (accuracy is misleading on imbalanced data)
•	ROC Curve and Precision-Recall Curve plotted and saved

Getting Started
Prerequisites
•	Python 3.9+
•	pip
Installation

cd fraud-detection
pip install -r requirements.txt

Dataset
Download creditcard.csv from Kaggle (kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root. It is excluded from version control due to its size (~150 MB).
Run the Notebook

jupyter notebook notebooks/Credit_Card_Fraud_Detection.ipynb


Predicts whether a transaction is fraudulent. Returns prediction (0/1), fraud probability, and a human-readable status.
Request body (all 30 fields required):

{
  "Time": 406.0,
  "V1": -2.3122, "V2": 1.9519, "V3": -1.6098,
  "V4":  3.9979, "V5": -0.5220, "V6": -1.4265,
  ...
  "V28": -0.1104, "Amount": 149.62
}

Response:

{
  "prediction": 1,
  "fraud_probability": 0.9732,
  "status": "Fraudulent"
}
Requirements
Package	Purpose
pandas, numpy	Data manipulation
scikit-learn	Train/test split, metrics
imbalanced-learn	SMOTE, ADASYN
xgboost	Gradient boosted classifier
matplotlib, seaborn	Visualisation

Important Notes
•	Do not commit creditcard.csv — it is ~150 MB and listed in .gitignore
•	Do not commit *.joblib / *.pkl model files — regenerate by running the notebook
•	Evaluate with F1 and AUC-ROC, not accuracy, on imbalanced datasets
•	SMOTE is applied only to training data; the test set remains real and unbalanced

Licence
MIT — see LICENSE for details.

Credit Card Fraud Detection — XGBoost + SMOTE + FastAPI  |  github.com/Tijani191
