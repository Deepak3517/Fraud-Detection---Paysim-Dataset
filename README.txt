README
Fraud Detection — PaySim Dataset

Problem Statement
Detect fraudulent transactions in a highly imbalanced financial dataset (PaySim) containing 6.3 million transactions with only 8,213 fraud cases (0.13%).

Dataset Features
 •step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
Key Challenge — Class Imbalance
Only 0.13% transactions were fraud. A naive model would predict everything as non-fraud and still show 99.8% accuracy — but catch zero fraud cases.
Solution: Used scale_pos_weight = non_fraud_count / fraud_count = 773 This means the model receives 773x more penalty for missing a fraud case during training.

Approach
 •Train-test split with stratify=y to maintain fraud ratio in both sets
 •XGBoost Classifier with scale_pos_weight to handle imbalance
 •SHAP values for model explainability

Results
  Metric	                Score
  ROC-AUC	                0.999
  Recall (Fraud)	        1.00
  Precision (Fraud)     	0.06
Recall = 1.00 means every single fraud transaction was detected.
Low precision is acceptable here — missing a fraud costs far more than a false alarm.

Key Insight — SHAP Analysis
newbalanceOrig was the strongest fraud signal — when a sender's balance gets completely wiped out after a transaction, it is the most reliable indicator of fraud.
This pattern was not manually coded — XGBoost discovered it automatically from data.

SQL vs ML Comparison
Approach	Method
SQL Rules	Manually defined conditions — amount > X or balance = 0
XGBoost	        Automatically learned complex patterns from 6.3M transactions

ML outperformed rule-based detection by combining multiple weak signals that SQL rules would miss individually.

Tech Stack
Python, XGBoost, SHAP, Pandas, Scikit-learn
