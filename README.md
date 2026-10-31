# Telco Customer Churn Prediction — Final Data Science Capstone

**Prepared by:** Akash Yadav  
**Problem:** Binary classification — predict whether a telecom customer churns (`Churn=Yes`).

## Dataset
IBM Telco Customer Churn sample dataset, containing 7,043 customer rows and 21 columns. The project uses the supplied CSV file `Telco-Customer-Churn.csv`. The `customerID` is excluded from predictors because it is an identifier, not a behavioural feature.

## Reproducible workflow
1. Convert `TotalCharges` to numeric. Blank strings become missing values (11 rows in the supplied file).
2. Split data into 80% train and 20% test using stratification and `random_state=42`.
3. Fit preprocessing only on training data through a scikit-learn Pipeline: median imputation and scaling for numeric variables; most-frequent imputation and one-hot encoding for categorical variables.
4. Compare Logistic Regression, Decision Tree, Random Forest and Gradient Boosting.
5. Report Accuracy, Precision, Recall, F1, ROC-AUC and PR-AUC on the same held-out test set.

## Test results
See `model_comparison.csv` for exact values. In this run, Decision Tree achieved churn recall 0.794 and F1 0.619. Gradient Boosting had the highest accuracy (0.806) and ROC-AUC (0.843). The Decision Tree is selected for the retention-focused demonstration because recall is prioritized, not because it wins every metric.

## Run
```bash
python -m pip install -r requirements.txt
python main.py
```
Keep `Telco-Customer-Churn.csv` in the same folder as `main.py`.

## Important limitations
- This is a historical sample dataset; model performance may not transfer to a current telecom business.
- The dataset is observational. Feature relationships are associations, not proof of causes.
- The 0.5 classification threshold is a default; a real retention campaign should tune it using campaign capacity and the relative cost of false positives and missed churners.
- Recall-focused models can generate many false alarms. A human review and controlled pilot are recommended before contacting customers.
- The dataset does not include intervention costs, customer lifetime value or the outcomes of retention offers.

## Project files
- `main.py`: end-to-end training and evaluation
- `analysis.ipynb`: exploratory and modelling notebook
- `model_comparison.csv`: test metrics
- `decision_tree_churn_pipeline.joblib`: fitted selected pipeline
- `visualizations/`: charts used in the report
- `Akash_Yadav_Task6_Telco_Churn_Capstone_Report.docx`: full report

Dataset source attribution: IBM Telco Customer Churn sample repository, https://github.com/IBM/telco-customer-churn-on-icp4d
