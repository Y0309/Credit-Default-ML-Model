# Credit Default ML Model

This project predicts whether a customer will default on their loan using machine learning models (Logistic Regression and Random Forest) in Python. 

---

## Objective  
The goal of this project was to build a predictive model that identifies high-risk customers and explains which factors contribute most to default behavior.  
The project combines data cleaning, feature engineering, and model evaluation to simulate how banks assess loan applicants.  

---

## Dataset  
**Credit Card Default Prediction Dataset**  

Files used:
- `application_record.csv` — customer demographics, employment, income, and family details  
- `credit_record.csv` — monthly repayment and account status history  

 Dataset can be downloaded directly from Kaggle: [Credit Card Default Prediction Dataset](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction)

---

## Process  
| Step | Description |
|------|--------------|
| 1. Data Cleaning | Merged application and credit data, handled missing values, filled occupation type with “Unknown.” |
| 2. Encoding | Converted categorical features (e.g., gender, car ownership, income type) into numeric form. |
| 3. Label Creation | Marked customers as *risky* if their credit history showed any STATUS = 2–5. |
| 4. Model Training | Built Logistic Regression (baseline) and Random Forest (improved recall). |
| 5. Evaluation | Compared accuracy, recall, and precision to assess how well models detect risky clients. |

---

## Results  

| Model | Accuracy | Recall (Risky) | Key Observation |
|--------|-----------|----------------|----------------|
| Logistic Regression | 99.9% | 0% | Model predicted almost all customers as “Safe” because data was highly imbalanced. |
| Random Forest (Balanced) | 99.8% | **46%** | Better detection of risky customers after removing data leakage from the ID column. |

**Top Factors Affecting Credit Risk**
1. Age (younger customers more likely to default)  
2. Employment length (shorter = higher risk)  
3. Income level (lower = higher risk)  
4. Family size and ownership indicators  

---

## Key Insights  
- Removing the `ID` column prevented data leakage and improved recall from 0 to 46%.  
- Random Forest handled class imbalance better than Logistic Regression.  
- The most important factors for predicting default are age, employment stability, and income.  
- Financially stable, older customers with longer employment are less likely to default payment.

---

## Tools and Skills  
- **Python** (Pandas, scikit-learn, Matplotlib)  
- **Data Cleaning & Feature Engineering**  
- **Model Training & Evaluation**
- **Feature Importance Visualization**  
- **Interpretation of Business Insights**

---
