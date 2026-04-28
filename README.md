# Credit Risk, Customer Churn & Car Features Analysis

## 📊 Overview

This repository contains three comprehensive data science projects demonstrating end-to-end machine learning workflows for different business problems:

1. **Credit Risk Prediction** - Predicting loan default risk using customer financial data
2. **Bank Customer Churn Prediction** - Identifying customers likely to leave the bank
3. **Car Features Exploratory Data Analysis** - Analyzing automotive dataset with visualizations

---

## 🚀 Project 1: Credit Risk Prediction

### Problem Statement
Financial institution needs to reduce loan defaults while maximizing approvals by predicting default risk using customer financial and demographic data.

### Key Steps

| Step | Description |
|------|-------------|
| **Data Understanding** | 255,347 rows, 22 features, no missing values |
| **Preprocessing** | Dropped `LoanID`, encoded binary features (Yes→1/No→0), one-hot encoded categoricals |
| **EDA Insights** | Lower income, lower credit score, higher loan amount, unemployment, higher interest rate → higher default risk |
| **Imbalance Handling** | Class imbalance addressed with `class_weight="balanced"` |

### Models Tested

| Model | Accuracy | Recall (Default) | Precision (Default) |
|-------|----------|------------------|---------------------|
| Logistic Regression (baseline) | 89% | 4% | 50% |
| Logistic Regression (balanced) | — | 68% | 22% |
| Decision Tree (threshold 0.3) | — | 68% | 34% |
| **XGBoost (threshold 0.4)** ✅ | — | **70%** | **43%** |

### Final Model: XGBoost
- **Threshold**: 0.4 (balance between recall and precision)
- **Top Features**: Income, EmploymentType, InterestRate, Age, CreditScore

### Business Impact
- Identifies 70% of high-risk customers while limiting false positives
- Enables targeted interventions before default occurs

---

## 🏦 Project 2: Bank Customer Churn Prediction

### Problem Statement
Identify factors contributing to customer churn and build a predictive model to classify customers likely to exit.

### Dataset
- **Rows**: 10,000
- **Features**: 14 (CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Exited)
- **Target**: `Exited` (20% churn rate)

### Feature Engineering
| New Feature | Formula |
|-------------|---------|
| `BalanceSalaryRatio` | Balance / EstimatedSalary |
| `TenureByAge` | Tenure / Age |
| `CreditScoreGivenAge` | CreditScore / Age |

### EDA Insights
- Germany has highest churn proportion relative to customer base
- Female customers churn more than males
- Inactive members have significantly higher churn
- Customers with high balances and older age are more likely to churn

### Best Model: Random Forest

| Metric | Train | Test |
|--------|-------|------|
| Precision (Churn) | 0.88 | 0.45 |
| Recall (Churn) | 0.53 | 0.41 |
| AUC | 0.88 | 0.81 |

### Recommendation
Adjust classification threshold to prioritize recall for identifying at-risk customers for retention programs.

---

## 🚗 Project 3: Car Features EDA

### Objective
Perform exploratory data analysis on car dataset to understand feature relationships and prepare data for modeling.

### Data Processing Steps

1. **Loading & Inspection** - 11,914 rows, 16 columns
2. **Column Dropping** - Removed: Engine Fuel Type, Market Category, Vehicle Style, Popularity, Number of Doors, Vehicle Size
3. **Renaming** - Simplified column names (HP, Cylinders, MPG-H, MPG-C, Price)
4. **Duplicate Removal** - Removed 989 duplicate rows (final: 10,925)
5. **Missing Values** - Dropped nulls (final: 10,827 rows)
6. **Outlier Detection** - Used IQR method, removed ~1,600 outliers (final: 9,191 rows)

### Visualizations

| Chart Type | Purpose |
|------------|---------|
| Box plots | Outlier detection (Price, HP, Cylinders) |
| Heatmap | Feature correlation analysis |
| Scatter plot | HP vs Price relationship |
| Bar chart | Average price by car make |

### Key Correlation Findings
- **Price** strongly correlated with **HP** (0.74) and **Year** (0.59)
- **MPG-C** and **MPG-H** strongly correlated (0.94) - multicollinearity
- **Cylinders** negatively correlated with MPG metrics

---

## 🛠️ Technologies Used

```
Python 3.x
pandas, numpy          # Data manipulation
matplotlib, seaborn    # Visualization
scikit-learn           # ML models (LogisticRegression, RandomForest, SVM)
xgboost, lightgbm      # Gradient boosting
joblib                 # Model serialization
```

## 👥 Author

Saud Ijaz

---

## 📄 License

This project is for educational and portfolio purposes.
