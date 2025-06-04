# Customer Churn Prediction using Machine Learning

## Table of Contents
1.  [Introduction](#introduction)
2.  [Problem Statement](#problem-statement)
3.  [Dataset](#dataset)
4.  [Project Workflow](#project-workflow)
5.  [Exploratory Data Analysis (EDA) Highlights](#exploratory-data-analysis-eda-highlights)
6.  [Modeling and Evaluation](#modeling-and-evaluation)
7.  [Feature Importance](#feature-importance)
8.  [Hyperparameter Tuning](#hyperparameter-tuning)
9.  [Key Results & Findings](#key-results--findings)
10. [Business Implications & Recommendations](#business-implications--recommendations)
11. [Technologies Used](#technologies-used)
12. [How to Run](#how-to-run)
13. [Future Work](#future-work)

---

## 1. Introduction
This project focuses on predicting customer churn for a telecommunications company. Customer churn, the phenomenon where customers stop doing business with a company, is a critical concern for businesses as acquiring new customers is often more expensive than retaining existing ones. By identifying customers who are likely to churn, businesses can take proactive measures to retain them. This project aims to build such a predictive model and derive actionable insights.

---

## 2. Problem Statement
The primary objectives of this project are:
* To develop and evaluate various machine learning models to accurately predict whether a customer will churn.
* To identify the key demographic, account, and service-related factors that are strong indicators of customer churn.
* To provide data-driven insights that can help a telecommunications business formulate targeted customer retention strategies.

---

## 3. Dataset
The dataset used for this project is the **"Telco Customer Churn"** dataset, publicly available on Kaggle.
* **Source:** [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **Content:** The dataset includes information about a fictional telecom company that provided home phone and Internet services to 7043 customers. It indicates which customers have left ("Churn": Yes) or stayed ("Churn": No). The data includes:
    * Customer demographics (gender, Senior Citizen, Partner, Dependents).
    * Account information (tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges).
    * Services subscribed to (PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies).

---

## 4. Project Workflow
The project followed a structured data science lifecycle:

1.  **Data Loading and Initial Exploration:** Loaded the dataset using Pandas and performed an initial inspection of its structure, data types (`df.info()`, `df.head()`), and basic statistics (`df.describe()`).
2.  **Data Cleaning and Preprocessing:**
    * Handled incorrect data types: `TotalCharges` was converted from object to numeric.
    * Addressed missing values that arose in `TotalCharges` after conversion (imputed with 0 for customers with 0 tenure, otherwise with the median [or as per your specific imputation choice]).
    * Converted the target variable `Churn` from "Yes"/"No" to binary (1/0).
3.  **Exploratory Data Analysis (EDA):**
    * Analyzed the distribution of the target variable (`Churn`) to understand class balance.
    * Visualized distributions of numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) and their relationship with churn.
    * Investigated relationships between categorical features and churn using count plots and grouped analysis.
4.  **Feature Engineering & Scaling:**
    * Dropped the `customerID` column as it's a unique identifier not useful for prediction.
    * Applied One-Hot Encoding to convert categorical features into a numerical format using `pd.get_dummies(drop_first=True)`.
    * Scaled numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) using `StandardScaler` to prepare data for models sensitive to feature magnitudes.
5.  **Model Building and Training:**
    * Split the preprocessed data into training (80%) and testing (20%) sets using `train_test_split` with stratification by the target variable `Churn`.
    * Trained and evaluated several classification models:
        * Logistic Regression
        * Decision Tree Classifier (Default Parameters)
        * Random Forest Classifier (Default Parameters)
6.  **Model Evaluation:**
    * Evaluated models on the unseen test set using key metrics: Accuracy, Precision (for churn class), Recall (for churn class), F1-Score (for churn class), and ROC AUC Score.
    * Analyzed Confusion Matrices and Classification Reports for detailed performance insights.
7.  **Feature Importance Analysis:**
    * Extracted and visualized feature importances from the Random Forest model (default parameters) to identify the most influential features in predicting churn.
8.  **Hyperparameter Tuning:**
    * Performed hyperparameter tuning for the Random Forest model using `GridSearchCV` with 3-fold cross-validation, optimizing for the `roc_auc` scoring metric, to enhance its predictive performance.

---

## 5. Exploratory Data Analysis (EDA) Highlights
Key insights derived from the EDA phase pointed towards several factors influencing churn:
* **Contract Type:** Customers on **month-to-month contracts** exhibited a significantly higher propensity to churn compared to those on one-year or two-year contracts.
* **Payment Method:** The use of **Electronic checks** for payment was strongly associated with higher churn rates. Customers using automatic payment methods (bank transfer or credit card) showed lower churn.
* **Tenure:** Customers with **shorter tenures** (newer customers) were more likely to churn, indicating the importance of early customer lifecycle management.
* **Monthly Charges:** While complex, there appeared to be segments where **higher Monthly Charges** correlated with increased churn, especially if not coupled with long-term contracts or valuable services.
* **Internet Service Type:** Customers with **Fiber optic** internet service showed a different churn pattern, which might be linked to service satisfaction, price sensitivity or contract terms associated with this service type.
* **Value-Added Services:** Customers not availing services like **Online Security**, **Online Backup**, and **Tech Support** tended to churn more frequently.

---

## 6. Modeling and Evaluation
Four models were trained and evaluated. The performance on the test set is summarized below:

| Model                     | Accuracy | Precision (Churn=1) | Recall (Churn=1) | F1-Score (Churn=1) | ROC AUC |
|---------------------------|----------|---------------------|------------------|--------------------|---------|
| Logistic Regression       | 0.8041   | 0.6541              | 0.5561           | 0.6012             | 0.8425  |
| Decision Tree (Default)   | 0.7253   | 0.4825              | 0.4786           | 0.4805             | 0.6460  |
| Random Forest (Default)   | 0.7850   | 0.6187              | 0.4947           | 0.5498             | 0.8248  |
| **Random Forest (Tuned)** | **0.8020** | **0.6655** | **0.5107** | **0.5779** | **0.8424** |

* The **Tuned Random Forest** and **Logistic Regression** models emerged as the top performers.
* Logistic Regression demonstrated the highest Recall (0.5561) for the churn class among the models.
* The Tuned Random Forest significantly improved upon its default version and achieved a ROC AUC score (0.8424) almost identical to Logistic Regression (0.8425), with slightly better Precision for the churn class (0.6655 vs 0.6541).
* The default Decision Tree showed comparatively weaker performance.

---

## 7. Feature Importance
The feature importance analysis from the initial Random Forest model (default parameters) highlighted the following as key drivers of churn:

1.  **`TotalCharges`** (Importance Score: ~0.191)
2.  **`tenure`** (Importance Score: ~0.175)
3.  **`MonthlyCharges`** (Importance Score: ~0.169)
4.  **`PaymentMethod_Electronic check`** (Importance Score: ~0.039)
5.  **`InternetService_Fiber optic`** (Importance Score: ~0.037)
6.  **`Contract_Two year`** (Importance Score: ~0.030 - likely its absence indicating shorter contracts)
7.  `OnlineSecurity_Yes`
8.  `gender_Male`
9.  `PaperlessBilling_Yes`
10. `TechSupport_Yes`

These features confirm that financial aspects, customer relationship length, payment behavior, and specific service configurations are crucial in predicting churn.

---

## 8. Hyperparameter Tuning
Hyperparameter tuning was performed on the Random Forest model using `GridSearchCV` (3-fold CV, optimizing for `roc_auc`).
* **Best Parameters Found:** `{'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 200}`
* This tuning process resulted in a more robust Random Forest model, improving all key evaluation metrics and making it highly competitive with Logistic Regression.

---

## 9. Key Results & Findings
* Machine learning models, particularly Logistic Regression and a Tuned Random Forest, can effectively predict customer churn with high discriminatory power (ROC AUC ≈ 0.84).
* Financial commitment and history (TotalCharges, MonthlyCharges, tenure) are the most dominant predictors.
* Service engagement (e.g., contract type, internet service type, uptake of online security/tech support) and payment behavior (e.g., use of electronic checks) significantly influence churn likelihood.
* While the Tuned Random Forest offered excellent overall performance and insights via feature importances, Logistic Regression provided slightly higher recall for the churn class, which can be critical for minimizing missed churn cases. `[State your final preferred model and brief justification here if you have one, e.g., "For this project, Logistic Regression might be preferred if maximizing the identification of at-risk customers (Recall) is the primary business goal, despite the Tuned Random Forest offering comparable overall predictive power."]`

---

## 10. Business Implications & Recommendations
The insights derived from this project can empower the telecommunications company to:
1.  **Implement a Proactive Churn Management System:** Deploy the chosen predictive model to regularly score customers and identify those at high risk of churning.
2.  **Develop Targeted Retention Campaigns:**
    * For customers with **short tenures** or on **month-to-month contracts**: Offer incentives for longer-term contracts, loyalty discounts, or personalized service upgrades.
    * For customers using **Electronic check**: Actively encourage and incentivize a switch to automatic payment methods (bank transfer/credit card) which are associated with lower churn.
    * For customers with **high MonthlyCharges** but low engagement (e.g., not using support services): Offer plan reviews or value consultations.
3.  **Enhance Service Offerings:**
    * Promote and highlight the benefits of services like **Online Security** and **Tech Support**, as their uptake appears to correlate with higher retention.
    * Investigate service satisfaction levels for **Fiber optic** customers to understand if specific issues contribute to their churn pattern.
4.  **Optimize Customer Onboarding and Early Lifecycle Management:** Given the importance of tenure, ensure a smooth onboarding process and engage with new customers proactively to build loyalty from the start.

---

## 11. Technologies Used
* **Programming Language:** Python 3
* **Core Libraries:**
    * Pandas (Data manipulation and analysis)
    * NumPy (Numerical operations)
    * Matplotlib (Static and interactive visualizations)
    * Seaborn (Statistical data visualization)
    * Scikit-learn (Machine learning tools: classification, model selection, preprocessing, metrics)
* **Environment:** Google Colaboratory

---

## 12. How to Run
1.  Ensure you have Python and the necessary libraries installed (or use Google Colab).
2.  Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the same directory as the notebook, or update the file path in the notebook.
3.  Open the Jupyter Notebook (`tele custmer churn.ipynb`).
4.  Run the cells sequentially from top to bottom to reproduce the analysis and results.

---

## 13. Future Work
Potential areas for further exploration and improvement include:
* Experimenting with more advanced ensemble models (e.g., XGBoost, LightGBM, CatBoost).
* Implementing techniques to explicitly handle class imbalance (e.g., SMOTE, ADASYN, or using class weights more directly in models).
* Conducting a more granular feature engineering process, potentially creating interaction terms or more complex features.
* Developing a simple interactive web application (e.g., using Streamlit or Flask) to deploy the chosen model and allow for on-demand churn predictions.
* Performing a cohort analysis or survival analysis for a deeper understanding of churn over time.
