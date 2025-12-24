
# Applied Machine Learning Project – Regression Analysis Report

## 1. Problem Definition and Evaluation Metric

The objective of this project is to predict **individual medical insurance charges** based on demographic and lifestyle attributes such as age, sex, BMI, smoking status, region, and number of dependents. Accurately predicting medical costs can help insurers estimate premiums and better manage financial risk.

This is a **supervised regression** problem where the target variable is `charges`, a continuous numerical value representing annual medical insurance costs.

### Evaluation Metric
The primary evaluation metric used in this project is **Root Mean Squared Log Error (RMSLE)**. RMSLE is well-suited for this task because:
- Medical costs are positively skewed, and RMSLE reduces the impact of large outliers.
- It penalizes underestimation more heavily than overestimation, which is important in financial forecasting.
- It allows comparison of relative errors rather than absolute differences.

Additional metrics such as Mean Absolute Error (MAE) and R-squared (R²) were also used for interpretability and comparison.


## 2. Data Preprocessing

### Dataset Overview
The dataset contains **1,338 rows and 7 features**, sourced from a publicly available medical cost dataset. The features include both numerical and categorical variables.

### Missing Value Handling
An initial inspection of the dataset revealed **no missing values** in either numerical or categorical columns. As a result, no imputation was required. However, the dataset was explicitly checked to ensure robustness and reproducibility of the preprocessing pipeline.

### Categorical Encoding
Categorical features (`sex`, `smoker`, and `region`) were converted into numerical representations using encoding techniques to make them compatible with machine learning models.

### Feature Scaling
Numerical features were left unscaled because tree-based models (such as Random Forests) are not sensitive to feature scaling.


## 3. Modeling Approach and Hyperparameter Tuning

### Baseline Model
A **RandomForestRegressor** was selected as the baseline model due to its ability to:
- Capture non-linear relationships
- Handle mixed data types
- Reduce overfitting through ensemble learning

The baseline model was trained using default hyperparameters and evaluated on a validation split of the dataset.

### Hyperparameter Tuning
To improve performance, **RandomizedSearchCV** was used to tune key hyperparameters, including:
- Number of estimators
- Maximum tree depth
- Minimum samples per split
- Minimum samples per leaf

To reduce computational cost during tuning, the number of iterations and cross-validation folds was limited.

### Final Model
The best-performing model from the randomized search was retrained on the full training dataset using the optimal hyperparameters identified during tuning.


## 4. Model Performance Comparison

 Model : RMSLE , MAE , R² 
 Baseline Random Forest : Improved , Moderate , Strong 
Tuned Random Forest : **Lower** , **Lower** , **Higher** 

The hyperparameter-tuned model demonstrated improved performance across all evaluation metrics, particularly RMSLE, indicating better generalization and predictive accuracy.


## 5. Feature Importance Analysis

The final model’s feature importance scores were analyzed to understand the drivers of medical insurance costs.

### Top 3 Most Important Features
1. **Smoker Status** – Smoking was the strongest predictor of higher medical costs.
2. **BMI** – Higher BMI values were associated with increased healthcare expenses.
3. **Age** – Medical costs generally increased with age due to higher health risks.

These results align with real-world expectations and provide confidence in the model’s interpretability.


## 6. Deployment Summary

The final, best-performing model was deployed as a **RESTful API using FastAPI** and hosted on the **Render platform**. The API exposes a `/predict` endpoint that accepts raw input features and returns predicted medical insurance charges after applying the same preprocessing steps used during training.

**Deployment Link:**  
> *(Insert your live Render URL here)*


## 7. Conclusion

This project demonstrates a complete end-to-end applied machine learning workflow, including problem formulation, data preprocessing, model training, evaluation, hyperparameter tuning, and deployment. The tuned Random Forest model achieved strong predictive performance and was successfully operationalized as a web service, making it suitable for real-world use cases.
