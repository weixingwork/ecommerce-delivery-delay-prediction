# 📦 Delivering Trust: Predicting E-Commerce Delivery Delays and Smart Insurance Pricing

## Overview
This project predicts e-commerce delivery delays and designs a dynamic shipping insurance pricing strategy using machine learning.  
We analyze real-world marketplace data, develop predictive models, and propose a smart, risk-based insurance pricing mechanism to improve customer experience and increase platform profitability.

## Project Structure
- **Data Cleaning:** Preprocessing and merging datasets for modeling (`1_cleaning_and_save.ipynb`)
- **Exploratory Data Analysis (EDA):** Delay rate analysis, seasonality, and geographic patterns (`4_EDA.ipynb`)
- **Modeling:**
  - **Binary Classification:** Predict on-time vs. delayed deliveries (`2_month_feature_model.ipynb`)
  - **Multi-class Classification:** Predict severity of delay (no delay, 1–3 days, >3 days) (`2_month_feature_model_multiclass.ipynb`)
  - **Regression:** Predict the number of delay days (`2_month_feature_model_regression.ipynb`)
  - **Monthly Models:** Separate models trained per month to capture seasonality (`3_monthly_models.ipynb`)
- **Insurance Pricing Design:** Risk-based shipping insurance pricing based on predicted delay probabilities (`5_insurance_pricing.ipynb`, `price.ipynb`)

## Machine Learning Techniques
- Logistic Regression (Classification)
- Random Forest Classifier (Classification)
- Linear Regression (Regression)
- Random Forest Regressor (Regression)
- Feature Engineering: One-Hot Encoding, Missing Value Handling
- Month-wise Model Training

## Dataset
- Real-world commercial e-commerce data (anonymized)
- Includes order, shipping, payment, product, customer review, and geolocation information
- Time range: 2016–2018
- Size: 100k+ orders after cleaning

## Key Results
- **Classification Recall:** 68% (predicting delayed orders)
- **Regression R²:** 0.73 (predicting delay days)
- **Insurance Pricing Strategy:** Designed based on predicted delay risk, achieving approximately 20% profit margin.

## How to Run
1. Start from `1_cleaning_and_save.ipynb` to prepare the dataset.
2. Perform EDA with `4_EDA.ipynb`.
3. Train models using:
   - `2_month_feature_model.ipynb` (Binary classification)
   - `2_month_feature_model_multiclass.ipynb` (Multi-class classification)
   - `2_month_feature_model_regression.ipynb` (Regression)
   - `3_monthly_models.ipynb` (Month-specific models)
4. Generate insurance pricing strategies using `5_insurance_pricing.ipynb` and `price.ipynb`.

## Dependencies
```bash
pip install pandas scikit-learn matplotlib seaborn
```

## Authors
- Wei Xing
- Rui Guo
