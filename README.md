# Real Estate Anomaly Detector

This repository hosts our CS4824 Capstone Project, which aims to develop an end-to-end machine learning model that detects anomalous pricing in the housing market. In this context, housing market anomalies refer to listings that deviate from expected market behavior, typically driven by supply, demand, and broader economic fundamentals. Anomalies challenge the idea that markets are perfectly efficient; they may arise intentionally (as scams and unassured pricing) or unintentionally through human error or flawed valuation models. Establishing a reliable way to identify these irregularities can help minimize losses for both sellers and buyers.

Authors: Rohan Magesh (mrohan@vt.edu), Shravan Athikinasetti (sathikinasetti@vt.edu), Gayatri Kamtala (gayatrikam@vt.edu) 

## Repository Structure
- [`/exploratory_data_analysis`](https://github.com/gayatrikam/estate-anomaly-detector/tree/main/exploratory_data_analysis): Contains all data cleaning, preprocessing, and exploratory data analysis scripts.
- [`/model`](https://github.com/gayatrikam/estate-anomaly-detector/tree/main/model): Contains model training logic, evaluation metrics, and final model files.

## Literature Review
- [The Effect of Outlier Detection Methods in Real Estate Valuation with Machine Learning](https://dergipark.org.tr/en/download/article-file/3033205)
    - Purpose - Quantify how outlier detection during preprocessing affects real-estate price prediction accuracy. 
    - Methodology - Compared IQR, Modified Z-Score, and Isolation Forest to KNN, Lasso, and Random Forest. 
    - Findings - Proper outlier handling improved prediction accuracy by about 7% across models, and in Random Forest, by over 21%.
- [Enhancing Zillow Zestimates: Leveraging Machine Learning for Precise Property Valuation Predictions](https://ieeexplore.ieee.org/abstract/document/10652318)
    - Purpose - Identify the best ML model for housing price prediction to improve Zillow's Zestimate.
    - Methodology - Applied a diverse set of regression algorithms (Lasso, Ridge, Elastic Net, Linear Regression) and decision tree methods (Gradient Boosting, Random Forest) on the same Kaggle Zillow dataset used in our project.
    - Findings - Gradient boosting was shown to be the best model, with an R2 score of 0.88, MAE of 20000, MSE of 1000000000, and RMSE of 31622.78.
- [Exploring the Impact of Zestimate on Real Estate Market Dynamics: A Case Study of Buyer, Seller, and Renter Perspectives](https://digibug.ugr.es/handle/10481/103600)
    - Purpose - Investigate how Zillow’s Zestimate influences buyer, seller, renter, and broader market behavior.
    - Methodology - Tracked the evolution of the Zestimate since 2006 and analyzed behavioral shifts through a stakeholder framework.
    - Findings - The Zestimate significantly affects decision-making in the housing industry; the effects of inaccuracies in its predictions are non-negligible.
- [Advanced Machine Learning Techniques for Predictive Modeling of Property Prices](https://pure.solent.ac.uk/ws/portalfiles/portal/66849608/information-15-00295_1_.pdf)
    - Purpose - Improve model prediction accuracy for housing price datasets through experiments in data selection and outlier removal.
    - Methodology - Compared Ridge Regression, Random Forest, and XGBoost, alongside robust outlier-handling techniques.
    - Findings - Ensemble models perform better after outlier handling.
- [Utilizing Model Residuals to Identify Rental Properties of Interest: The Price Anomaly Score (PAS) and Its Application to Real-time Data in Manhattan](https://arxiv.org/abs/2311.17287)
    - Purpose - Detect overpriced Manhattan rental properties in real time.
    - Methodology - Introduced a Price Anomaly Score (PAS) based on model residuals and statistical significance.
    - Findings - Developed a metric capable of capturing boundaries between irregularly predicted prices.

## Dataset

We chose the dataset published by Zillow for their 2017 model-building competition on Kaggle: https://www.kaggle.com/competitions/zillow-prize-1/data

This dataset is a subset of the now-retired ZTRAX database focused on Southern California. It includes nearly one million property records across Los Angeles, Orange, and Ventura counties from 2016, with 58 structural and locational features.

Before we can build our anomaly detection mechanism, we must first develop a price prediction model. The dataset was specifically released to be compatible with housing price prediction tasks, containing features that capture structural, locational, and transactional attributes of properties. This makes it an ideal foundation for benchmarking our anomaly detector against Zillow’s Zestimate. Our goal is to train a predictive model, scale it according to real estate inflation, and cross-check its estimates against current Zestimates to determine whether historical data can still yield accurate predictions in today’s market.

Our current workflow:
- Build a price prediction model to mimic the Zestimate, test on 2016 data
- Add the anomaly detection feature to flag over and under-priced properties within the 2016 dataset
- Scale the model to 2025 data using real-estate inflation rates to analyze modern listings

## Library Documentation
- [Pandas Library](https://pandas.pydata.org/docs/)
- [NumPy Library](https://numpy.org/doc/2.3/)
- [Matplotlib Library](https://matplotlib.org/stable/index.html)
- [scikit-learn Library](https://scikit-learn.org/stable/)

## Current Status and Future Goals

We have completed data preprocessing, exploratory data analysis, and initial model selection. Currently, we are developing the first iteration of our price prediction model.

Our upcoming milestones include:
1. Improve model accuracy by refining feature engineering and hyperparameter tuning.
2. Implement the anomaly detection module, flagging listings that deviate beyond a defined Z-score or confidence interval threshold.
3. Scale predictions to current (2025) housing data, adjusting for inflation and market shifts in Southern California.
4. Develop a simple deployment tool, such as a lightweight web app using Flask, that allows users to input property details and determine whether a listing is anomalous.

## How to Run

More information to come once final model and deployment details are configured.