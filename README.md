# 🏡 California Housing Price Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange?logo=scikit-learn&logoColor=white)
![Optuna](https://img.shields.io/badge/Optimization-Optuna-blueviolet?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

This project builds a robust Machine Learning pipeline to classify housing prices in California into **Low**, **Medium**, and **High** categories. 

It moves beyond basic modeling by implementing **Advanced Feature Engineering** (Geospatial Analysis) and rigorous **Hyperparameter Tuning** using **Bayesian Optimization (Optuna)**. The final model is deployed as an interactive web app using **Streamlit**.

---

## 📂 Dataset

**Source:** [California Housing Dataset (Scikit-Learn)](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)  
**Total Samples:** 20,640 records

### **Columns Used**

* `MedInc` (Median Income - Strongest Predictor)
* `HouseAge`
* `AveRooms`
* `AveBedrms`
* `Population`
* `AveOccup`
* `Latitude` & `Longitude`

### **Target Variable (Engineered)**

The original continuous target `MedHouseValue` was binned into 3 classes to transform the problem into a classification task:
* **0:** Low Price
* **1:** Medium Price
* **2:** High Price

---

## 🎯 Project Objective

Develop a **Machine Learning Classifier** to predict property value categories based on demographic and geographic data, optimizing for **Accuracy** using Ensemble methods.

---

## 🧠 Model Architecture & Methodology

The project follows a strict pipeline implemented in a **Jupyter Notebook**.

### **Techniques Applied**

1.  **Data Preprocessing:**
    * Quantile Binning (`pd.qcut`) for balanced classes.
    * Outlier handling (Clipping Rooms/Bedrooms).
    * Standard Scaling (`StandardScaler`).

2.  **Feature Engineering:**
    * **Geospatial Analysis:** Calculated distance to economic hubs (Los Angeles & San Francisco).
    * **Rotated Coordinates:** Applied 45° rotation to help Decision Trees capture diagonal geographic patterns.

3.  **Models & Optimization:**
    * **Decision Tree:** Tuned `max_depth`, `min_samples_leaf`.
    * **Bagging Classifier:** Tuned `n_estimators`, `max_samples`.
    * **Random Forest:** Tuned using **Optuna** (Bayesian Optimization).

---

## 🛠️ Tech Stack

* **Python**
* **Pandas & NumPy** (Data Manipulation)
* **Matplotlib & Seaborn** (Visualization)
* **Scikit-Learn** (Modeling)
* **Optuna** (Hyperparameter Tuning)
* **Streamlit** (Web Deployment)

---

## 📘 Project Structure

```text
California-Housing-Classification/
│── Trees_1.ipynb          # 📓 Main Notebook (EDA, Engineering, Optuna)
│── app.py                 # 🚀 Streamlit Application Script
│── housing_model.pkl      # 💾 Saved Optimized Model
│── scaler.pkl             # ⚖️ Saved Scaler
│── requirements.txt       # 📦 Dependencies
└── README.md              # 📄 Project Documentation
