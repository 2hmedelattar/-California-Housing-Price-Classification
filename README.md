# 🏡 California Housing Price Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange?logo=scikit-learn&logoColor=white)
![Optuna](https://img.shields.io/badge/Optimization-Optuna-blueviolet?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A comprehensive Machine Learning project to classify California housing prices into **Low**, **Medium**, and **High** categories.

This project goes beyond basic modeling by implementing **Advanced Feature Engineering** (Geospatial Analysis), **Ensemble Learning** (Bagging & Random Forest), and rigorous **Hyperparameter Tuning** using **Bayesian Optimization (Optuna)**.

---

## 📂 Dataset Overview

**Source:** [California Housing Dataset (via Scikit-Learn)](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)  
**Records:** 20,640 samples  
**Problem Type:** Multi-class Classification (Target binned into 3 classes).

### Key Features
| Feature | Description |
| :--- | :--- |
| `MedInc` | Median Income in block group (**Strongest Predictor**) |
| `HouseAge` | Median House Age in block group |
| `AveRooms` | Average number of rooms per household |
| `Latitude/Longitude` | Geographic coordinates |
| **Derived Features** | Distance to Coast, Rotated Coordinates, etc. |

---

## 🎯 Project Objective

To build a high-performance classification model that predicts property value categories. The workflow emphasizes:
1.  **Interpretation:** Understanding *why* a house is expensive.
2.  **Optimization:** Using `Optuna` to find the global optimum hyperparameters.
3.  **Deployment:** Serving the model via an interactive web app.

---

## 🧠 Methodology & Feature Engineering

The project logic is encapsulated in `Trees_1.ipynb`.

### 1. Advanced Feature Engineering
We didn't just use raw data. We created features to capture the **geographical context**:

* **Rotated Coordinates:** Since California lies diagonally, we rotated coordinates by 45° to help Decision Trees split data more effectively.
* **Proximity to Hubs:** Calculated Euclidean distance to major economic centers (LA & SF).

```python
# Snippet: Calculating Distance to Major Cities
def calculate_distance(lat, lon, city_lat, city_lon):
    return np.sqrt((lat - city_lat) ** 2 + (lon - city_lon) ** 2)

# Feature: Distance to Los Angeles
df['Dist_to_LA'] = calculate_distance(df['Latitude'], df['Longitude'], 34.05, -118.24)
