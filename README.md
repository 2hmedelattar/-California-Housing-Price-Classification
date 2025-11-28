# 🏡 California Housing Price Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Optuna](https://img.shields.io/badge/Optimization-Optuna-blueviolet)
![Streamlit](https://img.shields.io/badge/Deployment-Streamlit-red)

## 📌 Project Overview
This project aims to classify housing prices in California into three categories (**Low**, **Medium**, **High**) based on various features like median income, house age, and location.

The solution leverages **Ensemble Learning (Bagging)** and **Decision Trees**, with hyperparameters rigorously tuned using **Bayesian Optimization (Optuna)** to maximize accuracy.

## 🛠️ Key Features
- **Advanced Preprocessing:** - Quantile Binning (`pd.qcut`) for balanced target classes.
  - Interaction features & Geospatial Engineering (Rotated Coordinates, Distance to LA/SF).
- **Model Optimization:** - Used **Optuna** to tune hyperparameters for Decision Trees, Random Forests, and Bagging Classifiers.
  - Comparison between Default vs. Tuned models to quantify improvement.
- **Deployment:** - Interactive Web App built with **Streamlit** for real-time predictions.

## 📊 Results
| Model | Accuracy |
|-------|----------|
| Decision Tree | ~79% |
| **Bagging (Optimized)** | **~85%** |
| Random Forest | ~85% |

*Key Insight: The **Median Income** and **Geographic Location** proved to be the strongest predictors of housing prices.*

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/California-Housing-Price-Classification.git](https://github.com/YOUR_USERNAME/California-Housing-Price-Classification.git)
