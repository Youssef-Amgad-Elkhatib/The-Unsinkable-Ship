# 🚢 Unsinkable Ensemble: Titanic Survival Prediction  

![Titanic](https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg)  

> A machine learning project built on the **Titanic dataset**, where we use ensemble methods to predict passenger survival.  
> Inspired by the tragedy of the Titanic, this project explores feature engineering, model tuning, and ensembling to create an **unsinkable predictor**.  

---

## 📚 Libraries Used  
- 🐼 **Pandas** – data wrangling  
- 🔢 **NumPy** – numerical operations  
- 📊 **Matplotlib / Seaborn** – visualization  
- 🌲 **RandomForest** – baseline model  
- 📈 **GradientBoosting** – boosting power  
- ⚡ **XGBoost (GPU)** – optimized boosting  
- 🤝 **Voting & Stacking Ensembles** – combining strengths  

---

## 🛠 Features Engineered  
- 👤 **Title extraction** from passenger names  
- 💰 **Fare binning** (grouping fares)  
- 👪 **Family size** feature  
- 🎂 **Age categories** (child, teen, adult, senior)
  
---

## 📊 Survival by Sex (EDA Insight)  
- 🚹 **Males:** ~20% survival rate  
- 🚺 **Females:** ~75% survival rate  
- 🔑 Gender is one of the **most important predictors** of survival  

---

## 🧩 Model Architecture  
Our best-performing ensemble combines:  
- 🎯 **GradientBoostingClassifier**  
- ⚡ **XGBClassifier (GPU accelerated)**  

Ensemble type:  
- 🤝 **Soft Voting Ensemble** → Weighted (GB : XGB = 2:1)  
- Optionally tested with **Stacking** using Logistic Regression  

---

## 📈 Results  
- ✅ **Cross-Validation Accuracy:** ~0.805 ± 0.034  
- ✅ **Private Test Accuracy:** ~0.80  

---

## 💾 Saving and loading the Model  
```python
import joblib

# Save
joblib.dump(ensemble, "titanic_ensemble_model.pkl")

# Load
model = joblib.load("titanic_ensemble_model.pkl")
