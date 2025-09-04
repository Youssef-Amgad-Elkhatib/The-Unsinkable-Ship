# 🚢 The Unsinkable Ship 


> A machine learning project built on the **Titanic dataset**, where I use ensemble methods to predict passenger survival.  
> Inspired by the tragedy of the Titanic, this project explores feature engineering, model tuning, and ensembling to create an **unsinkable predictor**.  

---
# 🎥 Video
 
[![The Unsinkable Ship](https://ytcards.demolab.com/?id=M-NlX0MDmQQ&title=The+Unsinkable+Ship&lang=en&timestamp=1756995583&background_color=%230d1117&title_color=%23ffffff&stats_color=%23dedede&max_title_lines=2&width=300&border_radius=5&duration=157 "The Unsinkable Ship")]([https://www.youtube.com/watch?v=wJ710_eJ5uw](https://www.youtube.com/watch?v=M-NlX0MDmQQ))

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
The best-performing ensemble combines:  
- 🎯 **GradientBoostingClassifier**  
- ⚡ **XGBClassifier (GPU accelerated)**  

Ensemble type:  
- 🤝 **Soft Voting Ensemble** → Weighted (GB : XGB = 2:1)  
 
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
