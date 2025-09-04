import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import streamlit as st

# -----------------------------
# Warnings
# -----------------------------
warnings.filterwarnings("ignore")


# -----------------------------
# Data Preprocessing Functions
# -----------------------------
def load_data():
    df = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    df.drop_duplicates(inplace=True)
    return df, test


def extract_title(df):
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    title_map = {
        "Mlle": "Miss", "Ms": "Miss", "Miss": "Miss",
        "Mme": "Mrs", "Mrs": "Mrs", "Countess": "Mrs", "Lady": "Mrs",
        "Mr": "Mr", "Dr": "Mr", "Rev": "Mr", "Major": "Mr",
        "Col": "Mr", "Capt": "Mr", "Sir": "Mr", "Don": "Mr", "Jonkheer": "Mr",
        "Master": "Master"
    }
    df["Title"] = df["Title"].map(title_map).fillna("Mr")
    return df


def engineer_features(df):
    # Family size & alone
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Age categories
    title_ages = df.groupby('Title')['Age'].median()
    df['Age'] = df.apply(lambda row: title_ages[row['Title']] if pd.isnull(row['Age']) else row['Age'], axis=1)

    df['Child'] = df['Age'].between(0, 12.99).astype(int)
    df['Teen'] = df['Age'].between(13, 19.99).astype(int)
    df['Young_Adult'] = df['Age'].between(20, 35.99).astype(int)
    df['Adult'] = df['Age'].between(36, 60.99).astype(int)
    df['Senior'] = (df['Age'] >= 61).astype(int)

    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    # Now drop columns
    df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin", "Title", "Age", "SibSp", "Parch"], axis=1, inplace=True,
            errors='ignore')
    return df, title_ages


def prepare_test_data(test, title_ages):
    test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    test = extract_title(test)

    test["FamilySize"] = test["SibSp"] + test["Parch"] + 1
    test['IsAlone'] = (test['FamilySize'] == 1).astype(int)

    test['Age'] = test.apply(lambda row: title_ages[row['Title']] if pd.isnull(row['Age']) else row['Age'], axis=1)

    test['Child'] = test['Age'].between(0, 12.99).astype(int)
    test['Teen'] = test['Age'].between(13, 19.99).astype(int)
    test['Young_Adult'] = test['Age'].between(20, 35.99).astype(int)
    test['Adult'] = test['Age'].between(36, 60.99).astype(int)
    test['Senior'] = (test['Age'] >= 61).astype(int)

    test["Sex"] = test["Sex"].map({"male": 0, "female": 1})
    test["Fare"] = test["Fare"].fillna(test["Fare"].median())

    test.drop(columns=["Name", "Ticket", "Cabin", "Title", "Age", "SibSp", "Parch"], axis=1, inplace=True,
              errors='ignore')
    return test


# -----------------------------
# Model Training Function
# -----------------------------
@st.cache_resource
def train_model(x, y):
    gb = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.01, max_depth=3,
        min_samples_split=2, min_samples_leaf=5,
        subsample=1.0, max_features='sqrt', random_state=214
    )

    xgb = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.01,
        subsample=0.8, colsample_bytree=0.6, gamma=1, min_child_weight=1,
        scale_pos_weight=1, random_state=244, device="cuda",
        use_label_encoder=False, eval_metric="logloss"
    )

    ensemble = VotingClassifier(
        estimators=[("gb", gb), ("xgb", xgb)],
        voting="soft", weights=[2, 1], n_jobs=-1
    )

    ensemble.fit(x, y)
    return ensemble


# -----------------------------
# Streamlit App
# -----------------------------
def main():
    st.set_page_config(
        page_title="The Unsinkable Ship",
        page_icon="🚢",
        layout="centered"
    )

    # Load and preprocess data
    df, test = load_data()
    df = extract_title(df)
    df, title_ages = engineer_features(df)
    test = prepare_test_data(test, title_ages)

    features = ['Pclass', 'Sex', 'Fare', 'FamilySize', 'IsAlone', 'Child', 'Teen', 'Young_Adult', 'Adult', 'Senior']
    x = df[features]
    y = df["Survived"]

    # Train ensemble model
    ensemble = train_model(x, y)

    # -----------------------------
    # Streamlit UI
    # -----------------------------
    set_styles()

    st.title("🚢 The Unsinkable Ship")
    st.subheader("\u26A1 Predict Titanic survival with lightning speed!")

    user_input = get_user_input()
    if st.button("Predict Survival"):
        predict_survival(user_input, ensemble)


# -----------------------------
# Functions for Streamlit UI
# -----------------------------
def set_styles():
    st.markdown("""
    <style>
    body { background-color: #0d1117; color: #fff; }
    h1, h2, h3 { text-align: center; color:#fff; text-shadow:0 0 3px #b68d2c,0 0 7px #e0c168,0 0 12px #f9e79f; }
    label, .stMarkdown, .stRadio, .stSelectbox, .stTextInput, .stNumberInput, .stDataFrame{
        color:#f9e79f !important; text-shadow:0 0 3px #b68d2c,0 0 7px #e0c168,0 0 12px #f9e79f; margin-top:30px;}
    .stSelectbox{margin-top:70px;}
    .stButton {display:flex; justify-content:center; margin-top:30px;}
    .stButton>button{background: linear-gradient(90deg,#b68d2c,#e0c168,#f9e79f,#e0c168,#b68d2c) !important;
    color:black;border-radius:12px;border:2px solid #e0c168;font-weight:bold;padding:10px 25px;font-size:18px;transition:all 0.3s ease;
    text-shadow:0 0 3px #b68d2c,0 0 7px #e0c168,0 0 12px #f9e79f;}
    .stButton>button:hover{background:linear-gradient(90deg,#f9e79f,#e0c168,#b68d2c) !important;transform:scale(1.05);}
    .stNumberInput button{color:#e0c168 !important; transition: all 0.2s ease-in-out;}
    .stNumberInput button:hover{background:#e0c168 !important;color:#0d1117 !important;}
    input[type=number]::-webkit-inner-spin-button,input[type=number]::-webkit-outer-spin-button{-webkit-appearance:none;margin:0;}
    </style>
    """, unsafe_allow_html=True)


def get_user_input():
    pclass = st.selectbox("Ticket Class (Pclass)", [1, 2, 3])
    sex = st.radio("Gender", ["Male", "Female"])
    fare = st.number_input("Fare (GBP)", min_value=0.0, value=32.0, step=0.5)
    sibsp = st.number_input("No. of Siblings / Spouses aboard", min_value=0, value=0, step=1)
    parch = st.number_input("No. of Parents / Children aboard", min_value=0, value=0, step=1)
    age = st.number_input("Age", min_value=0, max_value=100, value=30, step=1)

    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0
    sex_encoded = 1 if sex == "Male" else 0
    child = 1 if age <= 12.99 else 0
    teen = 1 if 13 <= age <= 19.99 else 0
    young_adult = 1 if 20 <= age <= 35.99 else 0
    adult = 1 if 36 <= age <= 60.99 else 0
    senior = 1 if age >= 61 else 0

    return np.array([[pclass, sex_encoded, fare, family_size, is_alone,
                      child, teen, young_adult, adult, senior]])


def predict_survival(features, model):
    survival = model.predict(features)[0]
    survival_proba = model.predict_proba(features)[0][1]

    st.subheader("Result")
    if survival == 1:
        st.success(f"\U0001F30A Survived! Probability: {survival_proba:.2%}")
    else:
        st.error(f"\u2693 Did not survive. Probability of survival: {survival_proba:.2%}")


# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    main()
