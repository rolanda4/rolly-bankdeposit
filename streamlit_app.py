import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# App title
st.title('💶Bank Deposit Subscription Predictor')
st.info('This app predicts the likelihood that a person will subscribe to a bank deposit given certain parameters!')

# Load dataset
dup_add = pd.read_csv('https://raw.githubusercontent.com/rolanda4/rolly-bankdeposit/refs/heads/main/cleaned_add_full.csv')

# Prepare features and target
features_to_drop = ['default', 'contact', 'previous']
X = dup_add.drop(columns=features_to_drop + ['y'])
y = dup_add['y']

# Identify column types
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# Split data (time-aware)
split_index = int(len(X) * 0.8)
X_train, y_train = X.iloc[:split_index], y.iloc[:split_index]

# Preprocessing and model pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
    ]
)

scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42
    ))
])

# Train the model once per session
if 'model' not in st.session_state:
    st.session_state.model = model_pipeline.fit(X_train, y_train)

model = st.session_state.model

# --- Sidebar Inputs ---
with st.sidebar:
    st.header('Input features')
    age = st.slider('Age (yrs)', 17, 98, 25)
    job = st.selectbox('Job', sorted(dup_add['job'].unique()))
    marital = st.selectbox('Marital', sorted(dup_add['marital'].unique()))
    education = st.selectbox('Education', sorted(dup_add['education'].unique()))
    housing = st.selectbox('Housing', sorted(dup_add['housing'].unique()))
    loan = st.selectbox('Loan', sorted(dup_add['loan'].unique()))
    month = st.selectbox('Month', sorted(dup_add['month'].unique()))
    day_of_week = st.selectbox('Day of Week', sorted(dup_add['day_of_week'].unique()))
    duration = st.slider('Call Duration (secs)', 0, 4918, 1000)
    campaign = st.slider('Campaign Contacts so Far', 1, 43, 20)
    pdays = st.slider('Days Since Last Contact', -1, 999, 200)
    poutcome = st.selectbox('Previous Outcome', sorted(dup_add['poutcome'].unique()))
    emp_var_rate = st.slider('Employment Variation Rate', -3.4, 1.4, 0.0)
    cons_price_idx = st.slider('Consumer Price Index', 80.0, 100.0, 93.2)
    cons_conf_idx = st.slider('Consumer Confidence Index', -100.0, 100.0, -40.0)
    euribor3m = st.slider('Euribor 3m', 0.0, 6.0, 4.0)
    nr_employed = st.slider('NR.Employed', 4000, 6000, 5191)

# --- Prediction ---
if st.button("Predict Likelihood of Subscription"):
    input_data = pd.DataFrame({
        'age': [age],
        'job': [job],
        'marital': [marital],
        'education': [education],
        'housing': [housing],
        'loan': [loan],
        'month': [month],
        'day_of_week': [day_of_week],
        'duration': [duration],
        'campaign': [campaign],
        'pdays': [pdays],
        'poutcome': [poutcome],
        'emp.var.rate': [emp_var_rate],
        'cons.price.idx': [cons_price_idx],
        'cons.conf.idx': [cons_conf_idx],
        'euribor3m': [euribor3m],
        'nr.employed': [nr_employed]
    })

    prediction = model.predict_proba(input_data)[0][1]
    result = "✅ Likely to Subscribe" if prediction >= 0.3 else "❌ Not Likely to Subscribe"

    st.subheader("Prediction Result")
    st.write(f"**Probability of Subscription:** {prediction:.2%}")
    st.write(result)
