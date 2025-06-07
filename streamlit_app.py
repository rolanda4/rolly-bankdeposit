import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib

st.title('💶 Bank Deposit Subscription Predictor')
st.info('This app predicts the likelihood that a person will subscribe to a bank deposit given certain parameters!')

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/rolanda4/rolly-bankdeposit/refs/heads/main/cleaned_add_full.csv')

# Drop unused features
df = df.drop(columns=['default', 'contact', 'previous'])
X_raw = df.drop(columns=['y'])
y_raw = df['y'].apply(lambda x: 1 if x == 'yes' else 0)  # Binary encoding

# Model training and saving
if 'model_xgb.pkl' not in st.session_state:
    X_encoded = pd.get_dummies(X_raw)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_raw, test_size=0.2, shuffle=False)

    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Compute class imbalance
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    # Train model
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # Save model and components
    joblib.dump((model, scaler, X_encoded.columns), 'model_xgb.pkl')
    st.session_state['model_xgb.pkl'] = 'model_xgb.pkl'

# Load trained model
model, scaler, feature_cols = joblib.load(st.session_state['model_xgb.pkl'])

# Sidebar Inputs
with st.sidebar:
    st.header('Input features')
    age = st.slider('Age (yrs)', 17, 98, 25)
    job = st.selectbox('Job', sorted(df['job'].unique()))
    marital = st.selectbox('Marital', sorted(df['marital'].unique()))
    education = st.selectbox('Education', sorted(df['education'].unique()))
    housing = st.selectbox('Housing', sorted(df['housing'].unique()))
    loan = st.selectbox('Loan', sorted(df['loan'].unique()))
    month = st.selectbox('Month', sorted(df['month'].unique()))
    day_of_week = st.selectbox('Day of Week', sorted(df['day_of_week'].unique()))
    duration = st.slider('Call Duration (secs)', 0, 4918, 1000)
    campaign = st.slider('Campaign Contacts so Far', 1, 43, 20)
    pdays = st.slider('Days Since Last Contact', -1, 999, 200)
    poutcome = st.selectbox('Previous Outcome', sorted(df['poutcome'].unique()))
    emp_var_rate = st.slider('Employment Variation Rate', -3.4, 1.4, 0.0)
    cons_price_idx = st.slider('Consumer Price Index', 80.0, 100.0, 93.2)
    cons_conf_idx = st.slider('Consumer Confidence Index', -100.0, 100.0, -40.0)
    euribor3m = st.slider('Euribor 3m', 0.0, 6.0, 4.0)
    nr_employed = st.slider('NR.Employed', 4000, 6000, 5191)

# Prediction
if st.button("Predict Likelihood of Subscription"):
    input_df = pd.DataFrame([{
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'housing': housing,
        'loan': loan,
        'month': month,
        'day_of_week': day_of_week,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'poutcome': poutcome,
        'emp.var.rate': emp_var_rate,
        'cons.price.idx': cons_price_idx,
        'cons.conf.idx': cons_conf_idx,
        'euribor3m': euribor3m,
        'nr.employed': nr_employed
    }])

    # Encode and align columns
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=feature_cols, fill_value=0)

    # Scale numeric features
    input_scaled = scaler.transform(input_encoded)

    # Predict
    prediction = model.predict_proba(input_scaled)[0][1]
    result = "✅ Likely to Subscribe" if prediction >= 0.3 else "❌ Not Likely to Subscribe"

    st.subheader("Prediction Result")
    st.write(f"**Probability of Subscription:** {prediction:.2%}")
    st.write(result)
