import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib  # For saving/loading the model

st.title('💹 Credit Risk Analysis App')

st.info('This app predicts the likelihood of loan default given certain parameters!')

# Load and process data
df = pd.read_csv('https://raw.githubusercontent.com/rolanda4/stream/refs/heads/main/cleaned_dataset.csv')
X_raw = df.drop('Default', axis=1)
y_raw = df['Default'].apply(lambda x: 1 if x == 'Y' else 0)  # Encode target

# Train model once, save, and load it for predictions
if 'model.pkl' not in st.session_state:
    X_encoded = pd.get_dummies(X_raw, columns=['Home', 'Intent'])
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_raw, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    joblib.dump((model, scaler, X_encoded.columns), 'model.pkl')  # Save model, scaler, and feature columns
    st.session_state['model'] = 'model.pkl'  # Store in session state

# Load the trained model and scaler
model, scaler, feature_cols = joblib.load(st.session_state['model'])

with st.sidebar:
    st.header('Input features')
    Age = st.slider('Age (yrs)', 20, 27, 78)
    Income = st.slider('Income (Ghc)', 4000, 2039784, 65878)
    Home = st.selectbox('Home', ('OWN', 'MORTGAGE', 'RENT', 'OTHER'))
    Emp_length = st.slider('Emp_length (yrs)', 0, 41, 5)
    Intent = st.selectbox('Intent', ('EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'))
    Amount = st.slider('Amount (mm)', 500.00, 35000.00, 9588.19)
    Rate = st.slider('Rate', 5.42, 23.22, 11.00)
    Cred_length = st.slider('Cred_length (months)', 2.00, 30.00, 5.80)

if st.button("Predict Likelihood of Default"):
    # Convert user input into DataFrame
    input_data = pd.DataFrame({
        'Age': [Age],
        'Income': [Income],
        'Home': [Home],
        'Emp_length': [Emp_length],
        'Intent': [Intent],
        'Amount': [Amount],
        'Rate': [Rate],
        'Cred_length': [Cred_length]
    })

    # Encode and scale user input data
    input_parameters = pd.get_dummies(input_data, columns=['Home', 'Intent'])
    input_parameters = input_parameters.reindex(columns=feature_cols, fill_value=0)  # Align columns with training data
    input_scaled = scaler.transform(input_parameters)

    # Make predictions
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0][1]  # Probability of default

    # Display results
    st.write("Prediction:", "No Default" if prediction == 0 else "Default")
    st.write("Probability of Default:", prediction_proba)
