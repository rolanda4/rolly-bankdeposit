import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib

st.title('💶 Bank Deposit Subscription Predictor')
st.info('This app predicts the likelihood that a person will subscribe to a bank deposit given certain parameters!')

# Sidebar Inputs
with st.sidebar:
    st.header('Input features')
    age = st.slider('Age (yrs)', 17, 98, 25)
    job = st.selectbox('Job', ('admin' , 'unknown' , 'unemployed' , 'management' , 'housemaid' , 'entrepreneur' , 'student' ,
                               'blue-collar' , 'self-employed' , 'retired' , 'technician' , 'services'))
    marital = st.selectbox('Marital', ('married' , 'divorced' , 'single'))
    education = st.selectbox('Education', ('unknown' , 'secondary' , 'primary' , 'tertiary'))
    housing = st.selectbox('Housing', ('yes' , 'no'))
    loan = st.selectbox('Loan', ('yes' , 'no'))
    month = st.selectbox('Month', ('jan' , 'feb', 'mar', 'apr' , 'may', 'jun', 'jul' , 'aug' , 'sep' , 'oct' , 'nov', 'dec'))
    day_of_week = st.selectbox('Day of Week', ('mon' , 'tue' , 'wed' , 'thu' , 'fri'))
    duration = st.slider('Call Duration (secs)', 0, 4918, 1000)
    campaign = st.slider('Campaign Contacts so Far', 1, 43, 20)
    pdays = st.slider('Days Since Last Contact', -1, 999, 200)
    poutcome = st.selectbox('Previous Outcome', ('unknown' , 'failure' , 'success'))
    emp_var_rate = st.slider('Employment Variation Rate', -3.4, 1.4, 0.0)
    cons_price_idx = st.slider('Consumer Price Index', 80.0, 100.0, 93.2)
    cons_conf_idx = st.slider('Consumer Confidence Index', -100.0, 100.0, -40.0)
    euribor3m = st.slider('Euribor 3m', 0.0, 6.0, 4.0)
    nr_employed = st.slider('NR.Employed', 4000, 6000, 5191)

#input entries into a dataframe
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

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/rolanda4/rolly-bankdeposit/refs/heads/main/cleaned_add_full.csv')

# Drop unused features
df = df.drop(columns=['default', 'contact', 'previous'])

# Model training and saving
if 'model_xgb.pkl' not in st.session_state:
        # making sure time order is covered and assuming time is implied in row order, to avoid data leakage
    n_rows = len(df)
    split_index = int(n_rows * 0.8)  # Use 80% for training, 20% for testing
    
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    
    X_train_raw = train_df.drop(columns=['y'])
    y_train = train_df['y'].apply(lambda x: 1 if x == 'yes' else 0)  # Binary encoding
    
    X_test_raw = test_df.drop(columns=['y'])
    y_test = test_df['y'].apply(lambda x: 1 if x == 'yes' else 0)  # Binary encoding

        # One-hot encode 
    X_train_encoded = pd.get_dummies(X_train_raw)
    X_test_encoded = pd.get_dummies(X_test_raw)

        # Align columns to ensure test matches train
    X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

        # Scaling numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)

        # Handling class imbalance
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

         # Train the model
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

    # Store in session
    st.session_state['model'] = model
    st.session_state['scaler'] = scaler
    st.session_state['feature_cols'] = X_train_encoded.columns

# Load from session
model = st.session_state['model']
scaler = st.session_state['scaler']
feature_cols = st.session_state['feature_cols']






