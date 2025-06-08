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
