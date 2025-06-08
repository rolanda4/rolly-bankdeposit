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
    # housing = st.selectbox('Housing', sorted(df['housing'].unique()))
    # loan = st.selectbox('Loan', sorted(df['loan'].unique()))
    # month = st.selectbox('Month', sorted(df['month'].unique()))
    # day_of_week = st.selectbox('Day of Week', sorted(df['day_of_week'].unique()))
    # duration = st.slider('Call Duration (secs)', 0, 4918, 1000)
    # campaign = st.slider('Campaign Contacts so Far', 1, 43, 20)
    # pdays = st.slider('Days Since Last Contact', -1, 999, 200)
    # poutcome = st.selectbox('Previous Outcome', sorted(df['poutcome'].unique()))
    # emp_var_rate = st.slider('Employment Variation Rate', -3.4, 1.4, 0.0)
    # cons_price_idx = st.slider('Consumer Price Index', 80.0, 100.0, 93.2)
    # cons_conf_idx = st.slider('Consumer Confidence Index', -100.0, 100.0, -40.0)
    # euribor3m = st.slider('Euribor 3m', 0.0, 6.0, 4.0)
    # nr_employed = st.slider('NR.Employed', 4000, 6000, 5191)

