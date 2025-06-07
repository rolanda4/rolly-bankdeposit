import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib  

st.title('💹 Bank Deposit Subscription Predictor')

st.info('This app predicts the likelihood that a person will subscribe to a bank deposit given certain parameters!')

# Load and process data
dup_add = pd.read_csv('https://raw.githubusercontent.com/rolanda4/stream/refs/heads/main/cleaned_add_full.csv')

features_to_drop2 = ['default', 'contact', 'previous']
X2_train = train_df.drop(columns=features_to_drop2 + ['y'])
y2_train = train_df["y"]

#identify categorical and numeric variables
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(exclude=['object']).columns.tolist()

#creating pre-processing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
           ("num", StandardScaler(), numeric_cols),
           ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_cols)
        ],
        remainder="passthrough"  # keep numeric columns as-is
    )

#model pipeline
scale_pos_weight = len(y[y == 0]) / len(y[y == 1])
    
xgb_pipeline = Pipeline(steps=[
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

# making sure time order is covered and assuming time is implied in row order, to avoid data leakage
n_rows = len(dup_add)
split_index = int(n_rows * 0.8)  # Use 80% for training, 20% for testing

X_train, y_train = X.iloc[:split_index], y.iloc[:split_index]

# Fit the model
if 'model' not in st.session_state: #save trained model so it doesn't have to constantly retrain
    st.session_state.model = model_pipeline.fit(X_train, y_train)

model = st.session_state.model

with st.sidebar:
    st.header('Input features')
    age = st.slider('Age (yrs)', 17, 98, 25)
    job = st.selectbox('Job', dup_add['job'].unique())
    marital = st.selectbox('Marital', dup_add['marital'].unique())
    education = st.selectbox('Education', dup_add['education'].unique())
    housing = st.selectbox('Housing', dup_add['housing'].unique())
    loan = st.selectbox('Loan', dup_add['loan'].unique())
    month = st.selectbox('Month', dup_add['month'].unique())
    day_of_week = st.selectbox('Day of Week', dup_add['day_of_week'].unique())
    duration = st.slider('Call Duration (secs)', 0, 4918, 1000)
    campaign = st.slider('Campaign Contacts so Far', 1, 43, 20)
    pdays = st.slider('Days Since Last Contact', -1, 999, 200)
    poutcome = st.selectbox('Previous Outcome', dup_add['poutcome'].unique())
    emp_var_rate = st.slider('Employment Variation Rate', -3.4, 1.4, 5)
    cons_price_idx = st.slider('Consumer Price Index', 80, 100, 85)
    cons_conf_idx = st.slider('Consumer Confidence Index', -100, 100, 0)
    euribor3m = st.slider('Euribor 3m', -1, 1, 0)
    nr_employed = st.slider('NR.Employed', 4000, 6000, 5000)


if st.button("Predict Likelihood of Subscription"):
    # Convert user input into DataFrame
    input_data = pd.DataFrame({
        'age' : [age]
        'job' : [job]
        'marital' : [marital]
        'education' : [education]
        'housing' : [housing]
        'loan' : [month]
        'month' : [month]
        'day_of_week' : [day_of_week]
        'duration' : [duration]
        'campaign' : [campaign]
        'pdays' : [pdays]
        'poutcome' : [poutcome]
        'emp.var.rate' : ['emp_var_rate']
        'cons.price.idx' : ['cons_price_idx']
        'cons.conf.idx' : ['cons_conf_idx']
        'euribor3m' : [euribor3m]
        'nr.employed' : ['nr_employed'] 
    })

prediction = model_pipeline.predict_proba(input_df)[0][1]
    result = "Likely to Subscribe ✅" if prediction >= 0.3 else "Not Likely to Subscribe ❌"

    st.subheader("Prediction:")
    st.write(f"**Probability of Subscription:** {prediction:.2%}")
    st.write(result)
