import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
#streamlit run streamlit_app.py --server.enableCORS false --server.enableXsrfProtection false

import mlflow
import dagshub


st.title("Data Science Final")

st.markdown("# :blue[📊 Introduction:]")
st.write("We will be analyzing the Airplane data, we will be predicting whether or not a customer is a loyal or disloyal customer")


df = pd.read_csv("airline.csv")

app_mode = st.sidebar.selectbox("Select a page",["Business Case and Data Presentation","Data Visualization","Logistic Regression","Decision Tree","Feature Importance and Driving Variables","Hyperparameter Tuning"])

if app_mode == "Business Case and Data Presentation":
    st.dataframe(df.head(5))

if app_mode == "Data Visualization":
    st.dataframe(df.head(5))

if app_mode == "Logistic Regression":
    st.dataframe(df.head(5))

if app_mode == "Decision Tree":
    st.dataframe(df.head(5))

if app_mode == "Feature Importance and Driving Variables":
    st.dataframe(df.head(5))

if app_mode == "Hyperparameter Tuning":
    st.dataframe(df.head(5))

