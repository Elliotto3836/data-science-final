import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

#streamlit run streamlit_app.py --server.enableCORS false --server.enableXsrfProtection false

import mlflow
import dagshub



from sklearn.tree import export_graphviz



st.title("Data Science Final")

df_original = pd.read_csv("airline.csv")



# Turns categorical variables into numbers
#df is the one with numbers, original has the label
df=df_original.copy()
label_encoders = {}

for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le



app_mode = st.sidebar.selectbox("Select a page",["Business Case and Data Presentation","Data Visualization","Logistic Regression","Decision Tree 🌳","Feature Importance and Driving Variables","Hyperparameter Tuning"])


if app_mode == "Business Case and Data Presentation":
    st.markdown("# :blue[📊 Introduction:]")
    st.write("We will be analyzing the Airplane data, we will be predicting whether or not a customer is a loyal or disloyal customer")

    st.dataframe(df_original.head(5))

if app_mode == "Data Visualization":
    st.dataframe(df.head(5))

if app_mode == "Logistic Regression":
    st.dataframe(df.head(5))
    sns.countplot(data=df,x="satisfaction")
    st.pyplot()
    df2= df.drop('id',axis=1)
    fig, ax = plt.subplots(figsize=(60, 50)) 
    sns.heatmap(df2.corr(), annot=True, ax=ax)
    st.pyplot(fig)
    df2 = df2.dropna()
    X = df2.drop("satisfaction",axis=1)
    y = df2["satisfaction"]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)
    log = LogisticRegression()
    log.fit(X_train,y_train)
    predictions = log.predict(X_test)
    
    from sklearn.metrics import classification_report
    st.write(classification_report(predictions,y_test))


    





if app_mode == "Decision Tree 🌳":
    st.markdown("# :blue[📊 Introduction:]")
    X_tree = df.drop(["satisfaction"],axis=1)
    y_tree = df["satisfaction"] # Target variable

    X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(X_tree, y_tree, test_size=0.2, random_state=1)
    #init
    clf = DecisionTreeClassifier(max_depth=3)
    #train
    clf = clf.fit(X_train_tree,y_train_tree)
    #predict
    y_pred_tree = clf.predict(X_test_tree)

    st.write("Accuracy:",metrics.accuracy_score(y_test_tree, y_pred_tree))

    feature_names = X_tree.columns
    feature_cols = X_test_tree.columns
    dot_data = export_graphviz(clf, out_file=None,feature_names=feature_cols,class_names=['0','1'],filled=True, rounded=True,special_characters=True)
    
    graph = graphviz.Source(dot_data)
    st.graphviz_chart(graph)

if app_mode == "Feature Importance and Driving Variables":
    st.dataframe(df.head(5))

if app_mode == "Hyperparameter Tuning":
    st.dataframe(df.head(5))

