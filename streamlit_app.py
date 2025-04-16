import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
import graphviz


#git add .
#git commit -am "Message here"
#git push

#Make sure to import everything

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
    st.markdown("# :blue[Decision Tree 🌳]")
    
    # Prepare features and target
    X_tree = df.drop(["satisfaction"], axis=1)
    y_tree = df["satisfaction"]

    # Train-test split
    X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(X_tree, y_tree, test_size=0.2, random_state=1)

    # Hyperparameter tuning: searching for best max_depth
    param_grid = {"max_depth": list(range(1, 21))}  # Search from depth 1 to 20
    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=1),
        param_grid,
        cv=5,
        scoring='accuracy'
    )
    grid_search.fit(X_train_tree, y_train_tree)


    # Use the best model from grid search
    best_clf = grid_search.best_estimator_
    y_pred_tree = best_clf.predict(X_test_tree)
    

    # Get results from GridSearchCV
    results = grid_search.cv_results_
    max_depths = results["param_max_depth"]
    mean_scores = results["mean_test_score"]

    fig, ax = plt.subplots()
    ax.set_xticks(range(min(max_depths), max(max_depths)+1))
    ax.plot(max_depths, mean_scores, marker='o')
    ax.set_title("Decision Tree Accuracy vs. Max Depth")
    ax.set_xlabel("Max Depth")
    ax.set_ylabel("Cross-Validated Accuracy")
    ax.grid(True)

    # Show in Streamlit
    st.pyplot(fig)


    # Display results
    st.write("Best max_depth:", grid_search.best_params_["max_depth"])
    st.write("Accuracy with best max_depth:", metrics.accuracy_score(y_test_tree, y_pred_tree))
    st.write("For this page we will be predicting whether someone will return to an airline based on a decision tree.")
    st.write("This is the decision tree for the more efficient max depth of 10:")

    # Visualize decision tree
    feature_names = X_tree.columns
    feature_cols = X_test_tree.columns
    dot_data = export_graphviz(
        best_clf,
        out_file=None,
        feature_names=feature_cols,
        class_names=['0', '1'],
        filled=True,
        rounded=True,
        special_characters=True
    )
    

    graph = graphviz.Source(dot_data)
    st.graphviz_chart(graph)

    #doing max depth of 3
    clf_depth3 = DecisionTreeClassifier(max_depth=3, random_state=1)
    clf_depth3.fit(X_train_tree, y_train_tree)
    y_pred_depth3 = clf_depth3.predict(X_test_tree)

    # Display results
    st.markdown("Decision Tree with max_depth = 3")
    st.write("Accuracy:", metrics.accuracy_score(y_test_tree, y_pred_depth3))

    # Visualize tree
    dot_data_depth3 = export_graphviz(
        clf_depth3,
        out_file=None,
        feature_names=X_test_tree.columns,
        class_names=['0', '1'],
        filled=True,
        rounded=True,
        special_characters=True
    )
    graph_depth3 = graphviz.Source(dot_data_depth3)
    st.graphviz_chart(graph_depth3)



if app_mode == "Feature Importance and Driving Variables":
    st.dataframe(df.head(5))

if app_mode == "Dagshub + MLFlow":

    st.dataframe(df.head(5))

