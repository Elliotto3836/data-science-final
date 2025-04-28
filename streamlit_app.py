import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pycaret.classification import setup, compare_models, pull
import os
from streamlit_extras.let_it_rain import rain
from streamlit_extras.dataframe_explorer import dataframe_explorer

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

#For the rainbow 
st.markdown("""
<style>
@keyframes rainbow {
  0%{color: red;}
  14%{color: orange;}
  28%{color: yellow;}
  42%{color: green;}
  57%{color: blue;}
  71%{color: indigo;}
  85%{color: violet;}
  100%{color: red;}
}

.rainbow-text {
  font-size: 48px;
  font-weight: bold;
  animation: rainbow 4s infinite;
  text-align: center;
}

/* Center the div itself */
.center {
  display: flex;
  justify-content: center;
  align-items: center;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="center"><div class="rainbow-text">Airline Satisfaction!</div></div>', unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: skyblue;'>By Elliot, Katie, and Kanishk</h1>", unsafe_allow_html=True)

st.image("HighQualityAirfrance.png")

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



app_mode = st.sidebar.selectbox("Select a page",["Business Case and Data Presentation","Data Visualization","Logistic Regression","Decision Tree 🌳","Feature Importance / AI Explainability", "AutoML with PyCaret"])


if app_mode == "Business Case and Data Presentation":
    rain(emoji="🛩️",font_size=54,falling_speed=5,animation_length="10",)
    st.markdown("# :blue[📊 Introduction:]")
    st.write("We will be analyzing the Airplane data, we will be predicting whether or not a customer will be satisfied or not. This would be extremely" \
    "helpful for an airpline company to determine what they need to prioritize.")
    num = st.slider("Select number of rows to view", min_value=5, max_value=100, value=10)
    st.dataframe(df_original.head(num))
    st.write("(From this point on, we will convert the non-numerical variables to numerical variables through the label encoder function for the purposes of data presentation and model prediction.)")
    st.markdown("## :blue[🔍 Description of the Data]")
    st.dataframe(df.describe())

    st.markdown("## :blue[Filter Data]")
    filtered_df = dataframe_explorer(df_original, case=False)
    st.dataframe(filtered_df, use_container_width=True)

    st.markdown("## :blue[Variables Used]")

    features = {
        ":blue[Gender]": "",
        ":blue[Customer Type]": "",
        ":blue[Age]": "",
        ":blue[Type of Travel]": "",
        ":blue[Class]": "",
        ":blue[Flight Distance]": "",
        ":blue[Inflight wifi service]": "",
        ":blue[Departure/Arrival time convenient]": "",
        ":blue[Ease of Online booking]": "",
        ":blue[Gate location]": "",
        ":blue[Food and drink]": "",
        ":blue[Online boarding]": "",
        ":blue[Seat comfort]": "",
        ":blue[Inflight entertainment]": "",
        ":blue[On-board service]": "",
        ":blue[Leg room service]": "",
        ":blue[Baggage handling]": "",
        ":blue[Checkin service]": "",
        ":blue[Inflight service]": "",
        ":blue[Cleanliness]": "",
        ":blue[Departure Delay in Minutes]": "",
        ":blue[Arrival Delay in Minutes]": "",
        ":blue[Satisfaction]": ""
    }
    
    for key, value in features.items():
        st.markdown(f"- **{key}** - {value}")

    st.markdown("## :blue[Rows, Columns]")
    st.write(df.shape)

    st.markdown("## :blue[Pandas Profiling Report]")
    
    #profile = ProfileReport(df, minimal=True)
    #profile.to_file("data_report.html")
    #html = profile.to_html()

    html_file_path = "data_report.html"

    # Check if the file exists
    if os.path.exists(html_file_path):
        # Open the HTML file and read its content
        with open(html_file_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Display the HTML content in Streamlit
        st.components.v1.html(html_content, height=800,scrolling=True)  # Adjust height as needed
    else:
        st.write("HTML report not found. Please generate the report first.") 


if app_mode == "Data Visualization":
    st.dataframe(df.head(5))

if app_mode == "Logistic Regression":
   
    st.dataframe(df.head(5))
    fig2, ax = plt.subplots()
    sns.countplot(data=df,x="Satisfaction")
    st.pyplot(fig2)
    
    
    df2= df.drop('id',axis=1)
    default_vars = ['Age','Satisfaction','Type of Travel']
    all_vars = ['Gender',
               'Customer Type',
               'Age',
               'Type of Travel',
                'Class',
               'Flight Distance',
               'Inflight wifi service',
               'Departure/Arrival time convenient',
               'Ease of Online booking',
               'Gate location',
               'Food and drink',
               'Online boarding',
               'Seat comfort',
              'Inflight entertainment',
               'On-board service',
               'Leg room service',
               'Baggage handling',
               'Checkin service',
               'Inflight service',
               'Cleanliness',
               'Departure Delay in Minutes',
               'Arrival Delay in Minutes',
               'Satisfaction']
    other_vars = [var for var in all_vars if var not in default_vars]
    additional_vars = st.multiselect(
    "Add more variables to the correlation matrix:",
    options=other_vars)
    selected_vars = default_vars + additional_vars
    fig_width = max(8,len(selected_vars))
    fig_height = max(6,len(selected_vars))
    corr_matrix = df2[selected_vars].corr()
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
    st.pyplot(fig)
    
    df2 = df2.dropna()
    X = df2.drop("Satisfaction",axis=1)
    y = df2["Satisfaction"]
   
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)
    log = LogisticRegression()
    log.fit(X_train,y_train)
    predictions = log.predict(X_test)
    
    from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

    accuracy = accuracy_score(y_test, predictions)

    st.markdown("### Model Evaluation Metrics")

    st.write(f"**Accuracy**: {accuracy:.4f}")



if app_mode == "Decision Tree 🌳":

    st.markdown("# :blue[Decision Tree 🌳]")
    # Prepare features and target

    #First just doing 3 and 10 as examples. Since 3 is a good graph and 10 is the most accurate
    max_depth_user = st.slider("Select max_depth for Decision Tree:", min_value=1, max_value=20, value=3)

    X_tree = df.drop(["Satisfaction"], axis=1)
    y_tree = df["Satisfaction"]

    # Train-test split
    X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(X_tree, y_tree, test_size=0.2, random_state=1)

    model = DecisionTreeClassifier(max_depth=max_depth_user, random_state=1)
    model.fit(X_train_tree, y_train_tree)
    y_pred_tree = model.predict(X_test_tree)

    # Metrics
    mae = mean_absolute_error(y_test_tree, y_pred_tree)
    mse = mean_squared_error(y_test_tree, y_pred_tree)
    rmse = np.sqrt(mse)
    accuracy = metrics.accuracy_score(y_test_tree, y_pred_tree)

    st.success(f"Accuracy: {accuracy:.4f}")
    st.success(f"mae: {mae:.4f}")
    st.success(f"mse: {mse:.4f}")

    dot_data = export_graphviz(
    model,
    out_file=None,
    feature_names=X_tree.columns,
    class_names=['0', '1'],
    filled=True,
    rounded=True,
    special_characters=True)
    
    graph = graphviz.Source(dot_data)
    st.graphviz_chart(graph)



    #max depth 10

    st.markdown("-----")
    st.markdown("This is for max_depth = 10, which was the most accurate according to my hyperparameter tuning")


    model = DecisionTreeClassifier(max_depth=10, random_state=1)
    model.fit(X_train_tree, y_train_tree)
    y_pred_tree = model.predict(X_test_tree)

    # Metrics
    mae = mean_absolute_error(y_test_tree, y_pred_tree)
    mse = mean_squared_error(y_test_tree, y_pred_tree)
    rmse = np.sqrt(mse)
    accuracy = metrics.accuracy_score(y_test_tree, y_pred_tree)

    st.success(f"Accuracy: {accuracy:.4f}")
    st.success(f"mae: {mae:.4f}")
    st.success(f"mse: {mse:.4f}")

    dot_data = export_graphviz(
    model,
    out_file=None,
    feature_names=X_tree.columns,
    class_names=['0', '1'],
    filled=True,
    rounded=True,
    special_characters=True)
    
    graph = graphviz.Source(dot_data)
    st.graphviz_chart(graph)

    st.markdown("-----")

    st.markdown("# Decision Tree Accuracy vs Max Depth")
    
    st.image("DecisionTreeAccuracy.png")

    st.markdown("## Mlflow Results")

    st.image("MLFlowData.png")




    if st.button("Run Full Decision Tree Hyperparameter Tuning with MLFlow + Dagshub"):
        # Initialize DagsHub and MLflow
        dagshub.init(repo_owner='Elliotto3836', repo_name='Final', mlflow=True)

        X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(X_tree, y_tree, test_size=0.2, random_state=1)

        # Hyperparameter tuning: searching for best max_depth
        param_grid = {"max_depth": list(range(1, 21))}  # Search from depth 1 to 20
        grid_search = GridSearchCV(
            DecisionTreeClassifier(random_state=1),
            param_grid,
            cv=5,
            scoring='accuracy'
        )


        # Start an MLflow experiment
        mlflow.start_run()

        grid_search.fit(X_train_tree, y_train_tree)

        # Loop through each value of max_depth and create a new nested run for each iteration
        for max_depth in param_grid["max_depth"]:
            name = "Decision Tree " + str(max_depth)
            with mlflow.start_run(nested=True,run_name=name):  # This creates a new nested run for each max_depth
                # Create and fit the model with the current max_depth
                model = DecisionTreeClassifier(max_depth=max_depth, random_state=1)
                model.fit(X_train_tree, y_train_tree)
                y_pred_tree = model.predict(X_test_tree)

                # Metrics
                mae = mean_absolute_error(y_test_tree, y_pred_tree)
                mse = mean_squared_error(y_test_tree, y_pred_tree)
                rmse = np.sqrt(mse)
                accuracy = metrics.accuracy_score(y_test_tree, y_pred_tree)

                # Log parameters and metrics
                mlflow.log_param("model_name", "DecisionTree")
                mlflow.log_param("max_depth", max_depth)
                mlflow.log_params(model.get_params())
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("rmse", rmse)

                # Log model
                mlflow.sklearn.log_model(model, artifact_path="model")

                # Display metrics in Streamlit
                st.markdown(f'<p class="sub-title">Decision Tree with max_depth={max_depth}</p>', unsafe_allow_html=True)
                st.write(f"MAE: {mae:.3f}")
                st.write(f"MSE: {mse:.3f}")
                st.write(f"RMSE: {rmse:.3f}")
                st.write(f"Accuracy: {accuracy:.3f}")

                # Visualize decision tree
                dot_data = export_graphviz(
                    model,
                    out_file=None,
                    feature_names=X_tree.columns,
                    class_names=['0', '1'],
                    filled=True,
                    rounded=True,
                    special_characters=True)

                graph = graphviz.Source(dot_data)
                st.graphviz_chart(graph)

        # End the MLflow run
        mlflow.end_run()

        best_max_depth = grid_search.best_params_["max_depth"]
        best_accuracy = grid_search.best_score_
        st.write("Best max_depth:", best_max_depth)
        st.write("Best accuracy from GridSearch:", best_accuracy)

        results = grid_search.cv_results_
        max_depths = results["param_max_depth"]
        mean_scores = results["mean_test_score"]

        fig, ax = plt.subplots()
        ax.set_xticks(range(min(max_depths), max(max_depths) + 1))
        ax.plot(max_depths, mean_scores, marker='o')
        ax.set_title("Decision Tree Accuracy vs. Max Depth")
        ax.set_xlabel("Max Depth")
        ax.set_ylabel("Cross-Validated Accuracy")
        ax.grid(True)

        st.pyplot(fig)



if app_mode == "Dagshub + MLFlow":

    st.dataframe(df.head(5))

if app_mode == "Feature Importance / AI Explainability":
    import shap
    from streamlit_shap import st_shap
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    st.markdown("### AI Explainability ")

    dfShap = df.dropna()

    X = dfShap.drop("Satisfaction", axis=1)
    y = dfShap["Satisfaction"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train_SHAP, X_test_SHAP, y_train_SHAP, y_test_SHAP = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000, random_state=1)
    model.fit(X_train_SHAP, y_train_SHAP)

    explainer = shap.Explainer(model, X_train_SHAP)
    shap_values = explainer(X_test_SHAP)

    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values.values, X_test_SHAP, feature_names=X.columns, plot_type="dot", show=False)

    plt.title("Feature Importance for the Satisfaction Prediction (Logistic Regression)", fontsize=16)

    st.pyplot(fig)


if app_mode == "AutoML with PyCaret":
    st.title("🔮 AutoML with PyCaret")

    st.markdown("""PyCaret will automatically try different models and then select the best one.""")

    if st.button("Test Multiple Models using PyCaret"):

        # Prepare the data
        df_pycaret = df.dropna()  # PyCaret needs no missing values
        df_pycaret = df_pycaret.drop(columns=['id'])  # Drop id column if still exists

        # Setup PyCaret
        clf = setup(data=df_pycaret, target='Satisfaction', session_id=123)

        # Compare different models
        best_model = compare_models()

        # Show the comparison table
        st.markdown("### 📋 Model Comparison Results")
        comparison_df = pull()
        #comparison_df = comparison_df[['Model', 'Accuracy', 'Kappa', 'TT', 'PT']]
        comparison_df = comparison_df.sort_values(by='Accuracy', ascending=False)

        st.dataframe(comparison_df)



        # Show the best model
        st.markdown("### 🏆 Best Model Found")
        st.write(best_model)
