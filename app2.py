import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# App Title
st.title("Automated Machine Learning! 😉")

# File Upload
uploaded_file = st.file_uploader("Upload your file (CSV or Excel)", type=["csv", "xlsx"])

# Initialize session state for df if it's not already initialized
if 'df' not in st.session_state:
    st.session_state.df = None

if uploaded_file:
    # Read the uploaded file
    file_extension = uploaded_file.name.split(".")[-1]
    if file_extension == "csv":
        st.session_state.df = pd.read_csv(uploaded_file)
    elif file_extension == "xlsx":
        st.session_state.df = pd.read_excel(uploaded_file)

    df = st.session_state.df  # Use the df from session state

    # Options for Head, Tail, Random Sample, Shape
    st.subheader("Data Insights")
    option = st.selectbox("Choose an option to view data", 
                          ["Head", "Tail", "Random Sample", "Number of Rows and Columns"])
    
    if option == "Head":
        st.write(df.head())
    elif option == "Tail":
        st.write(df.tail())
    elif option == "Random Sample":
        st.write(df.sample(5))
    elif option == "Number of Rows and Columns":
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # Basic Info and Statistics
    st.subheader("Basic Information and Statistics")
    info_option = st.selectbox("Choose an option", ["Info", "Statistics"])
    
    if info_option == "Info":
        st.write("### Dataset Information:")
        info_dict = {
            # "Column": df.columns,
            "Non-Null Count": df.notnull().sum(),
            "Data Type": [str(dtype) for dtype in df.dtypes],
        }
        info_df = pd.DataFrame(info_dict)
        st.dataframe(info_df)

    elif info_option == "Statistics":
        st.write(df.describe())

    # Checking for Missing and Duplicate Values
    st.subheader("Null and Duplicate Values")
    if st.checkbox("Show Missing Values Percentage"):
        missing_percentage = df.isnull().sum() * 100 / len(df)
        st.write(missing_percentage)

    if st.checkbox("Check for Duplicate Rows"):
        duplicate_count = df.duplicated().sum()
        if duplicate_count == 0:
            st.write("There are no duplicate rows.")
        else:
            st.write(f"Number of duplicate rows: {duplicate_count}")

    # Data Cleaning Options
    st.subheader("Data Cleaning")
    if st.checkbox("Remove Duplicate Rows"):
        st.session_state.df = df.drop_duplicates()  # Update df in session state
        df = st.session_state.df  # Use updated df from session state
        st.success("Duplicate rows removed!")

    if st.checkbox("Drop Columns"):
        columns_to_drop = st.multiselect("Select columns to drop", df.columns)
        if st.button("Drop Selected Columns"):
            if columns_to_drop:
                st.session_state.df = df.drop(columns=columns_to_drop)  # Update df in session state
                df = st.session_state.df  # Use updated df from session state
                st.success(f"Columns {columns_to_drop} dropped successfully!")
            else:
                st.warning("No columns selected to drop.")

    if st.checkbox("Drop Rows with Missing Values"):
        if st.button("Drop Rows"):
            st.session_state.df = df.dropna()  # Update df in session state
            df = st.session_state.df  # Use updated df from session state
            st.success("Rows with missing values dropped!")

    # Updated Data Section (After all cleaning)
    st.subheader("Updated Dataset (After Cleaning)")
    st.write(df)  # This should reflect the updated DataFrame

    # Visualization
    st.subheader("Data Visualization")
    chart_type = st.selectbox(
        "Select Visualization Type",
        ["Scatter Plot", "Box Plot", "Pie Chart", "Bar Plot", "Distribution Plot", "Violin Plot", "Swarm Plot", "Count Plot"]
    )

    x_axis = st.selectbox("Select X-axis", df.columns)
    y_axis = st.selectbox("Select Y-axis (Optional for some plots)", ["None"] + list(df.columns))
    hue = st.selectbox("Select Hue (Optional)", ["None"] + list(df.columns))

    if st.button("Generate Chart"):
        plt.figure(figsize=(10, 6))

        if chart_type == "Scatter Plot":
            sns.scatterplot(x=x_axis, y=y_axis if y_axis != "None" else None, hue=hue if hue != "None" else None, data=df)
            st.pyplot(plt)

        elif chart_type == "Box Plot":
            sns.boxplot(x=x_axis, y=y_axis if y_axis != "None" else None, hue=hue if hue != "None" else None, data=df)
            st.pyplot(plt)

        elif chart_type == "Pie Chart":
            if df[x_axis].nunique() > 10:
                st.warning("Pie Chart is not suitable for large categories.")
            else:
                pie_data = df[x_axis].value_counts()
                plt.pie(pie_data, labels=pie_data.index, autopct="%1.1f%%")
                st.pyplot(plt)

        elif chart_type == "Bar Plot":
            sns.barplot(x=x_axis, y=y_axis if y_axis != "None" else None, hue=hue if hue != "None" else None, data=df)
            st.pyplot(plt)

        elif chart_type == "Distribution Plot":
            sns.histplot(x=x_axis, hue=hue if hue != "None" else None, data=df, kde=True)
            st.pyplot(plt)

        elif chart_type == "Violin Plot":
            sns.violinplot(x=x_axis, y=y_axis if y_axis != "None" else None, hue=hue if hue != "None" else None, data=df, split=True)
            st.pyplot(plt)

        elif chart_type == "Swarm Plot":
            sns.swarmplot(x=x_axis, y=y_axis if y_axis != "None" else None, hue=hue if hue != "None" else None, data=df)
            st.pyplot(plt)

        elif chart_type == "Count Plot":
            sns.countplot(x=x_axis, hue=hue if hue != "None" else None, data=df)
            st.pyplot(plt)
##################################################################################################
# ... [Previous code remains unchanged until the Visualization section]

    # Visualization code here...

    # Feature Engineering moved before Prepare Features and Target
    st.subheader("Feature Engineering")

    # Encoding
    if st.checkbox("Apply Encoding to Categorical Features"):
        df = st.session_state.df.copy()  # Get the latest DataFrame
        # Identify all categorical columns in the dataset
        categorical_features = [col for col in df.columns if df[col].dtype == 'object']
        if categorical_features:
            encoding_method = st.selectbox("Select Encoding Method", ["One-Hot Encoding", "Label Encoding"])
            if encoding_method == "One-Hot Encoding":
                # Apply one-hot encoding to all categorical features at once
                df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
            elif encoding_method == "Label Encoding":
                le = LabelEncoder()
                for feature in categorical_features:
                    df[feature] = le.fit_transform(df[feature])
            # Update session state with the modified DataFrame
            st.session_state.df = df.copy()
            st.success("Categorical features encoded successfully!")
        else:
            st.warning("No categorical features found in the dataset.")

    # Feature Scaling
    if st.checkbox("Apply Feature Scaling"):
        df = st.session_state.df.copy()  # Get the latest DataFrame
        numerical_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if numerical_features:
            scaling_method = st.selectbox("Select Scaling Method", ["Standardization (Z-Score)", "Normalization (Min-Max)"])
            scaler = StandardScaler() if scaling_method == "Standardization (Z-Score)" else MinMaxScaler()
            df[numerical_features] = scaler.fit_transform(df[numerical_features])
            # Update session state with the scaled DataFrame
            st.session_state.df = df.copy()
            st.success("Feature scaling applied successfully!")
        else:
            st.warning("No numerical features found in the dataset.")

    # Prepare Features and Target after feature engineering
    st.subheader("Prepare Features and Target")
    df = st.session_state.df  # Use the latest DataFrame from session state
    features = st.multiselect("Select Features (Input Variables):", df.columns)
    target = st.selectbox("Select Target (Output Variable):", df.columns)

    # Ensure features and target are selected
    if features and target:
        X = df[features]
        y = df[target]

        # Train-Test Split and Model Training remain unchanged...
        # ... [Rest of the code remains the same]
###############################################################################################
        # Step 2: Train-Test Split
        st.subheader("Train-Test Split")
        test_size = st.slider("Test Set Size (as % of data)", 10, 50, 20, step=5) / 100.0
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        st.write("Data split successfully. Ready for model training!")

        # Step 3: Model Selection
        st.subheader("Model Selection")
        problem_type = st.radio("Problem Type:", ["Regression", "Classification"], index=0)

        models = {
            "Regression": {
                "Linear Regression": LinearRegression(),
                "Random Forest Regressor": RandomForestRegressor(),
                "Support Vector Regressor (SVR)": SVR(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "K-Neighbors Regressor (KNN)": KNeighborsRegressor(),
            },
            "Classification": {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest Classifier": RandomForestClassifier(),
                "Support Vector Classifier (SVC)": SVC(),
                "Decision Tree Classifier": DecisionTreeClassifier(),
                "K-Neighbors Classifier (KNN)": KNeighborsClassifier(),
                "Naive Bayes Classifier": GaussianNB(),
            },
        }

        model_name = st.selectbox("Choose a Model:", list(models[problem_type].keys()))
        model = models[problem_type][model_name]


        # Step 4: Train Model
        if st.button("Train Model"):
            model.fit(X_train, y_train)
            st.session_state.ml_model = model
            st.success(f"{model_name} has been trained successfully!")

            # Step 5: Model Evaluation
            if st.session_state.ml_model:
                st.subheader("Model Evaluation")
                y_pred = model.predict(X_test)

                if problem_type == "Regression":
                    st.write("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
                    st.write("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
                    st.write("R-squared Score (R²):", r2_score(y_test, y_pred))
                elif problem_type == "Classification":
                    st.write("Accuracy:", accuracy_score(y_test, y_pred))
                    st.write("Precision:", precision_score(y_test, y_pred, average="weighted"))
                    st.write("Recall:", recall_score(y_test, y_pred, average="weighted"))
                    st.write("F1 Score:", f1_score(y_test, y_pred, average="weighted"))

                # # Step 6: Predictions
                # st.subheader("Make a Prediction")
                # if features:
                #     test_inputs = {}
                #     for feature in features:
                #         test_inputs[feature] = st.number_input(f"Input value for {feature}:", key=f"input_{feature}")

                #     if st.button("Predict"):
                #         if st.session_state.ml_model:
                #             test_df = pd.DataFrame([test_inputs])  # Create a test DataFrame
                #             try:
                #                 prediction = st.session_state.ml_model.predict(test_df)
                #                 st.success(f"Prediction Result: {prediction[0]}")
                #             except Exception as e:
                #                 st.error(f"Error during prediction: {e}")
                #         else:
                #             st.warning("Please train a model first.")