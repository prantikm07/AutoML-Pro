import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# App Title
st.title("AutoML Pro 🚀")

# File Upload
uploaded_file = st.file_uploader("Upload your file (CSV or Excel)", type=["csv", "xlsx"])

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.update({
        'df': None,
        'encoders': {},
        'scaler': None,
        'ml_model': None,
        'preprocessor': None,
        'features': [],
        'target': '',
        'problem_type': ''
    })

# Helper function to download files
def create_download_link(object_to_download, filename):
    buffer = BytesIO()
    joblib.dump(object_to_download, buffer)
    buffer.seek(0)
    st.download_button(label=f"Download {filename}", data=buffer, file_name=filename)

if uploaded_file:
    # Read the uploaded file
    file_extension = uploaded_file.name.split(".")[-1]
    if file_extension == "csv":
        st.session_state.df = pd.read_csv(uploaded_file)
    elif file_extension == "xlsx":
        st.session_state.df = pd.read_excel(uploaded_file)

    df = st.session_state.df

    # Data Exploration
    with st.expander("📊 Data Exploration", expanded=True):
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

    # Data Cleaning
    with st.expander("🧹 Data Cleaning", expanded=False):
        st.subheader("Clean the Data")
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
    with st.expander("🎨 Data Visualization", expanded=False):
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


    # Feature Engineering
    with st.expander("⚙️ Feature Engineering", expanded=False):
        # Encoding
        if st.checkbox("Apply Encoding to Categorical Features"):
            categorical_features = df.select_dtypes(include=['object']).columns.tolist()
            if categorical_features:
                encoding_method = st.selectbox("Select Encoding Method", ["Label Encoding", "One-Hot Encoding"])
                
                # In the Feature Engineering section under Label Encoding
                if encoding_method == "Label Encoding":
                    for feature in categorical_features:
                        le = LabelEncoder()
                        df[feature] = le.fit_transform(df[feature])
                        st.session_state.encoders[feature] = le  # Store encoder per feature
                    st.success("Label encoding applied successfully!")
                
                elif encoding_method == "One-Hot Encoding":
                    ct = ColumnTransformer(
                        [('onehot', OneHotEncoder(drop='first'), categorical_features)],
                        remainder='passthrough'
                    )
                    df_encoded = ct.fit_transform(df)
                    df = pd.DataFrame(df_encoded, columns=ct.get_feature_names_out())
                    st.session_state.encoders['onehot'] = ct
                    st.success("One-hot encoding applied successfully!")
                
                st.session_state.df = df

        # # Feature Scaling
        # if st.checkbox("Apply Feature Scaling"):
        #     numerical_features = df.select_dtypes(include=['number']).columns.tolist()
        #     if numerical_features:
        #         scaling_method = st.selectbox("Select Scaling Method", ["Standardization (Z-Score)", "Normalization (Min-Max)"])
                
        #         scaler = StandardScaler() if scaling_method == "Standardization (Z-Score)" else MinMaxScaler()
        #         df[numerical_features] = scaler.fit_transform(df[numerical_features])
        #         st.session_state.scaler = scaler
        #         st.success("Feature scaling applied successfully!")
        #         st.session_state.df = df

    # Model Configuration
    with st.expander("🤖 Model Configuration", expanded=True):
        st.subheader("Prepare Features and Target")
        target = st.selectbox("Select Target (Output Variable):", df.columns)
        features = st.multiselect("Select Features (Input Variables):", df.columns.drop(target))
        problem_type = st.radio("Problem Type:", ["Regression", "Classification"], index=0)
        
        if features and target:
            X = df[features]
            y = df[target]

            # Train-Test Split
            st.subheader("Train-Test Split")
            test_size = st.slider("Test Set Size (as % of data)", 10, 50, 20, step=5) / 100.0
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # Model Selection
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
            base_model = models[problem_type][model_name]

            # Hyperparameter Tuning
            st.subheader("Advanced Settings")
            hp_tuning = st.checkbox("Enable Hyperparameter Tuning")
            tuned_model = base_model
            
            if hp_tuning:
                if "Random Forest" in model_name:
                    n_estimators = st.slider("Number of Trees", 10, 200, 100)
                    max_depth = st.number_input("Max Depth", min_value=1, max_value=50, value=20)
                    
                    if problem_type == "Regression":
                        tuned_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                    else:
                        tuned_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                
                elif "SVC" in model_name or "SVR" in model_name:
                    C = st.number_input("Regularization (C)", 0.01, 10.0, 1.0)
                    kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
                    
                    if problem_type == "Regression":
                        tuned_model = SVR(C=C, kernel=kernel)
                    else:
                        tuned_model = SVC(C=C, kernel=kernel, probability=True)
                
                elif "K-Neighbors" in model_name:
                    n_neighbors = st.slider("Number of Neighbors", 1, 15, 5)
                    
                    if problem_type == "Regression":
                        tuned_model = KNeighborsRegressor(n_neighbors=n_neighbors)
                    else:
                        tuned_model = KNeighborsClassifier(n_neighbors=n_neighbors)

            # Model Training
            if st.button("🎯 Train Model"):
                try:
                    tuned_model.fit(X_train, y_train)
                    st.session_state.ml_model = tuned_model
                    st.success(f"{model_name} trained successfully!")
                    
                    # Model Evaluation
                    st.subheader("📈 Evaluation Metrics")
                    y_pred = tuned_model.predict(X_test)
                    
                    if problem_type == "Regression":
                        metrics = {
                            "MSE": f"{mean_squared_error(y_test, y_pred):.4f}",
                            "MAE": f"{mean_absolute_error(y_test, y_pred):.4f}",
                            "R² Score": f"{r2_score(y_test, y_pred):.4f}"
                        }
                    else:
                        metrics = {
                            "Accuracy": f"{accuracy_score(y_test, y_pred):.4f}",
                            "Precision": f"{precision_score(y_test, y_pred, average='weighted'):.4f}",
                            "Recall": f"{recall_score(y_test, y_pred, average='weighted'):.4f}",
                            "F1 Score": f"{f1_score(y_test, y_pred, average='weighted'):.4f}"
                        }
                    
                    # Display metrics in columns
                    cols = st.columns(len(metrics))
                    for (name, value), col in zip(metrics.items(), cols):
                        col.metric(label=name, value=value)

                    # Model Download
                    st.subheader("💾 Model Export")
                    create_download_link(tuned_model, "trained_model.pkl")

                except Exception as e:
                    st.error(f"Error training model: {str(e)}")

        st.write(df) 

    # Prediction Interface
    if st.session_state.ml_model:
        with st.expander("🔮 Make Predictions", expanded=True):
            st.subheader("Live Prediction")
            input_data = {}

            for feature in features:
                # Handle encoded categorical features
                if feature in st.session_state.encoders:
                    encoder = st.session_state.encoders[feature]
                    input_val = st.selectbox(f"Select {feature}", encoder.classes_)
                    input_data[feature] = encoder.transform([input_val])[0]
                else:
                    # Handle numerical features
                    default_val = df[feature].mean() if feature in df.columns else 0.0
                    input_data[feature] = st.number_input(f"Enter {feature}", value=default_val)

            if st.button("🔮 Predict"):
                try:
                    # Create DataFrame from user inputs
                    input_df = pd.DataFrame([input_data])

                    # Apply scaling if scaler exists
                    if st.session_state.scaler:
                        num_features = input_df.select_dtypes(include=['number']).columns
                        input_df[num_features] = st.session_state.scaler.transform(input_df[num_features])

                    # Make prediction
                    prediction = st.session_state.ml_model.predict(input_df)

                    # Display result
                    # st.success(f"**Prediction Result:** {prediction[0]:.2f}")
                    st.success(f"**Prediction Result:** {prediction[0]}")

                    # For classification, display confidence
                    if problem_type == "Classification":
                        proba = st.session_state.ml_model.predict_proba(input_df)
                        st.write(f"**Confidence:** {proba.max():.2%}")

                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")


# Add a footer
st.markdown("---")
st.markdown("Built with ❤️ by Prantik | AutoML Pro v1.0")