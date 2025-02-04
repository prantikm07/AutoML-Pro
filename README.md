# AutoML Pro 🚀

This is a Streamlit-based application that provides a user-friendly interface for automated machine learning.  It simplifies the process of data exploration, cleaning, visualization, feature engineering, model selection, training, evaluation, and prediction.  Users can upload their data (CSV or Excel), select a target variable, choose features, and train various machine learning models for both regression and classification tasks.

## Features

* **Data Upload:** Supports CSV and Excel file formats.
* **Data Exploration:** Provides options to view data head, tail, random samples, and shape. Displays basic information (data types, non-null counts) and descriptive statistics.
* **Data Cleaning:** Includes functionalities to remove duplicate rows, drop columns, and handle missing values.
* **Data Visualization:** Offers a variety of chart types (Scatter Plot, Box Plot, Pie Chart, Bar Plot, Distribution Plot, Violin Plot, Swarm Plot, Count Plot) for data visualization.
* **Feature Engineering:** Supports label encoding and one-hot encoding for categorical features.  *(Feature scaling is currently commented out but can be easily re-enabled)*
* **Model Selection:** Offers a range of regression (Linear Regression, Random Forest Regressor, SVR, Decision Tree Regressor, K-Neighbors Regressor) and classification (Logistic Regression, Random Forest Classifier, SVC, Decision Tree Classifier, K-Neighbors Classifier, Naive Bayes Classifier) algorithms.
* **Hyperparameter Tuning:**  Allows basic hyperparameter tuning for Random Forest, SVC, SVR, and K-Nearest Neighbors models.
* **Model Training and Evaluation:** Trains the selected model and evaluates its performance using appropriate metrics (MSE, MAE, R² for regression; Accuracy, Precision, Recall, F1-score for classification).
* **Model Download:** Enables downloading the trained model in a pickle file (`.pkl`).
* **Prediction Interface:** Provides a user-friendly interface to make predictions on new data.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/prantikm07/AutoML-Pro.git
   cd AutoML-Pro
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv  # or python -m venv venv on Windows
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Create `requirements.txt` containing the following)*
   ```
   streamlit
   pandas
   matplotlib
   seaborn
   scikit-learn
   joblib
   ```

## Usage

1. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Open the app in your web browser:** Streamlit will provide a URL (usually `http://localhost:8501`) that you can use to access the application.

## How to Use

1. **Upload Data:** Upload your data file (CSV or Excel).
2. **Explore Data:** Use the "Data Exploration" section to understand your data.
3. **Clean Data:** Use the "Data Cleaning" section to handle missing values, duplicates, and unwanted columns.
4. **Visualize Data:** Use the "Data Visualization" section to create various charts.
5. **Feature Engineering:** Use the "Feature Engineering" section to encode categorical variables.
6. **Model Configuration:** Select your target variable, features, and problem type (regression or classification). Choose a model and optionally tune hyperparameters.
7. **Train Model:** Click the "Train Model" button.
8. **Evaluate Model:** View the evaluation metrics.
9. **Download Model:** Download the trained model.
10. **Make Predictions:** Use the "Make Predictions" section to input new data and get predictions.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

If you have any questions, feel free to contact me via:
- Email: [prantik25m@gmail.com](mailto:prantik25m@gmail.com)
- LinkedIn: [Prantik Mukhopadhyay](https://www.linkedin.com/in/prantikm07/)

## Acknowledgements

* Streamlit for the awesome framework.
* scikit-learn for the machine learning library.
* pandas, matplotlib, and seaborn for data analysis and visualization.
