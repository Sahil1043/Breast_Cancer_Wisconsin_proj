import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import io

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv('data.csv')
    df = df.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore')
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df

# Train the model
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X_train_scaled, X_test_scaled, y_train, y_test

# Load data and train model
df = load_data()
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']
model, scaler, X_train_scaled, X_test_scaled, y_train, y_test = train_model(X, y)

# Sidebar options
st.sidebar.title("Breast Cancer Detection")
option = st.sidebar.radio("Choose an option:", ['Dataset Overview', 'Input Data for Prediction'])

# Overview toggle options
if option == 'Dataset Overview':
    st.sidebar.write("Overview Options:")
    show_first_last = st.sidebar.checkbox("Show First/Last Rows", value=True)
    show_shape = st.sidebar.checkbox("Show DataFrame Shape", value=True)
    show_info = st.sidebar.checkbox("Show DataFrame Info", value=True)
    show_describe = st.sidebar.checkbox("Show Descriptive Statistics", value=True)
    show_value_counts = st.sidebar.checkbox("Show Label Value Counts", value=True)
    show_model_metrics = st.sidebar.checkbox("Show Model Metrics", value=True)

# Option 1: Show overview of the dataset
if option == 'Dataset Overview':
    st.header("Breast Cancer Wisconsin (Diagnostic)")

    if show_first_last:
        # First 5 rows
        st.subheader("First 5 Rows of the Dataset")
        st.write(df.head())

        # Last 5 rows
        st.subheader("Last 5 Rows of the Dataset")
        st.write(df.tail())

    if show_shape:
        # Dataframe size
        st.subheader("Dataframe Shape")
        st.write(f"Number of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")

    if show_info:
        # Dataframe info
        st.subheader("Dataframe Information")
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

    if show_describe:
        # Descriptive statistics
        st.subheader("Descriptive Statistics")
        st.write(df.describe())

    if show_value_counts:
        # Label value counts
        st.subheader("Label Value Count (Malignant/Benign)")
        st.write(df['diagnosis'].value_counts())

    if show_model_metrics:
        # Model metrics
        st.subheader("Model Metrics")

        # Value counts for training and testing data
        st.write("Training Data Distribution:")
        st.write(pd.Series(y_train).value_counts())
        st.write("Testing Data Distribution:")
        st.write(pd.Series(y_test).value_counts())

        # Calculate metrics
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        train_precision = precision_score(y_train, y_train_pred)
        test_precision = precision_score(y_test, y_test_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)

        # Create a DataFrame for metrics
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'F1-Score'],
            'Training': [train_accuracy, train_precision, train_f1],
            'Testing': [test_accuracy, test_precision, test_f1]
        })
        metrics_df = metrics_df.set_index('Metric')
        metrics_df = metrics_df.applymap(lambda x: f"{x:.4f}")

        st.write(metrics_df)

        # Full classification report
        st.subheader("Classification Report")
        report = classification_report(y_test, y_test_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.write(report_df)

# Option 2: Input values and check prediction
elif option == 'Input Data for Prediction':
    st.header("Enter the Data to Predict")

    # Input fields for each feature
    input_data = {}
    for column in X.columns:
        input_data[column] = st.number_input(f'{column} value', value=X[column].mean())

    if st.button("Predict"):
        # Convert inputs to a numpy array for prediction
        input_array = np.array(list(input_data.values())).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        
        # Make a prediction
        prediction = model.predict(input_scaled)

        # Display the result
        if prediction[0] == 1:
            st.write("The tumor is **malignant**.")
        else:
            st.write("The tumor is **benign**.")