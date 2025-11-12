

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------- Page Setup --------------------
st.set_page_config(page_title="DDoS Attack Classifier", layout="wide")

# -------------------- Sidebar Navigation --------------------
st.sidebar.title("🔍 Navigation")
menu = st.sidebar.radio("Go to", [" Home", "DDoS Classification", "Model Insights"])

# -------------------- Home Page --------------------
if menu == "Home":
    st.title("DDoS Attack Classification Web App")
    st.write("""
    This application classifies network traffic as **Benign** or a type of **DDoS attack**
    using a Random Forest machine learning model.
    ---
    **Features:**
    - Upload CSV datasets  
    - Visualize class distributions  
    - Train and test Random Forest models  
    - View model accuracy and metrics  
    - Predict attack types interactively  
    """)
    st.image("https://miro.medium.com/v2/resize:fit:1200/1*2QZ8kJDG7Yvfs7bQPL4OBQ.png", caption="DDoS Attack Concept", use_container_width=True)
    st.markdown("### Navigate to the 'DDoS Classification' tab to start →")

# -------------------- DDoS Classification --------------------
elif menu == "DDoS Classification":
    st.title("DDoS Attack Classification")

    # Upload CSV
    uploaded_file = st.file_uploader("C:\Users\Hansika\OneDrive\Documents\rinnyy\Projects\ddos-streamlit-app\sample_ddos_dataset.csv", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Data Preview
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        # Handle missing values
        df = df.dropna()

        # Class Distribution
        st.subheader("Class Distribution")
        st.bar_chart(df.iloc[:, -1].value_counts())

        # Select features and labels
        X = df.iloc[:, :6]
        y = df.iloc[:, -1]

        # Impute missing values if any
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Display metrics
        st.subheader("Model Performance")
        st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")
        st.text("Classification Report:")
        st.dataframe(pd.DataFrame(report).transpose())

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # -------------------- Prediction Section --------------------
        st.subheader("Make a New Prediction")

        input_data = []
        st.write("Enter the following feature values:")
        for i, col in enumerate(df.columns[:6]):
            val = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
            input_data.append(val)

        if st.button("🚀 Predict DDoS Attack Type"):
            prediction = model.predict([input_data])
            st.success(f"**Predicted Attack Type:** {prediction[0]}")

    else:
        st.info("Please upload a CSV file to begin classification.")

# -------------------- Model Insights --------------------
elif menu == "Model Insights":
    st.title("Feature Importance Visualization")

    st.write("""
    This section shows which features contribute the most to the model’s decisions.
    Upload the same dataset used for training to visualize the feature importance graph.
    """)

    uploaded_file = st.file_uploader("Upload Dataset for Feature Importance", type=["csv"], key="insights")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = df.dropna()

        X = df.iloc[:, :6]
        y = df.iloc[:, -1]

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        importances = model.feature_importances_
        feature_names = df.columns[:6]

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=importances, y=feature_names, ax=ax)
        ax.set_title("Feature Importance in DDoS Classification")
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Feature")
        st.pyplot(fig)
    else:
        st.info("Upload your dataset to view feature importance.")

# -------------------- Footer --------------------
st.sidebar.markdown("---")
st.sidebar.info("Developed with using Streamlit | © 2025 DDoS Classifier App")
