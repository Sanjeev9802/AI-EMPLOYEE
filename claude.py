import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import io
import plotly.express as px
import plotly.graph_objects as go
import base64
import os
from dotenv import load_dotenv


# ... (previous code remains the same)

def decision_tree_classification(df, target_column):
    try:
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # Check if the target variable is categorical
        if not pd.api.types.is_categorical_dtype(y) and not pd.api.types.is_object_dtype(y):
            raise ValueError("Target variable must be categorical for classification.")

        # Ensure the target variable is encoded as integers
        le = LabelEncoder()
        y = le.fit_transform(y)

        model = DecisionTreeClassifier(random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)
        accuracy = accuracy_score(y, predictions)
        return model, accuracy
    except Exception as e:
        st.error(f"Error in decision tree classification: {str(e)}")
        return None, None


# ... (rest of the code remains the same)

def main():
    # ... (previous code remains the same)

    elif analysis_type == "Decision Tree":
    target_column = st.selectbox("Select Target Column", cleaned_data.columns)
    if st.button("Run Decision Tree Classification"):
        model, accuracy = decision_tree_classification(cleaned_data, target_column)
        if model is not None and accuracy is not None:
            st.write("Decision Tree Model Created")
            st.write("Model Accuracy:", accuracy)
            report_text = f"Decision Tree classification model created with Accuracy: {accuracy:.2f}"

            fig, ax = plt.subplots(figsize=(20, 10))
            from sklearn import tree
            tree.plot_tree(model, filled=True, ax=ax)
            st.pyplot(fig)
            charts.append(fig.to_html())
        else:
            st.error("Failed to create Decision Tree model. Please check your data and try again.")


# ... (rest of the code remains the same)

if __name__ == '__main__':
    main()