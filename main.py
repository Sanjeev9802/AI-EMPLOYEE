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


def load_data(file):
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith('.json'):
            return pd.read_json(file)
        elif file.name.endswith('.xlsx'):
            return pd.read_excel(file)
        else:
            st.error("Unsupported file format")
            return None
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


def clean_data(df):
    df = df.dropna()
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[num_cols] = scaler.fit_transform(df[num_cols])
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
    return df


def linear_regression(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    return model.coef_, model.intercept_, mse


def k_means_clustering(df, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(df.drop('cluster', axis=1, errors='ignore'))
    return df, kmeans


def decision_tree_classification(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    return model, accuracy


def generate_report(data, report_text, charts):
    report = io.StringIO()
    report.write("AI Employee Analysis Report\n\n")
    report.write(report_text + "\n\n")
    for i, chart in enumerate(charts):
        report.write(f"Chart {i + 1}:\n")
        report.write(chart + "\n\n")
    return report.getvalue()


def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="cleaned_data.csv">Download Cleaned Data as CSV</a>'
    return href


def main():
    st.title("AI Employee Prototype for Data Analysis and Reporting")

    uploaded_file = st.file_uploader("Upload your data file", type=["csv", "json", "xlsx"])

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.write("Data Preview:")
            st.write(data.head())

            if st.button("Clean Data"):
                cleaned_data = clean_data(data)
                st.session_state['cleaned_data'] = cleaned_data
                st.write("Cleaned Data:")
                st.write(cleaned_data.head())
                st.markdown(get_table_download_link(cleaned_data), unsafe_allow_html=True)

    if 'cleaned_data' in st.session_state:
        cleaned_data = st.session_state['cleaned_data']
        analysis_type = st.selectbox("Choose Analysis Type",
                                     ["Linear Regression", "K-Means Clustering", "Decision Tree"])

        report_text = ""
        charts = []

        if analysis_type == "Linear Regression":
            target_column = st.selectbox("Select Target Column", cleaned_data.columns)
            if st.button("Run Linear Regression"):
                coef, intercept, mse = linear_regression(cleaned_data, target_column)
                st.write("Linear Regression Coefficients:", coef)
                st.write("Intercept:", intercept)
                st.write("Mean Squared Error:", mse)
                report_text = f"Linear Regression Coefficients: {coef}, Intercept: {intercept}, MSE: {mse}"

                fig = px.scatter(cleaned_data, x=cleaned_data.columns[0], y=target_column,
                                 title='Linear Regression Plot')
                fig.add_trace(go.Scatter(x=cleaned_data[cleaned_data.columns[0]],
                                         y=coef[0] * cleaned_data[cleaned_data.columns[0]] + intercept, mode='lines',
                                         name='Fitted Line'))
                st.plotly_chart(fig)
                charts.append(fig.to_html())

        elif analysis_type == "K-Means Clustering":
            n_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)
            if st.button("Run K-Means Clustering"):
                clustered_data, kmeans = k_means_clustering(cleaned_data, n_clusters)
                st.write("Clustered Data:")
                st.write(clustered_data.head())
                report_text = f"K-Means Clustering performed with {n_clusters} clusters."

                fig = px.scatter_matrix(clustered_data, dimensions=clustered_data.columns[:-1], color='cluster')
                st.plotly_chart(fig)
                charts.append(fig.to_html())

        elif analysis_type == "Decision Tree":
            target_column = st.selectbox("Select Target Column", cleaned_data.columns)
            if st.button("Run Decision Tree Classification"):
                model, accuracy = decision_tree_classification(cleaned_data, target_column)
                st.write("Decision Tree Model Created")
                st.write("Model Accuracy:", accuracy)
                report_text = f"Decision Tree classification model created with Accuracy: {accuracy:.2f}"

                fig, ax = plt.subplots(figsize=(20, 10))
                from sklearn import tree
                tree.plot_tree(model, filled=True, ax=ax)
                st.pyplot(fig)
                charts.append(fig.to_html())

        if report_text:
            if st.button("Generate Report"):
                report = generate_report(cleaned_data, report_text, charts)
                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name="ai_employee_analysis_report.txt",
                    mime="text/plain"
                )


if __name__ == '__main__':
    main()