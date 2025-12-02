import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Churn EDA", layout="wide")

st.title("Customer Churn Exploratory Data Analysis")
st.write("Interactive EDA dashboard for the Telco Customer Churn dataset.")

# Upload or load dataset
st.info("Using default dataset from data/raw/ folder")
df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

st.subheader("Preview of Dataset")
st.dataframe(df.head())

# Display shape
st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# Missing values
st.subheader("Missing Values")
st.dataframe(df.isnull().sum())

# Summary statistics
st.subheader("Summary Statistics")
st.write(df.describe(include="all"))

# Plot: Churn Count
st.subheader("Churn Distribution")
fig, ax = plt.subplots()
sns.countplot(x="Churn", data=df, ax=ax)
st.pyplot(fig)

# Numerical correlation heatmap
st.subheader("Correlation Heatmap (Numerical Features)")
numeric_df = df.select_dtypes(include=['float64', 'int64'])
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Select column for distribution
st.subheader("Distribution of Any Column")
column = st.selectbox("Choose a column", df.columns)
fig, ax = plt.subplots()
sns.histplot(df[column].dropna(), kde=True, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

st.success("EDA Completed")