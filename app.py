# lab_eda_gui.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Page Configuration

st.set_page_config(
    page_title="EDA Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Exploratory Data Analysis Interface")


# 2. Sidebar: Dataset Ingestion

# set header for sidebar
st.sidebar.header("Dataset Controls")

# create a file uploader in the sidebar for CSV files
uploaded_file = st.sidebar.file_uploader("Titanic-Dataset.csv", type=["csv"])

if uploaded_file is not None:
    # Read dataset
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the uploaded file as a valid CSV. Error: {e}")
        st.stop()

    if df.empty:
        st.warning("The uploaded CSV file is empty. Please upload a file with data.")
        st.stop()

    # 3. Dataset Overview

    # set subheader for dataset overview
    st.subheader("Dataset Preview and Metadata")

    st.write("**First 5 Rows:**")
    # display the first 5 rows of the dataset
    st.dataframe(df.head())

    # display the shape of the dataset
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

    st.write("**Column Data Types:**")
    # display the data types of each column in the dataset
    dtypes_df = df.dtypes.astype(str).reset_index()
    dtypes_df.columns = ["Column", "Data Type"]
    st.dataframe(dtypes_df, use_container_width=True)

    # Missing value summary
    st.write("**Missing Values per Column:**")
    # display the count and percentage of missing values for each column in the dataset
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": missing_count.values,
        "Missing %": missing_pct.values
    })
    st.dataframe(missing_df, use_container_width=True)

    # Basic statistics for numerical columns
    st.write("**Basic Numerical Statistics:**")
    # display the basic statistics
    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        st.dataframe(numeric_df.describe().T, use_container_width=True)
    else:
        st.info("No numerical columns found in this dataset.")

    # 4. Attribute Selection

    # set header for attribute selection in the sidebar
    st.sidebar.header("Attribute Selection")

    # create a selectbox in the sidebar to choose an attribute for visualization
    selected_column = st.sidebar.selectbox("Choose an attribute to visualize", df.columns)

    # Detect column type
    if pd.api.types.is_numeric_dtype(df[selected_column]):
        column_type = "Numerical"
    else:
        column_type = "Categorical"

    st.sidebar.write(f"Detected type: **{column_type}**")

    # 5. Visualization Rendering

    st.subheader("Visualization")

    if column_type == "Numerical":
        # Histogram with seaborn
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df[selected_column].dropna(), kde=True, ax=ax, color="steelblue")
        ax.set_title(f"Distribution of {selected_column}")
        ax.set_xlabel(selected_column)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
    else:
        # Bar chart for categorical
        value_counts = df[selected_column].value_counts(dropna=False)
        value_pct = (value_counts / value_counts.sum() * 100).round(2)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=value_counts.index.astype(str), y=value_counts.values, ax=ax, color="steelblue")
        ax.set_title(f"Frequency Counts of {selected_column}")
        ax.set_xlabel(selected_column)
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")

        # annotate bars with percentage
        for i, (count, pct) in enumerate(zip(value_counts.values, value_pct.values)):
            ax.text(i, count, f"{pct}%", ha="center", va="bottom", fontsize=8)

        st.pyplot(fig)

else:
    st.info("Please upload a CSV file to start EDA.")