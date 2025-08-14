import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import openai
import os

# Load OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to call GPT for code generation
def call_openai_api(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that returns python matplotlib/seaborn code."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return response['choices'][0]['message']['content']

# Manual plot builder
def manual_plot_builder(df):
    st.subheader("Smart Manual Plot Builder")

    chart_types = [
        "scatter", "line", "bar", "hist", "box", "violin", "kde",
        "area", "count", "heatmap", "pairplot"
    ]

    chart_type = st.selectbox("Choose Chart Type", chart_types, key="chart_type")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    all_cols = df.columns.tolist()

    x_axis = y_axis = group_by = None

    if chart_type in ["scatter", "line", "box", "violin", "area"]:
        x_axis = st.selectbox("Select X-axis", all_cols, key="x_axis")
        y_axis = st.selectbox("Select Y-axis (numeric)", numeric_cols, key="y_axis")
    elif chart_type == "bar":
        x_axis = st.selectbox("Select X-axis (categorical)", categorical_cols, key="x_axis_bar")
        y_axis = st.selectbox("Select Y-axis (numeric)", numeric_cols, key="y_axis_bar")
    elif chart_type == "hist":
        x_axis = st.selectbox("Select Numeric Column", numeric_cols, key="x_hist")
    elif chart_type == "kde":
        x_axis = st.selectbox("Select Numeric Column", numeric_cols, key="x_kde")
    elif chart_type == "count":
        x_axis = st.selectbox("Select Categorical Column", categorical_cols, key="x_count")
    elif chart_type == "heatmap":
        st.info("Correlation matrix will be generated using numeric columns.")
    elif chart_type == "pairplot":
        st.info("Pairplot will show relationships between numeric features.")

    group_by = st.selectbox("Optional - Group by", [None] + all_cols, key="group_by")
    show_legend = st.checkbox("Show legend", value=True)

    if st.button("Generate Plot", key="plot_button"):
        plt.figure(figsize=(10, 6))
        try:
            if chart_type in ["scatter", "line", "box", "violin", "area", "bar"]:
                if df[x_axis].dropna().empty or df[y_axis].dropna().empty:
                    st.error("One of the selected columns has no valid data.")
                    return
            elif chart_type in ["hist", "kde", "count"]:
                if df[x_axis].dropna().empty:
                    st.error("Selected column has no valid data.")
                    return

            # Plotting logic
            if chart_type == "scatter":
                sns.scatterplot(data=df, x=x_axis, y=y_axis, hue=group_by)
            elif chart_type == "line":
                sns.lineplot(data=df, x=x_axis, y=y_axis, hue=group_by)
            elif chart_type == "bar":
                sns.barplot(data=df, x=x_axis, y=y_axis, hue=group_by)
            elif chart_type == "hist":
                sns.histplot(data=df, x=x_axis, bins=30, hue=group_by)
            elif chart_type == "box":
                sns.boxplot(data=df, x=x_axis, y=y_axis, hue=group_by)
            elif chart_type == "violin":
                sns.violinplot(data=df, x=x_axis, y=y_axis, hue=group_by)
            elif chart_type == "kde":
                sns.kdeplot(data=df[x_axis].dropna(), fill=True, label=x_axis)
            elif chart_type == "area":
                clean_df = df[[x_axis, y_axis]].dropna()
                clean_df.set_index(x_axis)[y_axis].plot(kind='area', label=y_axis)
            elif chart_type == "count":
                sns.countplot(data=df, x=x_axis, hue=group_by)
            elif chart_type == "heatmap":
                corr = df.select_dtypes(include="number").corr()
                sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
            elif chart_type == "pairplot":
                fig = sns.pairplot(df.select_dtypes(include="number"), hue=group_by)
                st.pyplot(fig)
                return

            # Show/hide legend
            if show_legend:
                plt.legend(loc="best")
            else:
                plt.legend([], [], frameon=False)

            st.pyplot(plt.gcf())
            plt.clf()

        except Exception as e:
            st.error(f"Error generating plot: {e}")

# Main app
def main():
    st.title("ChatGPT-Powered Data Visualizer")
    st.write("Upload a CSV file and describe the chart you want in plain English!")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of Uploaded Data")
            st.dataframe(df.head())

            prompt = st.text_input("Describe your plot (e.g. 'scatter plot of age vs income colored by gender')")

            if st.button("Generate & Run Visualization"):
                if prompt.strip():
                    try:
                        code_output = call_openai_api(prompt)
                        st.code(code_output, language="python")
                        st.info("GPT returned Python code. Copy it to your notebook to run.")
                        st.warning("This app does not auto-run code for safety. Use the manual plot builder below.")
                    except Exception as api_error:
                        st.warning(f"OpenAI API error: {api_error}")
                        st.info("Switching to manual fallback plotting mode.")
                        manual_plot_builder(df)
                else:
                    st.warning("Please enter a plot description or use the manual builder below.")
                    manual_plot_builder(df)
            else:
                st.info("Use the manual plot builder below.")
                manual_plot_builder(df)

        except Exception as e:
            st.error(f"Error reading CSV file: {e}")

if __name__ == "__main__":
    main()

