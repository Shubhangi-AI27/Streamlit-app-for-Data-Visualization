import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import google.generativeai as genai
from dotenv import load_dotenv
import io
import contextlib
import re
import numpy as np

# Load API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

def sanitize_code(code):
    code = re.sub(r"```(?:python)?\n", "", code)
    code = re.sub(r"```", "", code)
    return code.strip()

def call_gemini_api(prompt, df_columns):
    column_list = ", ".join(df_columns)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        f"""You are a helpful assistant that returns only Python matplotlib/seaborn code.
The DataFrame is named `df`. Its columns are: {column_list}.
Use aggregation, binning, kde, sampling if needed. No imports, no comments, no plt.show().
Prompt: {prompt}"""
    )
    return sanitize_code(response.text)

def main():
    st.title("Gemini Data Visualizer")
    st.write("Upload a CSV and describe your plot!")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("Data Preview")
            st.dataframe(df.head())

            # === Data Cleaning ===
            st.sidebar.subheader("Data Cleaning")
            if st.sidebar.checkbox("Handle Missing Values"):
                strategy = st.sidebar.selectbox("Strategy", ["drop", "mean", "median", "ffill"])
                if st.sidebar.button("Apply Missing Handling"):
                    if strategy == "drop":
                        df = df.dropna()
                    else:
                        for col in df.select_dtypes(include=np.number).columns:
                            if strategy == "mean":
                                df[col] = df[col].fillna(df[col].mean())
                            elif strategy == "median":
                                df[col] = df[col].fillna(df[col].median())
                            elif strategy == "ffill":
                                df[col] = df[col].fillna(method='ffill')
                    st.success("Missing values handled")
                    st.dataframe(df.head())

            if st.sidebar.checkbox("Detect Outliers"):
                outlier_cols = st.sidebar.multiselect(
                    "Columns to check for outliers",
                    df.select_dtypes(include=np.number).columns.tolist()
                )
                if st.sidebar.button("Detect Outliers Now"):
                    for col in outlier_cols:
                        q1 = df[col].quantile(0.25)
                        q3 = df[col].quantile(0.75)
                        iqr = q3 - q1
                        outliers = df[(df[col] < (q1 - 1.5*iqr)) | (df[col] > (q3 + 1.5*iqr))]
                        st.write(f"Outliers in {col}: {len(outliers)} rows")

            # === Plot Generation ===
            df_columns = df.columns.tolist()
            prompt = st.text_input("Describe your plot")

            if st.button("Generate Plot") and prompt.strip():
                code_output = call_gemini_api(prompt, df_columns)

                try:
                    if any(k in code_output for k in ["sns.", "plt.", "df"]):
                        plt.figure(figsize=(10, 6))
                        with contextlib.redirect_stdout(io.StringIO()):
                            exec(code_output, {"df": df, "plt": plt, "sns": sns, "np": np})

                        buf = io.BytesIO()
                        plt.savefig(buf, format="png", bbox_inches='tight')
                        buf.seek(0)
                        st.session_state['plot_buf'] = buf
                        st.session_state['code_output'] = code_output
                        plt.clf()
                    else:
                        st.warning("Gemini did not return usable plot code.")
                except Exception as e:
                    st.warning(f"Error executing plot: {e}")

            # === Plot Display & Download ===
            if 'plot_buf' in st.session_state:
                st.subheader("Your Plot")
                st.image(st.session_state['plot_buf'].getvalue())

                export_format = st.selectbox("Choose Format", ["PNG", "PDF", "SVG", "JPEG"])
                export_buf = io.BytesIO()

                plt.figure(figsize=(10, 6))
                exec(st.session_state['code_output'], {"df": df, "plt": plt, "sns": sns, "np": np})
                plt.savefig(export_buf, format=export_format.lower(), bbox_inches='tight')
                export_buf.seek(0)
                plt.clf()

                st.download_button(
                    label=f"Download {export_format}",
                    data=export_buf,
                    file_name=f"plot.{export_format.lower()}",
                    mime=f"image/{'jpeg' if export_format == 'JPEG' else export_format.lower()}"
                )

        except Exception as e:
            st.error(f"Could not read CSV: {e}")

if __name__ == "__main__":
    main()

