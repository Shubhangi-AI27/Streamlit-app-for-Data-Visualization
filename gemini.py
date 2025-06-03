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
import numpy as np # Added for potential use in generated code, though not explicitly imported in prompt

# Load API key from environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

def sanitize_code(code):
    """
    Removes markdown code fences and strips whitespace from generated code.
    """
    code = re.sub(r"```(?:python)?\n", "", code)
    code = re.sub(r"```", "", code)
    code = code.strip()
    return code

def call_gemini_api(prompt, df_columns):
    """
    Calls the Gemini API to generate Python plotting code based on the user's prompt.
    The prompt is enhanced with instructions for handling large datasets.
    """
    # Pass column names to Gemini so it knows what's available in the DataFrame
    column_list = ", ".join(df_columns)

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        f"""You are a helpful assistant that returns only Python matplotlib/seaborn code for data visualization.
The DataFrame is named `df`. Its columns are: {column_list}.
When dealing with potentially large datasets or continuous variables, prioritize clear and readable visualizations.
Consider using the following techniques to prevent clutter and improve readability:
-   **Aggregation:** For time-series or categorical data, use `df.groupby().mean()`, `df.groupby().sum()`, `df.value_counts()`, etc., to summarize data before plotting.
-   **Binning:** For continuous numerical data, use `pd.cut()` or `np.histogram` to create bins and then plot counts or aggregated values per bin.
-   **Density Plots:** For scatter plots with high overplotting, consider `sns.kdeplot` (for 1D or 2D density) or `plt.hexbin` (for 2D density with hexagonal bins).
-   **Transparency (Alpha Blending):** For scatter plots, set `alpha` parameter (e.g., `alpha=0.1`) to show data density where points overlap.
-   **Sampling:** For extremely large datasets where even density plots are slow, consider `df.sample()` to plot a representative subset. Only suggest this if explicitly needed or if other methods fail.

Ensure the generated code is self-contained and directly produces a plot.
Do not include explanations or comments.
Do not include any imports (assume `pandas`, `matplotlib.pyplot`, `seaborn`, and `numpy` are already imported as `pd`, `plt`, `sns`, `np`).
Do not include `plt.show()`.
Only return valid Python code.

Prompt: {prompt}
"""
    )
    return sanitize_code(response.text)

def main():
    """
    Main Streamlit application function for data visualization.
    """
    st.title("🤖📊 Gemini Powered- Data Visualizer")
    st.write("Upload a CSV file and describe your plot in plain English!")

    uploaded_file = st.file_uploader("📁 Upload CSV File", type=["csv"])
    if uploaded_file:
        try:
            # Read the uploaded CSV file into a Pandas DataFrame
            df = pd.read_csv(uploaded_file)
            st.subheader("🔍 Data Preview")
            st.dataframe(df.head())

            # Get column names to pass to the API for better context
            df_columns = df.columns.tolist()

            # Text input for user to describe the desired plot
            prompt = st.text_input(
                "🧠 Describe your plot (e.g., 'scatter plot of age vs income by gender', "
                "'monthly average sales', 'density of temperature and pressure', "
                "'histogram of customer age')"
            )

            # Button to trigger plot generation
            if st.button("Generate Plot") and prompt.strip():
                # Call Gemini API with the user's prompt and DataFrame columns
                code_output = call_gemini_api(prompt, df_columns)

                st.subheader("Generated Code (for debugging):")
                st.code(code_output, language='python') # Display the generated code for transparency and debugging

                try:
                    # Check if the generated code contains keywords indicating it's a plotting command
                    if any(keyword in code_output for keyword in ["sns.", "plt.", "df"]):
                        # Create a new Matplotlib figure before executing the generated code.
                        # This prevents plots from stacking and ensures a clean canvas for each generation.
                        plt.figure(figsize=(10, 6)) # Default figure size, can be adjusted by Gemini if prompted

                        # Redirect stdout to capture any print statements from exec (though Gemini is instructed not to print)
                        with contextlib.redirect_stdout(io.StringIO()):
                            # Execute the generated Python code.
                            # The `globals()` dictionary provides the context (df, plt, sns) for the executed code.
                            exec(code_output, {"df": df, "plt": plt, "sns": sns, "np": np})

                        # Display the generated plot in Streamlit
                        st.pyplot(plt.gcf())
                        plt.clf() # Clear the current figure to prevent it from being reused or showing old content
                    else:
                        st.warning("⚠ Gemini did not return usable plotting code. Please refine your prompt or try a different description.")
                except SyntaxError as e:
                    # Handle syntax errors in the generated code
                    st.error(f"⚠ Syntax error in generated code. Please check your prompt or the generated code structure: {e}")
                    st.text("Generated code was:\n" + code_output)
                except Exception as e:
                    # Handle other runtime errors during code execution
                    st.warning(f"⚠ Error executing Gemini code. This might be due to incorrect column names, data types, or a plotting issue: {e}")
                    st.text("Generated code was:\n" + code_output)

            else:
                st.info("Please enter a plot description and click 'Generate Plot'.")

        except Exception as e:
            # Handle errors during CSV file reading
            st.error(f"❌ Could not read CSV file or an unexpected error occurred: {e}")

if __name__ == "__main__":
    main()


