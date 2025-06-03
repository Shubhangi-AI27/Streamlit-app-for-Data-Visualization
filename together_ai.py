
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from together import Together
import re

# Load API Key
api_key = "tgp_v1_DAOBax7cmlPZpTutDGPvbU8qLNxNioI8d6rygKAGESk"
client = Together(api_key=api_key)

def extract_code_from_response(response_text):
    """Extract only the Python code block from Together AI's response."""
    code_blocks = re.findall(r"```(?:python)?\n(.*?)```", response_text, re.DOTALL)
    return code_blocks[0] if code_blocks else response_text.strip()

def get_code_from_together(prompt):
    """Call Together AI to get Python code for a matplotlib/seaborn plot."""
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": "You are a data visualization assistant using matplotlib and seaborn."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=500
    )
    return response.choices[0].message.content

def main():
    st.title("📊 Together AI Data Visualizer")

    f = st.file_uploader("Upload your CSV file", type=["csv"])
    if not f:
        st.info("Please upload a CSV file to begin.")
        return

    try:
        df = pd.read_csv(f)

        # Show data preview
        st.subheader("📑 Data Preview")
        st.dataframe(df.head())

        # Prompt for visualization
        user_prompt = st.text_input("🔍 Describe the plot you want (or leave blank for default):")

        if st.button("Generate Plot"):
            if not user_prompt.strip():
                cols = df.columns.tolist()
                if len(cols) >= 2:
                    user_prompt = f"Create a seaborn scatter plot using 'df' with x='{cols[0]}' and y='{cols[1]}'"
                else:
                    user_prompt = f"Create a histogram of the column '{cols[0]}' in DataFrame 'df'"

            try:
                raw_code = get_code_from_together(user_prompt)
                code = extract_code_from_response(raw_code)

                exec_globals = {"df": df, "sns": sns, "plt": plt, "pd": pd}
                exec(code, exec_globals)

                st.subheader("📈 Plot")
                st.pyplot(plt.gcf())
                plt.clf()

            except Exception as e:
                st.error(f"❌ Error generating plot: {e}")

    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")

if __name__ == "__main__":
    main()
