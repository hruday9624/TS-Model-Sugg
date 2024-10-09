import streamlit as st
import pandas as pd

def main():
    st.title("Time Series Forecasting Method Suggestion App")

    st.write("""
    Upload your time series data in CSV or Excel format. The file should contain at least two columns:
    - **Date**: The date or datetime information.
    - **Value**: The numerical values of your time series.
    """)

    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"], encoding='utf-8')

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format.")
                return

            st.write("First few rows of your data:")
            st.dataframe(df.head())

        except UnicodeEncodeError:
            st.error("There was an encoding error while reading the file. Please ensure the file is in UTF-8 encoding.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.info("Awaiting CSV or Excel file upload.")

if __name__ == "__main__":
    main()
