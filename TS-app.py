import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

def suggest_forecasting_methods(time_series, freq):
    if not isinstance(time_series.index, pd.DatetimeIndex):
        st.error("The time series index must be a pandas DatetimeIndex.")
        return

    if time_series.empty:
        st.error("The time series is empty after processing. Please check your data.")
        return

    st.subheader("Time Series Data Plot")
    st.line_chart(time_series)

    try:
        adf_result = adfuller(time_series.dropna())
        p_value = adf_result[1]
        st.write(f"**ADF Statistic**: {adf_result[0]:.4f}")
        st.write(f"**p-value**: {p_value:.4f}")

        stationary = p_value < 0.05
        if stationary:
            st.success("The data is stationary.")
        else:
            st.warning("The data is non-stationary.")
    except Exception as e:
        st.error(f"ADF test was not successful: {e}")
        return

    try:
        if len(time_series.dropna()) < 2 * freq:
            raise ValueError("The length of the time series is less than twice the specified frequency. Please provide more data or reduce the frequency.")
        decomposition = seasonal_decompose(time_series.dropna(), model='additive', period=freq)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid

        st.subheader("Decomposition of Time Series")
        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(time_series, label='Original')
        axes[0].legend(loc='upper left')
        axes[1].plot(trend, label='Trend', color='orange')
        axes[1].legend(loc='upper left')
        axes[2].plot(seasonal, label='Seasonality', color='green')
        axes[2].legend(loc='upper left')
        axes[3].plot(residual, label='Residuals', color='red')
        axes[3].legend(loc='upper left')
        st.pyplot(fig)
    except ValueError as e:
        st.error(f"Seasonal decomposition was not successful: {e}")
        trend = seasonal = residual = None

    has_trend = trend is not None and trend.dropna().std() > 0
    has_seasonality = seasonal is not None and seasonal.dropna().std() > 0

    if has_trend:
        st.info("Trend detected in the data.")
    else:
        st.info("No significant trend detected.")

    if has_seasonality:
        st.info("Seasonality detected in the data.")
    else:
        st.info("No significant seasonality detected.")

    st.subheader("Recommendations:")
    if not stationary:
        st.write("- Consider differencing or transforming the data to achieve stationarity.")

    if has_trend and not has_seasonality:
        st.write("- Recommended methods: **Double Exponential Smoothing**, **ARIMA** with trend.")
    elif has_trend and has_seasonality:
        st.write("- Recommended methods: **Holt-Winters Exponential Smoothing**, **SARIMA**.")
    elif not has_trend and has_seasonality:
        st.write("- Recommended methods: **Seasonal Decomposition**, **SARIMA**.")
    else:
        st.write("- Recommended methods: **Simple Exponential Smoothing**, **ARIMA**.")

    st.write("- For complex patterns, consider machine learning models like **LSTM networks**.")

def main():
    st.title("📈 Time Series Forecasting Method Suggestion App")

    st.write("""
    Upload your time series data in CSV or Excel format. The file should contain at least two columns:
    - **Date**: The date or datetime information.
    - **Value**: The numerical values of your time series.
    """)

    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format.")
                return

            columns = df.columns.tolist()
            date_col = st.selectbox("Select the date column:", options=columns, key='date_col')
            numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
            if len(numerical_columns) > 1:
                value_col = st.selectbox("Select the target column:", options=numerical_columns, key='value_col')
            else:
                value_col = numerical_columns[0] if numerical_columns else st.selectbox("Select the value column:", options=columns, index=1 if len(columns) > 1 else 0, key='value_col')

            st.write("First few rows of your data:")
            st.dataframe(df.head())

            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col])
            df.set_index(date_col, inplace=True)
            df.sort_index(inplace=True)
            df[value_col].interpolate(method='time', inplace=True)

            if df[value_col].isnull().all():
                st.error("The value column contains only NaN values after interpolation. Please check your data.")
                return

            freq_input = st.text_input("Specify the frequency of your data (e.g., 'D' for daily, 'M' for monthly):", value='M')
            try:
                df = df.asfreq(freq_input)
                freq = pd.infer_freq(df.index)
                if freq is None:
                    st.warning("Frequency could not be inferred. Please provide the period for seasonality.")
                    freq = st.number_input("Specify the period of seasonality (e.g., 12 for yearly seasonality in monthly data):", min_value=1, value=12)
                else:
                    st.write(f"Inferred frequency: {freq}")
                    freq = pd.Timedelta(freq).days if 'D' in freq else int(freq.strip('W')) * 7 if 'W' in freq else 1
            except ValueError as e:
                st.error(f"Error in setting frequency: {e}")
                freq = st.number_input("Specify the period of seasonality (e.g., 12 for yearly seasonality in monthly data):", min_value=1, value=12)

            suggest_forecasting_methods(df[value_col], freq)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.info("Awaiting CSV or Excel file upload.")

if __name__ == "__main__":
    main()
