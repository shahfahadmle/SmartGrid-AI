import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model import LSTMModel


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT_DIR / "data" / "processed" / "hourly_features.csv"
MODEL_PATH = ROOT_DIR / "models" / "best_model.pth"
SCALER_X_PATH = ROOT_DIR / "models" / "scaler_X.pkl"
SCALER_Y_PATH = ROOT_DIR / "models" / "scaler_y.pkl"
METRICS_PATH = ROOT_DIR / "models" / "metrics.json"

FEATURE_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
]

TARGET_COLUMN = "Global_active_power"
LOOKBACK = 24

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PLOTLY_TEMPLATE = "plotly_white"


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="SmartGrid AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #f4f7fb;
        color: #111827;
    }

    .main {
        background-color: #f4f7fb;
        color: #111827;
    }

    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #d1d5db;
    }

    .block-container {
        padding-top: 1.5rem;
    }

    [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #cbd5e1;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
    }

    [data-testid="stMetricLabel"] {
        color: #475569;
    }

    [data-testid="stMetricValue"] {
        color: #0f172a;
    }

    .dashboard-title {
        font-size: 34px;
        font-weight: 800;
        color: #0f172a;
    }

    .dashboard-subtitle {
        color: #475569;
        font-size: 15px;
    }
</style>
""", unsafe_allow_html=True)

def add_time_features(df):
    """Create calendar features from the DateTime index."""

    df = df.copy()

    df["is_weekend"] = (
        df.index.dayofweek >= 5
    ).astype(int)

    df["hour_sin"] = np.sin(
        2 * np.pi * df.index.hour / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df.index.hour / 24
    )

    df["weekday_sin"] = np.sin(
        2 * np.pi * df.index.dayofweek / 7
    )

    df["weekday_cos"] = np.cos(
        2 * np.pi * df.index.dayofweek / 7
    )

    return df


def prepare_uploaded_data(file):
    """
    Supports:
1. Processed CSV with DateTime column.
2. Raw UCI semicolon-separated TXT/CSV with Date and Time.
    """

    if file is None:
        if not DATA_PATH.exists():
            return None

        df = pd.read_csv(DATA_PATH)
    else:
        file.seek(0)

        if file.name.lower().endswith(".txt"):
            df = pd.read_csv(
                file,
                sep=";",
                na_values=["?", "NA", ""]
            )
        else:
            # First try normal CSV.
            df = pd.read_csv(
                file,
                na_values=["?", "NA", ""]
            )

            # If it looks like a semicolon-separated UCI file,
            # read it again with the correct delimiter.
            if "Date" not in df.columns and "DateTime" not in df.columns:
                file.seek(0)
                df = pd.read_csv(
                    file,
                    sep=";",
                    na_values=["?", "NA", ""]
                )

    # Raw UCI dataset
    if "Date" in df.columns and "Time" in df.columns:
        df["DateTime"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            dayfirst=True,
            errors="coerce"
        )

        df = df.drop(columns=["Date", "Time"])

    if "DateTime" not in df.columns:
        raise ValueError(
            "Your file must contain DateTime, or Date and Time columns."
        )

    df["DateTime"] = pd.to_datetime(
        df["DateTime"],
        errors="coerce"
    )

    df = df.dropna(subset=["DateTime"])
    df = df.set_index("DateTime").sort_index()

    # Convert measurements to numeric.
    measurement_columns = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]

    for col in measurement_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # If data is minute-level, aggregate it to hourly.
    if len(df) > 0:
        median_interval = df.index.to_series().diff().median()

        if pd.notna(median_interval) and median_interval < pd.Timedelta(hours=1):
            aggregation = {}

            for col in measurement_columns:
                if col not in df.columns:
                    continue

                if col.startswith("Sub_metering"):
                    aggregation[col] = lambda x: x.sum(min_count=1)
                else:
                    aggregation[col] = "mean"

            df = df.resample("h").agg(aggregation)

    # Check required measurement columns.
    missing = [
        col for col in measurement_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Forward-fill only. Do not use future values to fill earlier rows.
    df[measurement_columns] = (
        df[measurement_columns]
        .ffill(limit=3)
    )

    df = add_time_features(df)

    # Remove rows that still contain missing values.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLUMNS)

    return df


def check_hourly_continuity(df):
    """Check that the latest model window contains consecutive hours."""

    if len(df) < LOOKBACK:
        return False

    last_window = df.index[-LOOKBACK:]

    differences = last_window.to_series().diff().dropna()

    return bool(
        (differences == pd.Timedelta(hours=1)).all()
    )


# --------------------------------------------------
# LOAD MODEL AND SCALERS
# --------------------------------------------------

@st.cache_resource
def load_model_and_scalers():

    required_files = [
        MODEL_PATH,
        SCALER_X_PATH,
        SCALER_Y_PATH
    ]

    missing_files = [
        str(path) for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing model files:\n" +
            "\n".join(missing_files)
        )

    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)

    model = LSTMModel(
        input_dim=len(FEATURE_COLUMNS),
        hidden_dim=64,
        num_layers=2,
        output_dim=1,
        dropout=0.2
    ).to(DEVICE)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model, scaler_X, scaler_y


# --------------------------------------------------
# FORECASTING
# --------------------------------------------------

def forecast_next_hours(
    model,
    scaler_X,
    scaler_y,
    historical_df,
    horizon=24
):
    """
    Recursive forecasting.

    Important:
    The model predicts only one next-hour value per call.
    For future hours, unavailable non-target measurements
    are carried forward from the last observed hour.

    This is an approximation, not a full multivariate forecast.
    """

    if len(historical_df) < LOOKBACK:
        raise ValueError(
            f"At least {LOOKBACK} hourly rows are required."
        )

    if not check_hourly_continuity(historical_df):
        raise ValueError(
            "The latest 24 rows are not consecutive hourly observations. "
            "Check missing hours in your dataset."
        )

    working_df = historical_df.copy()
    predictions = []

    for step in range(horizon):

        # Select the latest 24 hours.
        window = working_df[FEATURE_COLUMNS].tail(LOOKBACK)

        # Scale input using the training scaler.
        scaled_window = scaler_X.transform(
            window.to_numpy(dtype=np.float64)
        )

        x = torch.tensor(
            scaled_window,
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            prediction_scaled = model(x).cpu().numpy()

        # Convert scaled prediction back to kW.
        prediction = scaler_y.inverse_transform(
            prediction_scaled.reshape(-1, 1)
        )[0, 0]

        prediction = max(float(prediction), 0.0)

        next_timestamp = (
            working_df.index[-1] + pd.Timedelta(hours=1)
        )

        # Carry forward the last known measurements.
        next_row = working_df.iloc[-1].copy()

        next_row[TARGET_COLUMN] = prediction

        # Update future calendar features.
        next_row["is_weekend"] = int(
            next_timestamp.dayofweek >= 5
        )

        next_row["hour_sin"] = np.sin(
            2 * np.pi * next_timestamp.hour / 24
        )

        next_row["hour_cos"] = np.cos(
            2 * np.pi * next_timestamp.hour / 24
        )

        next_row["weekday_sin"] = np.sin(
            2 * np.pi * next_timestamp.dayofweek / 7
        )

        next_row["weekday_cos"] = np.cos(
            2 * np.pi * next_timestamp.dayofweek / 7
        )

        predictions.append({
            "DateTime": next_timestamp,
            "Predicted_Global_active_power": prediction
        })

        new_row_df = pd.DataFrame(
            [next_row],
            index=[next_timestamp]
        )

        working_df = pd.concat([
            working_df,
            new_row_df
        ])

    return pd.DataFrame(predictions)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("⚡ SmartGrid AI")

st.sidebar.markdown(
    "Electricity Consumption Forecasting"
)

st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader(
    "Upload electricity dataset",
    type=["csv", "txt"]
)

st.sidebar.caption(
    "Upload your processed hourly CSV or raw UCI dataset."
)

st.sidebar.divider()

forecast_horizon = st.sidebar.slider(
    "Forecast horizon (hours)",
    min_value=1,
    max_value=48,
    value=24
)

st.sidebar.caption(
    f"Running on: {DEVICE}"
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

st.markdown(
    '<div class="dashboard-title">⚡ SmartGrid AI Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'Electricity consumption analysis and LSTM forecasting'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

try:
    with st.spinner("Loading electricity data..."):
        df = prepare_uploaded_data(uploaded_file)

    if df is None:
        st.error(
            "Dataset not found. Put hourly_features.csv inside "
            "data/processed/ or upload a CSV/TXT file."
        )
        st.stop()

    if df.empty:
        st.error("The dataset contains no usable rows.")
        st.stop()

except Exception as error:
    st.error(f"Data loading error: {error}")
    st.stop()


# --------------------------------------------------
# DATE FILTER
# --------------------------------------------------

st.sidebar.subheader("Historical date filter")

min_date = df.index.min().date()
max_date = df.index.max().date()

selected_dates = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = selected_dates

    filtered_df = df.loc[
        (df.index.date >= start_date) &
        (df.index.date <= end_date)
    ]
else:
    filtered_df = df.copy()

if filtered_df.empty:
    st.warning("No records available for the selected date range.")
    st.stop()


# --------------------------------------------------
# KPI SECTION
# --------------------------------------------------

total_energy = filtered_df[TARGET_COLUMN].sum()

average_power = filtered_df[TARGET_COLUMN].mean()

peak_power = filtered_df[TARGET_COLUMN].max()

minimum_power = filtered_df[TARGET_COLUMN].min()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Estimated Energy",
    f"{total_energy:,.2f} kWh",
    help="Sum of hourly average power values. Assumes one-hour intervals."
)

col2.metric(
    "Average Power",
    f"{average_power:.3f} kW"
)

col3.metric(
    "Peak Power",
    f"{peak_power:.3f} kW"
)

col4.metric(
    "Minimum Power",
    f"{minimum_power:.3f} kW"
)

st.divider()


# --------------------------------------------------
# TABS
# --------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Historical Analysis",
    "Forecasting",
    "Data Explorer",
    "Model Information"
])


# ==================================================
# TAB 1: OVERVIEW
# ==================================================

with tab1:

    st.subheader("Electricity Consumption Overview")

    fig = px.line(
        filtered_df,
        x=filtered_df.index,
        y=TARGET_COLUMN,
        title="Hourly Global Active Power",
        labels={
            "x": "Date and Time",
            TARGET_COLUMN: "Active Power (kW)"
        }
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:

        daily = filtered_df[TARGET_COLUMN].resample("D").mean()

        fig_daily = px.bar(
            x=daily.index,
            y=daily.values,
            title="Daily Average Power",
            labels={
                "x": "Date",
                "y": "Average Power (kW)"
            }
        )

        fig_daily.update_layout(template=PLOTLY_TEMPLATE)

        st.plotly_chart(
            fig_daily,
            use_container_width=True
        )

    with col2:

        hourly = filtered_df.groupby(
            filtered_df.index.hour
        )[TARGET_COLUMN].mean()

        fig_hourly = px.line(
            x=hourly.index,
            y=hourly.values,
            markers=True,
            title="Average Consumption by Hour",
            labels={
                "x": "Hour of Day",
                "y": "Average Power (kW)"
            }
        )

        fig_hourly.update_layout(template=PLOTLY_TEMPLATE)

        st.plotly_chart(
            fig_hourly,
            use_container_width=True
        )


# ==================================================
# TAB 2: HISTORICAL ANALYSIS
# ==================================================

with tab2:

    st.subheader("Historical Consumption Analysis")

    aggregation = st.selectbox(
        "Choose aggregation",
        ["Hourly", "Daily", "Weekly", "Monthly"]
    )

    if aggregation == "Hourly":
        chart_data = filtered_df[TARGET_COLUMN]

    elif aggregation == "Daily":
        chart_data = filtered_df[TARGET_COLUMN].resample("D").mean()

    elif aggregation == "Weekly":
        chart_data = filtered_df[TARGET_COLUMN].resample("W").mean()

    else:
        chart_data = filtered_df[TARGET_COLUMN].resample("MS").mean()

    fig = px.area(
        x=chart_data.index,
        y=chart_data.values,
        title=f"{aggregation} Average Active Power",
        labels={
            "x": "Date",
            "y": "Power (kW)"
        }
    )

    fig.update_layout(template=PLOTLY_TEMPLATE)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Consumption Distribution")

    fig_hist = px.histogram(
        filtered_df,
        x=TARGET_COLUMN,
        nbins=50,
        title="Active Power Distribution"
    )

    fig_hist.update_layout(template=PLOTLY_TEMPLATE)

    st.plotly_chart(
        fig_hist,
        use_container_width=True
    )

    st.subheader("Power Measurement Relationships")

    correlation_columns = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3"
    ]

    corr = filtered_df[correlation_columns].corr()

    fig_corr = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Feature Correlation Matrix",
        color_continuous_scale="RdBu_r"
    )

    fig_corr.update_layout(template=PLOTLY_TEMPLATE)

    st.plotly_chart(
        fig_corr,
        use_container_width=True
    )


# ==================================================
# TAB 3: FORECASTING
# ==================================================

with tab3:

    st.subheader("LSTM Electricity Forecasting")

    st.write(
        f"Forecast the next {forecast_horizon} hours "
        "using the latest 24 hourly observations."
    )

    if st.button(
        "Generate Forecast",
        type="primary"
    ):

        try:
            with st.spinner("Loading model and generating forecast..."):

                model, scaler_X, scaler_y = load_model_and_scalers()

                forecast_df = forecast_next_hours(
                    model=model,
                    scaler_X=scaler_X,
                    scaler_y=scaler_y,
                    historical_df=df,
                    horizon=forecast_horizon
                )

                st.session_state["forecast_df"] = forecast_df

            st.success("Forecast generated successfully!")

        except Exception as error:
            st.error(f"Forecasting error: {error}")

    if "forecast_df" in st.session_state:

        forecast_df = st.session_state["forecast_df"]

        fig = go.Figure()

        # Show recent historical values.
        recent_history = df.tail(168)

        fig.add_trace(
            go.Scatter(
                x=recent_history.index,
                y=recent_history[TARGET_COLUMN],
                mode="lines",
                name="Historical Power"
            )
        )

        # Show forecast.
        fig.add_trace(
            go.Scatter(
                x=forecast_df["DateTime"],
                y=forecast_df["Predicted_Global_active_power"],
                mode="lines+markers",
                name="Forecasted Power",
                line=dict(dash="dash")
            )
        )

        fig.update_layout(
            title="Historical vs Forecasted Electricity Consumption",
            xaxis_title="Date and Time",
            yaxis_title="Active Power (kW)",
            template=PLOTLY_TEMPLATE,
            hovermode="x unified"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        forecast_average = (
            forecast_df["Predicted_Global_active_power"].mean()
        )

        forecast_peak = (
            forecast_df["Predicted_Global_active_power"].max()
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Forecast Average Power",
            f"{forecast_average:.3f} kW"
        )

        col2.metric(
            "Forecast Peak Power",
            f"{forecast_peak:.3f} kW"
        )

        st.subheader("Forecast Values")

        st.dataframe(
            forecast_df,
            use_container_width=True
        )

        csv = forecast_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Forecast CSV",
            data=csv,
            file_name="electricity_forecast.csv",
            mime="text/csv"
        )

        st.info(
            "Forecasting assumption: future reactive power, voltage, "
            "intensity, and sub-metering features are carried forward "
            "from the last observed hour. For more reliable multi-hour "
            "forecasts, train a direct multi-step model or forecast "
            "those future features separately."
        )


# ==================================================
# TAB 4: DATA EXPLORER
# ==================================================

with tab4:

    st.subheader("Dataset Explorer")

    st.write("Dataset shape:", df.shape)

    st.write("Date range:")
    st.write(
        f"{df.index.min()} to {df.index.max()}"
    )

    st.write("Missing values:")

    missing_values = df[FEATURE_COLUMNS].isna().sum()

    st.dataframe(
        missing_values.rename("Missing Values"),
        use_container_width=True
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        filtered_df.tail(100),
        use_container_width=True
    )

    csv = filtered_df.to_csv().encode("utf-8")

    st.download_button(
        "Download Historical Data",
        data=csv,
        file_name="historical_electricity_data.csv",
        mime="text/csv"
    )


# ==================================================
# TAB 5: MODEL INFORMATION
# ==================================================

with tab5:

    st.subheader("Model Architecture")

    st.markdown("""
    **Model:** Long Short-Term Memory (LSTM)

    **Framework:** PyTorch

    **Input sequence:** Previous 24 hours

    **Input features:** 12

    **Hidden dimensions:** 64

    **LSTM layers:** 2

    **Dropout:** 0.2

    **Output:** Next-hour Global Active Power

    **Loss function:** Mean Squared Error

    **Optimizer:** Adam
    """)

    st.subheader("Model Evaluation Metrics")

    if METRICS_PATH.exists():

        try:
            with open(METRICS_PATH, "r") as file:
                metrics = json.load(file)

            metric_columns = st.columns(len(metrics))

            for index, (name, value) in enumerate(metrics.items()):

                metric_columns[index].metric(
                    name.upper(),
                    f"{value:.4f}"
                    if isinstance(value, (float, int))
                    else str(value)
                )

        except Exception as error:
            st.error(f"Could not read metrics file: {error}")

    else:
        st.warning(
            "No metrics.json file found. Save your test-set metrics "
            "to models/metrics.json to display them here."
        )

    st.subheader("Runtime Information")

    st.write("PyTorch device:", DEVICE)

    st.write("Forecast lookback:", f"{LOOKBACK} hours")

    st.write("Feature count:", len(FEATURE_COLUMNS))

    st.write("Model file:", str(MODEL_PATH))


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "SmartGrid AI | Electricity Consumption Forecasting "
    "| Built with PyTorch and Streamlit"
)