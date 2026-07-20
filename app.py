import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="BSNL Customer Segmentation",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Constants — must match train_model.py exactly
# ============================================================
NUMERICAL_COLS = [
    "SeniorCitizen",
    "Tenure_Months",
    "MonthlyCharges",
    "TotalCharges"
]

CATEGORICAL_COLS = [
    "Gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "InternetService",
    "Contract",
    "PaymentMethod",
    "PaperlessBilling",
    "StreamingTV",
    "StreamingMovies",
    "TechSupport",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "MultipleLines"
]

REQUIRED_COLS = NUMERICAL_COLS + CATEGORICAL_COLS

CLUSTER_MAP = {
    0: "👑 Loyal High-Value Customers",
    1: "🆕 New Customers",
    2: "⚠️ At-Risk Customers",
    3: "💎 Premium Service Users"
}

CLUSTER_COLORS = {
    "👑 Loyal High-Value Customers": "#2E86AB",
    "🆕 New Customers": "#06A77D",
    "⚠️ At-Risk Customers": "#D62839",
    "💎 Premium Service Users": "#9B5DE5"
}

MODEL_DIR = "model"


# ============================================================
# Load Model Artifacts (cached)
# ============================================================
@st.cache_resource
def load_artifacts():
    preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.pkl")
    kmeans_path = os.path.join(MODEL_DIR, "kmeans_model.pkl")
    sil_path = os.path.join(MODEL_DIR, "silhouette_score.pkl")

    missing = [p for p in [preprocessor_path, kmeans_path, sil_path] if not os.path.exists(p)]
    if missing:
        return None, None, None, missing

    preprocessor = joblib.load(preprocessor_path)
    kmeans = joblib.load(kmeans_path)
    sil_score = joblib.load(sil_path)
    return preprocessor, kmeans, sil_score, []


preprocessor, kmeans, sil_score, missing_files = load_artifacts()


# ============================================================
# Helper Functions
# ============================================================
def clean_column_names(df):
    """Best-effort fix for common casing / spacing mismatches."""
    rename_map = {}
    lower_map = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for target in REQUIRED_COLS:
        key = target.lower().replace(" ", "").replace("_", "")
        if key in lower_map and lower_map[key] != target:
            rename_map[lower_map[key]] = target
    return df.rename(columns=rename_map)


def validate_columns(df):
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    return missing_cols


def predict_segments(df):
    df = df.copy()
    X = preprocessor.transform(df)
    clusters = kmeans.predict(X)
    df["Cluster"] = clusters
    df["Customer Segment"] = df["Cluster"].map(CLUSTER_MAP)
    return df


@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")


def load_uploaded_file(uploaded_file):
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.title("📡 BSNL Segmentation")
    st.markdown("Upload your customer data to generate segments and explore the dashboard.")

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="File must contain the same columns used during training."
    )

    st.markdown("---")
    st.markdown("**Expected columns**")
    with st.expander("Numerical columns"):
        st.write(NUMERICAL_COLS)
    with st.expander("Categorical columns"):
        st.write(CATEGORICAL_COLS)

    st.markdown("---")
    if sil_score is not None:
        st.metric("Model Silhouette Score", f"{sil_score:.3f}")
    st.caption("Built with KMeans clustering (k=4)")


# ============================================================
# Guard: Missing model artifacts
# ============================================================
if missing_files:
    st.error(
        "⚠️ Model files not found. Please make sure the following files are present "
        f"in the `model/` folder alongside `app.py`:\n\n" +
        "\n".join(f"- `{m}`" for m in missing_files) +
        "\n\nRun `train_model.py` first to generate them."
    )
    st.stop()


# ============================================================
# Main Header
# ============================================================
st.title("📡 BSNL Customer Segmentation Dashboard")
st.markdown(
    "An interactive dashboard for exploring customer segments derived from "
    "KMeans clustering on telecom subscription and billing data."
)

# ============================================================
# No file uploaded yet
# ============================================================
if uploaded_file is None:
    st.info("👈 Upload a CSV or Excel file from the sidebar to get started.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 👑 Loyal High-Value")
        st.caption("Long tenure, high spend, low churn risk.")
    with col2:
        st.markdown("### 🆕 New Customers")
        st.caption("Recently onboarded, still building loyalty.")
    with col3:
        st.markdown("### ⚠️ At-Risk")
        st.caption("Signals suggest higher churn probability.")

    col4, col5 = st.columns(2)
    with col4:
        st.markdown("### 💎 Premium Service Users")
        st.caption("High engagement across add-on services.")

    st.stop()


# ============================================================
# Process Uploaded File
# ============================================================
try:
    raw_df = load_uploaded_file(uploaded_file)
except Exception as e:
    st.error(f"Could not read the uploaded file: {e}")
    st.stop()

raw_df = clean_column_names(raw_df)
missing_cols = validate_columns(raw_df)

if missing_cols:
    st.error(
        "The uploaded file is missing the following required columns:\n\n" +
        "\n".join(f"- `{c}`" for c in missing_cols)
    )
    st.dataframe(raw_df.head())
    st.stop()

try:
    result_df = predict_segments(raw_df)
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

st.success(f"✅ Segmented {len(result_df):,} customers successfully.")

# ============================================================
# KPI Row
# ============================================================
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("Total Customers", f"{len(result_df):,}")
with k2:
    st.metric("Avg Monthly Charges", f"₹{result_df['MonthlyCharges'].mean():,.2f}")
with k3:
    st.metric("Avg Tenure (Months)", f"{result_df['Tenure_Months'].mean():.1f}")
with k4:
    at_risk_pct = (result_df["Customer Segment"] == "⚠️ At-Risk Customers").mean() * 100
    st.metric("At-Risk Share", f"{at_risk_pct:.1f}%")

st.markdown("---")

# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Segment Overview", "💰 Revenue & Tenure", "🧾 Service Usage", "📥 Data & Export"]
)

# ------------------------------------------------------------
# Tab 1: Segment Overview
# ------------------------------------------------------------
with tab1:
    col1, col2 = st.columns([1, 1.3])

    with col1:
        seg_counts = result_df["Customer Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]

        fig_pie = px.pie(
            seg_counts,
            names="Segment",
            values="Count",
            color="Segment",
            color_discrete_map=CLUSTER_COLORS,
            hole=0.45,
            title="Customer Segment Distribution"
        )
        fig_pie.update_traces(textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        fig_bar = px.bar(
            seg_counts.sort_values("Count", ascending=True),
            x="Count",
            y="Segment",
            color="Segment",
            color_discrete_map=CLUSTER_COLORS,
            orientation="h",
            title="Customers per Segment",
            text="Count"
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("#### Segment Profile Summary")
    profile = result_df.groupby("Customer Segment").agg(
        Customers=("Customer Segment", "count"),
        Avg_Tenure=("Tenure_Months", "mean"),
        Avg_Monthly_Charges=("MonthlyCharges", "mean"),
        Avg_Total_Charges=("TotalCharges", "mean"),
        Senior_Citizen_Pct=("SeniorCitizen", "mean")
    ).reset_index()
    profile["Senior_Citizen_Pct"] = (profile["Senior_Citizen_Pct"] * 100).round(1)
    profile["Avg_Tenure"] = profile["Avg_Tenure"].round(1)
    profile["Avg_Monthly_Charges"] = profile["Avg_Monthly_Charges"].round(2)
    profile["Avg_Total_Charges"] = profile["Avg_Total_Charges"].round(2)
    st.dataframe(profile, use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# Tab 2: Revenue & Tenure
# ------------------------------------------------------------
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        fig_box = px.box(
            result_df,
            x="Customer Segment",
            y="MonthlyCharges",
            color="Customer Segment",
            color_discrete_map=CLUSTER_COLORS,
            title="Monthly Charges by Segment"
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    with col2:
        fig_tenure = px.box(
            result_df,
            x="Customer Segment",
            y="Tenure_Months",
            color="Customer Segment",
            color_discrete_map=CLUSTER_COLORS,
            title="Tenure (Months) by Segment"
        )
        fig_tenure.update_layout(showlegend=False)
        st.plotly_chart(fig_tenure, use_container_width=True)

    fig_scatter = px.scatter(
        result_df,
        x="Tenure_Months",
        y="MonthlyCharges",
        color="Customer Segment",
        color_discrete_map=CLUSTER_COLORS,
        opacity=0.6,
        title="Tenure vs Monthly Charges",
        hover_data=["TotalCharges", "Contract"]
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    fig_contract = px.histogram(
        result_df,
        x="Contract",
        color="Customer Segment",
        color_discrete_map=CLUSTER_COLORS,
        barmode="group",
        title="Contract Type by Segment"
    )
    st.plotly_chart(fig_contract, use_container_width=True)

# ------------------------------------------------------------
# Tab 3: Service Usage
# ------------------------------------------------------------
with tab3:
    service_cols = [
        "InternetService", "TechSupport", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "StreamingTV",
        "StreamingMovies", "MultipleLines"
    ]

    selected_service = st.selectbox("Select a service to analyze", service_cols)

    fig_service = px.histogram(
        result_df,
        x=selected_service,
        color="Customer Segment",
        color_discrete_map=CLUSTER_COLORS,
        barmode="group",
        title=f"{selected_service} Distribution by Segment"
    )
    st.plotly_chart(fig_service, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_payment = px.histogram(
            result_df,
            x="PaymentMethod",
            color="Customer Segment",
            color_discrete_map=CLUSTER_COLORS,
            barmode="group",
            title="Payment Method by Segment"
        )
        fig_payment.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_payment, use_container_width=True)

    with col2:
        fig_paperless = px.histogram(
            result_df,
            x="PaperlessBilling",
            color="Customer Segment",
            color_discrete_map=CLUSTER_COLORS,
            barmode="group",
            title="Paperless Billing by Segment"
        )
        st.plotly_chart(fig_paperless, use_container_width=True)

# ------------------------------------------------------------
# Tab 4: Data & Export
# ------------------------------------------------------------
with tab4:
    st.markdown("#### Filter Segmented Data")
    segments_selected = st.multiselect(
        "Filter by segment",
        options=list(CLUSTER_MAP.values()),
        default=list(CLUSTER_MAP.values())
    )

    filtered_df = result_df[result_df["Customer Segment"].isin(segments_selected)]
    st.dataframe(filtered_df, use_container_width=True, height=400)

    csv_data = convert_df_to_csv(filtered_df)
    st.download_button(
        label="⬇️ Download Segmented Data (CSV)",
        data=csv_data,
        file_name="Segmented_BSNL_Customers.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.caption(
        "Segmentation generated using a pre-trained KMeans model (k=4) with a "
        "StandardScaler + OneHotEncoder preprocessing pipeline."
    )
