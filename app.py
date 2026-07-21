import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Universal Customer Segmentation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

SEGMENT_COLORS = [
    "#2E86AB", "#06A77D", "#D62839", "#9B5DE5",
    "#F4A259", "#5DA9E9", "#E07A5F", "#43AA8B", "#8E44AD", "#118AB2"
]

SAMPLE_LIMIT_FOR_SILHOUETTE = 5000   # silhouette_score is O(n^2), keep it fast on big data
SAMPLE_LIMIT_FOR_ELBOW = 3000        # elbow re-fits KMeans many times, sample harder

# Ordered low-value -> high-value naming palette. Whatever K the user picks (2-10),
# we rank the resulting clusters by a composite "value" score and label them with
# a suitable spread of these tiers instead of generic "Segment 0/1/2..." names.
SEGMENT_TIERS = [
    ("Low Value Customers",     "Low usage, low recharge, short tenure — highest churn risk"),
    ("Budget Customers",        "Below-average usage and spend, price-sensitive"),
    ("Regular Customers",       "Moderate usage, regular recharge, medium tenure"),
    ("Growing Customers",       "Rising engagement, medium tenure, upsell potential"),
    ("Steady Customers",        "Consistent moderate-to-high engagement"),
    ("Engaged Customers",       "Above-average usage and recharge frequency"),
    ("High Value Customers",    "High usage, high recharge, long tenure — loyal customers"),
    ("Premium Customers",       "Very high usage and spend, strong tenure"),
    ("VIP Customers",           "Top-tier engagement, spend and loyalty"),
    ("Elite Customers",         "Highest value, longest tenure, most loyal customers"),
]


def get_segment_tiers(k):
    """Pick k evenly-spread (name, description) tiers, ordered low -> high value."""
    n_tiers = len(SEGMENT_TIERS)
    if k > n_tiers:
        # Shouldn't happen given the K slider is capped at 10, but stay safe.
        return [SEGMENT_TIERS[i % n_tiers] for i in range(k)]
    idxs = sorted(set(int(round(i)) for i in np.linspace(0, n_tiers - 1, k)))
    i = 0
    while len(idxs) < k:
        if i not in idxs:
            idxs.append(i)
            idxs = sorted(idxs)
        i += 1
    return [SEGMENT_TIERS[i] for i in idxs]


def name_clusters(cleaned_df, numeric_cols, labels, k):
    """
    Rank clusters by a composite 'value' score (mean z-score across numeric
    features) and map them to suitable business-style names, low -> high value.
    Falls back to natural cluster order if there are no numeric columns to rank by.
    Returns: cluster_to_name (dict), cluster_to_desc (dict), tier_order (list of names, low->high)
    """
    tiers = get_segment_tiers(k)
    tier_order = [name for name, _ in tiers]

    if numeric_cols:
        z = (cleaned_df[numeric_cols] - cleaned_df[numeric_cols].mean()) / cleaned_df[numeric_cols].std(ddof=0).replace(0, 1)
        composite = z.mean(axis=1)
        cluster_score = pd.Series(composite.values).groupby(labels).mean()
        ranked_clusters = cluster_score.sort_values().index.tolist()  # low value -> high value
    else:
        ranked_clusters = list(range(k))

    cluster_to_name, cluster_to_desc = {}, {}
    for rank, cluster_id in enumerate(ranked_clusters):
        name, desc = tiers[rank]
        cluster_to_name[cluster_id] = name
        cluster_to_desc[cluster_id] = desc

    return cluster_to_name, cluster_to_desc, tier_order


def ellipse_path_points(x, y, n_std=2.2, n_points=100):
    """Return (xs, ys) tracing a tilted ellipse around a cluster's points, sized to its spread."""
    if len(x) < 3:
        return None, None
    cov = np.cov(x, y)
    if np.any(np.isnan(cov)):
        return None, None
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    a, b = n_std * np.sqrt(np.maximum(eigvals, 0))
    t = np.linspace(0, 2 * np.pi, n_points)
    circle = np.stack([a * np.cos(t), b * np.sin(t)])
    ellipse = eigvecs @ circle
    return ellipse[0] + np.mean(x), ellipse[1] + np.mean(y)


def render_cluster_scatter(X, km_model, result_df, color_map):
    """
    Real 2D visualization of the clusters actually fitted on this dataset.
    Reduces to 2D via PCA when there are more than 2 features, so the plot
    reflects the true geometry KMeans clustered on (not a mocked-up example).
    """
    X = np.asarray(X)
    if X.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X)
        centers = pca.transform(km_model.cluster_centers_)
        xlabel, ylabel = "PCA Component 1", "PCA Component 2"
    else:
        coords = X
        centers = km_model.cluster_centers_
        xlabel, ylabel = "Feature 1", "Feature 2"

    clusters = result_df["Cluster"].to_numpy()
    segments = result_df["Segment"].to_numpy()

    fig = go.Figure()
    for cluster_id in sorted(result_df["Cluster"].unique()):
        mask = clusters == cluster_id
        seg_name = segments[mask][0]
        color = color_map.get(seg_name, "#888888")
        cx, cy = coords[mask, 0], coords[mask, 1]

        fig.add_trace(go.Scatter(
            x=cx, y=cy, mode="markers", name=seg_name,
            marker=dict(size=6, color=color, opacity=0.75),
            legendgroup=seg_name
        ))

        ex, ey = ellipse_path_points(cx, cy, n_std=2.2)
        if ex is not None:
            fig.add_trace(go.Scatter(
                x=ex, y=ey, mode="lines", line=dict(color="darkred", width=2),
                showlegend=False, hoverinfo="skip", legendgroup=seg_name
            ))

        fig.add_trace(go.Scatter(
            x=[centers[cluster_id, 0]], y=[centers[cluster_id, 1]], mode="markers",
            marker=dict(symbol="x", size=14, color="red", line=dict(color="black", width=1)),
            showlegend=False, hoverinfo="skip", legendgroup=seg_name
        ))

    fig.update_layout(
        title="Clustering",
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        legend=dict(orientation="v"),
        height=520,
        margin=dict(t=50, b=40)
    )
    return fig



# ============================================================
# Helpers — Loading
# ============================================================
def load_uploaded_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


# ============================================================
# Helpers — Auto profiling / cleaning
# ============================================================
def profile_dataframe(df):
    """Decide numeric vs categorical vs excluded columns, and report why."""
    n_rows = len(df)
    numeric_cols, categorical_cols, excluded = [], [], []

    for col in df.columns:
        series = df[col]
        nunique = series.nunique(dropna=True)

        # Constant column
        if nunique <= 1:
            excluded.append((col, "constant value (no variation)"))
            continue

        # Datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            excluded.append((col, "datetime column (not used for clustering)"))
            continue

        # ID-like: every row unique, or name suggests an identifier with high cardinality
        looks_like_id = "id" in col.lower()
        if nunique == n_rows or (looks_like_id and nunique > 0.5 * n_rows):
            excluded.append((col, "looks like a unique identifier"))
            continue

        # Numeric
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            numeric_cols.append(col)
            continue

        # Categorical / boolean
        if nunique <= 50:
            categorical_cols.append(col)
        else:
            excluded.append((col, f"too many unique values ({nunique}) for one-hot encoding"))

    return numeric_cols, categorical_cols, excluded


def clean_dataframe(df, numeric_cols, categorical_cols):
    """Impute missing values, report what was filled."""
    df = df.copy()
    fill_report = []

    for col in numeric_cols:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            fill_report.append((col, n_missing, f"median ({median_val:.2f})"))

    for col in categorical_cols:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            mode_val = df[col].mode(dropna=True)
            mode_val = mode_val.iloc[0] if len(mode_val) else "Unknown"
            df[col] = df[col].fillna(mode_val)
            fill_report.append((col, n_missing, f"most frequent value ('{mode_val}')"))

    return df, fill_report


def build_pipeline(numeric_cols, categorical_cols):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )


def analyze_feature_importance(cleaned_df, numeric_cols, categorical_cols, k, sample_limit=3000):
    """
    Fit a quick baseline KMeans on ALL candidate columns, then train a RandomForest
    to predict the resulting cluster label. The RF's feature importances show which
    columns actually drove the split vs which ones are just adding noise/dimensions.
    Returns: importance_df (Column, Importance, Type), n_numeric_dims, n_categorical_dims, silhouette
    """
    df = cleaned_df
    if len(df) > sample_limit:
        df = df.sample(sample_limit, random_state=42)

    pre = build_pipeline(numeric_cols, categorical_cols)
    X = pre.fit_transform(df[numeric_cols + categorical_cols])
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    labels = km.labels_
    sil = silhouette_score(X, labels) if len(set(labels)) > 1 else float("nan")

    feat_names = list(numeric_cols)
    n_numeric_dims = len(numeric_cols)
    if categorical_cols:
        cat_names = list(pre.named_transformers_["cat"].get_feature_names_out(categorical_cols))
        feat_names += cat_names
    n_categorical_dims = len(feat_names) - n_numeric_dims

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(X, labels)
    imp = pd.Series(rf.feature_importances_, index=feat_names)

    def base_col(f):
        if f in numeric_cols:
            return f
        for c in categorical_cols:
            if f.startswith(c + "_"):
                return c
        return f

    agg = imp.groupby([base_col(f) for f in feat_names]).sum().sort_values(ascending=False)
    col_type = {c: ("Numeric" if c in numeric_cols else "Categorical") for c in agg.index}
    importance_df = pd.DataFrame({
        "Column": agg.index,
        "Importance": agg.values,
        "Type": [col_type[c] for c in agg.index]
    })
    return importance_df, n_numeric_dims, n_categorical_dims, sil


# ============================================================
# Helpers — Clustering & Metrics
# ============================================================
def sample_for_metric(X, labels, limit):
    n = X.shape[0]
    if n <= limit:
        return X, labels
    rng = np.random.RandomState(42)
    idx = rng.choice(n, size=limit, replace=False)
    X_sampled = X[idx] if not hasattr(X, "toarray") else X[idx].toarray()
    return X_sampled, labels[idx]


def safe_silhouette(X, labels):
    X_arr = X.toarray() if hasattr(X, "toarray") else X
    X_s, labels_s = sample_for_metric(X_arr, labels, SAMPLE_LIMIT_FOR_SILHOUETTE)
    was_sampled = X_s.shape[0] < X_arr.shape[0]
    score = silhouette_score(X_s, labels_s)
    return score, was_sampled, X_s.shape[0]


@st.cache_data(show_spinner=False)
def run_elbow_analysis(_X_hash, X_dense, k_range):
    """X_dense must already be a small, dense sample."""
    wcss, sil = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = km.fit_predict(X_dense)
        wcss.append(km.inertia_)
        sil.append(silhouette_score(X_dense, labels))
    return wcss, sil


def run_clustering(X, k):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    return km, labels


# ============================================================
# Session State Helpers
# ============================================================
def reset_results():
    for key in ["result_df", "km_model", "preprocessor", "silhouette", "sil_sampled",
                "sil_n", "numeric_cols", "categorical_cols", "excluded_cols", "fill_report",
                "k_used", "X", "segment_desc_map", "tier_order",
                "importance_df", "importance_dims", "importance_sil"]:
        st.session_state.pop(key, None)


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.title("📊 Universal Segmentation")
    st.markdown(
        "Upload **any** customer dataset. The app profiles it, cleans it, "
        "and clusters it — no fixed schema, no pre-trained model."
    )

    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

    # Detect new file -> clear previous results
    if uploaded_file is not None:
        file_signature = (uploaded_file.name, uploaded_file.size)
        if st.session_state.get("file_signature") != file_signature:
            st.session_state["file_signature"] = file_signature
            reset_results()

    st.markdown("---")
    st.markdown("**How this works**")
    st.caption(
        "1. Detects numeric vs categorical columns automatically\n\n"
        "2. Cleans missing values and drops ID-like columns\n\n"
        "3. Scales numeric features, one-hot encodes categorical ones\n\n"
        "4. Runs KMeans with the K you choose\n\n"
        "5. Reports a silhouette score computed fresh for *this* run"
    )


# ============================================================
# Main — No file uploaded
# ============================================================
st.title("📊 Universal Customer Segmentation Dashboard")
st.markdown(
    "Drop in a dataset — telecom, retail, banking, anything with customer-level rows — "
    "and this dashboard will clean it, help you pick the right number of segments, "
    "and cluster it live."
)

if uploaded_file is None:
    st.info("👈 Upload a CSV or Excel file from the sidebar to get started.")
    st.stop()

try:
    raw_df = load_uploaded_file(uploaded_file)
except Exception as e:
    st.error(f"Could not read the uploaded file: {e}")
    st.stop()

st.success(f"Loaded **{uploaded_file.name}** — {raw_df.shape[0]:,} rows × {raw_df.shape[1]} columns")

with st.expander("🔍 Preview raw data", expanded=True):
    st.dataframe(raw_df.head(20), use_container_width=True)

# ============================================================
# Profiling & Cleaning
# ============================================================
numeric_cols, categorical_cols, excluded_cols = profile_dataframe(raw_df)

if not numeric_cols and not categorical_cols:
    st.error("No usable columns were detected for clustering. Please check your file.")
    st.stop()

cleaned_df, fill_report = clean_dataframe(raw_df, numeric_cols, categorical_cols)

with st.expander("🧹 Data Cleaning & Preprocessing Report", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Numeric columns (scaled)**")
        st.write(numeric_cols if numeric_cols else "None detected")
        st.markdown("**Categorical columns (one-hot encoded)**")
        st.write(categorical_cols if categorical_cols else "None detected")
    with c2:
        st.markdown("**Excluded columns**")
        if excluded_cols:
            for col, reason in excluded_cols:
                st.write(f"- `{col}` — {reason}")
        else:
            st.write("None excluded")

    st.markdown("**Missing values filled**")
    if fill_report:
        fill_df = pd.DataFrame(fill_report, columns=["Column", "Missing Values", "Filled With"])
        st.dataframe(fill_df, use_container_width=True, hide_index=True)
    else:
        st.write("No missing values found.")

# ============================================================
# Feature Selection & Importance Check
# ============================================================
st.markdown("---")
st.markdown("### 🎯 Choose Features & Check Their Importance")
st.caption(
    "Every categorical column gets one-hot encoded into several dimensions, which can "
    "outnumber and drown out your numeric columns in the distance calculation KMeans uses. "
    "Run the check below to see what's actually driving the clusters before committing to K."
)

fc1, fc2 = st.columns(2)
with fc1:
    sel_numeric_cols = st.multiselect(
        "Numeric columns to use", options=numeric_cols, default=numeric_cols
    )
with fc2:
    sel_categorical_cols = st.multiselect(
        "Categorical columns to use", options=categorical_cols, default=categorical_cols
    )

if not sel_numeric_cols and not sel_categorical_cols:
    st.error("Select at least one column to cluster on.")
    st.stop()

if st.button("🔎 Analyze Feature Importance"):
    with st.spinner("Fitting a quick baseline model to rank feature importance..."):
        probe_k = 4 if len(cleaned_df) >= 4 else 2
        importance_df, n_num_dims, n_cat_dims, probe_sil = analyze_feature_importance(
            cleaned_df, sel_numeric_cols, sel_categorical_cols, probe_k
        )
        st.session_state["importance_df"] = importance_df
        st.session_state["importance_dims"] = (n_num_dims, n_cat_dims)
        st.session_state["importance_sil"] = probe_sil

if "importance_df" in st.session_state:
    importance_df = st.session_state["importance_df"]
    n_num_dims, n_cat_dims = st.session_state["importance_dims"]
    probe_sil = st.session_state["importance_sil"]

    d1, d2, d3 = st.columns(3)
    with d1:
        st.metric("Numeric dimensions", n_num_dims)
    with d2:
        st.metric("Categorical dimensions (one-hot)", n_cat_dims)
    with d3:
        st.metric("Baseline silhouette (K=4 probe)", f"{probe_sil:.3f}" if not np.isnan(probe_sil) else "n/a")

    if n_cat_dims > 3 * max(n_num_dims, 1):
        st.warning(
            f"Categorical one-hot columns ({n_cat_dims} dims) heavily outnumber your numeric "
            f"columns ({n_num_dims} dims). Low-importance categoricals below are likely adding "
            "noise rather than signal — consider dropping them from the multiselect above."
        )

    fig_imp = px.bar(
        importance_df, x="Importance", y="Column", color="Type", orientation="h",
        title="What actually drives the cluster split (higher = more important)"
    )
    fig_imp.update_layout(yaxis=dict(categoryorder="total ascending"), height=max(300, 28 * len(importance_df)))
    st.plotly_chart(fig_imp, use_container_width=True)

    mean_imp = importance_df["Importance"].mean()
    low_imp_cols = importance_df.loc[importance_df["Importance"] < mean_imp, "Column"].tolist()
    if low_imp_cols:
        st.caption(
            "Below-average importance (candidates to drop): " + ", ".join(f"`{c}`" for c in low_imp_cols)
        )

numeric_cols, categorical_cols = sel_numeric_cols, sel_categorical_cols

# ============================================================
# Build preprocessing pipeline & transform
# ============================================================
preprocessor = build_pipeline(numeric_cols, categorical_cols)
try:
    X = preprocessor.fit_transform(cleaned_df)
except Exception as e:
    st.error(f"Preprocessing failed: {e}")
    st.stop()

n_rows = X.shape[0]

# ============================================================
# Optimal K helper — Elbow + Silhouette across K
# ============================================================
st.markdown("---")
st.markdown("### 🔎 Find the Right Number of Segments (K)")

show_elbow = st.checkbox("Run Elbow Method & Silhouette-by-K analysis", value=False)

if show_elbow:
    max_k = min(10, max(3, n_rows // 20))
    k_range = list(range(2, max_k + 1))

    X_dense_full = X.toarray() if hasattr(X, "toarray") else X
    if n_rows > SAMPLE_LIMIT_FOR_ELBOW:
        rng = np.random.RandomState(42)
        idx = rng.choice(n_rows, size=SAMPLE_LIMIT_FOR_ELBOW, replace=False)
        X_dense = X_dense_full[idx]
        st.caption(f"Analysis computed on a random sample of {SAMPLE_LIMIT_FOR_ELBOW:,} rows for speed.")
    else:
        X_dense = X_dense_full

    with st.spinner("Testing different values of K..."):
        wcss, sil_by_k = run_elbow_analysis(hash(uploaded_file.name), X_dense, k_range)

    col1, col2 = st.columns(2)
    with col1:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(x=k_range, y=wcss, mode="lines+markers"))
        fig_elbow.update_layout(
            title="Elbow Method (WCSS vs K)",
            xaxis_title="Number of Clusters (K)",
            yaxis_title="Within-Cluster Sum of Squares"
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

    with col2:
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Bar(x=k_range, y=sil_by_k))
        fig_sil.update_layout(
            title="Silhouette Score by K",
            xaxis_title="Number of Clusters (K)",
            yaxis_title="Silhouette Score"
        )
        st.plotly_chart(fig_sil, use_container_width=True)

    best_k = k_range[int(np.argmax(sil_by_k))]
    st.info(f"💡 Highest silhouette score in this scan is at **K = {best_k}**.")

# ============================================================
# K Selection & Run
# ============================================================
default_k = 4 if n_rows >= 4 else 2
k = st.slider("Choose number of segments (K)", min_value=2, max_value=10, value=default_k)

run_clicked = st.button("🚀 Run Segmentation", type="primary")

if run_clicked:
    with st.spinner("Fitting KMeans and scoring this run..."):
        km_model, labels = run_clustering(X, k)
        sil_score, was_sampled, sil_n = safe_silhouette(X, labels)

    cluster_to_name, cluster_to_desc, tier_order = name_clusters(cleaned_df, numeric_cols, labels, k)

    result_df = cleaned_df.copy()
    result_df["Cluster"] = labels
    result_df["Segment"] = result_df["Cluster"].map(cluster_to_name)

    segment_desc_map = {cluster_to_name[c]: cluster_to_desc[c] for c in cluster_to_name}

    st.session_state["result_df"] = result_df
    st.session_state["km_model"] = km_model
    st.session_state["preprocessor"] = preprocessor
    st.session_state["X"] = X
    st.session_state["silhouette"] = sil_score
    st.session_state["sil_sampled"] = was_sampled
    st.session_state["sil_n"] = sil_n
    st.session_state["numeric_cols"] = numeric_cols
    st.session_state["categorical_cols"] = categorical_cols
    st.session_state["k_used"] = k
    st.session_state["segment_desc_map"] = segment_desc_map
    st.session_state["tier_order"] = tier_order

# ============================================================
# Results
# ============================================================
if "result_df" not in st.session_state:
    st.info("Set K above and click **Run Segmentation** to see results.")
    st.stop()

result_df = st.session_state["result_df"]
sil_score = st.session_state["silhouette"]
was_sampled = st.session_state["sil_sampled"]
sil_n = st.session_state["sil_n"]
numeric_cols = st.session_state["numeric_cols"]
categorical_cols = st.session_state["categorical_cols"]
k_used = st.session_state["k_used"]
segment_desc_map = st.session_state["segment_desc_map"]
tier_order = st.session_state["tier_order"]
X_fitted = st.session_state["X"]
km_model = st.session_state["km_model"]

color_map = {name: SEGMENT_COLORS[i % len(SEGMENT_COLORS)] for i, name in enumerate(tier_order)}

st.markdown("---")
st.markdown("### ✅ Segmentation Results")

k1, k2, k3 = st.columns(3)
with k1:
    st.metric("Customers Segmented", f"{len(result_df):,}")
with k2:
    st.metric("Segments (K)", k_used)
with k3:
    label = "Silhouette Score"
    if was_sampled:
        label += f" (sample n={sil_n:,})"
    st.metric(label, f"{sil_score:.3f}")

st.caption(
    "This silhouette score is computed fresh for this dataset and this K — "
    "it is **not** a stored/fixed value. Re-run with a different K or a different "
    "file and it will update accordingly."
)

tab1, tab2, tab3 = st.tabs(["📊 Segment Overview", "🧾 Feature Breakdown", "📥 Data & Export"])

# ------------------------------------------------------------
with tab1:
    st.markdown("#### Segment Guide")
    guide_rows = [
        {"Segment": name, "Description": segment_desc_map[name]}
        for name in tier_order if name in segment_desc_map
    ]
    st.dataframe(pd.DataFrame(guide_rows), use_container_width=True, hide_index=True)

    st.markdown("#### Cluster Visualization")
    st.caption(
        "Real clusters from this run — X marks each segment's centroid, ellipses show its spread. "
        + ("Reduced to 2D via PCA since the data has more than 2 features." if X_fitted.shape[1] > 2 else "")
    )
    cluster_fig = render_cluster_scatter(X_fitted, km_model, result_df, color_map)
    st.plotly_chart(cluster_fig, use_container_width=True)

    col1, col2 = st.columns([1, 1.3])
    seg_counts = result_df["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]

    with col1:
        fig_pie = px.pie(
            seg_counts, names="Segment", values="Count",
            color="Segment", color_discrete_map=color_map,
            hole=0.45, title="Segment Distribution"
        )
        fig_pie.update_traces(textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        fig_bar = px.bar(
            seg_counts.sort_values("Count"), x="Count", y="Segment",
            color="Segment", color_discrete_map=color_map,
            orientation="h", title="Customers per Segment", text="Count"
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    if numeric_cols:
        st.markdown("#### Segment Profile — Numeric Averages")
        profile = result_df.groupby("Segment")[numeric_cols].mean().round(2)
        profile.insert(0, "Customers", result_df["Segment"].value_counts())
        st.dataframe(profile, use_container_width=True)

# ------------------------------------------------------------
with tab2:
    if numeric_cols:
        feature = st.selectbox("Numeric feature to compare across segments", numeric_cols)
        fig_box = px.box(
            result_df, x="Segment", y=feature,
            color="Segment", color_discrete_map=color_map,
            title=f"{feature} by Segment"
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    if categorical_cols:
        cat_feature = st.selectbox("Categorical feature to compare across segments", categorical_cols)
        fig_cat = px.histogram(
            result_df, x=cat_feature,
            color="Segment", color_discrete_map=color_map,
            barmode="group", title=f"{cat_feature} by Segment"
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    if not numeric_cols and not categorical_cols:
        st.write("No feature columns available to break down.")

# ------------------------------------------------------------
with tab3:
    st.markdown("#### Filter Segmented Data")
    segments_selected = st.multiselect(
        "Filter by segment",
        options=sorted(result_df["Segment"].unique()),
        default=sorted(result_df["Segment"].unique())
    )
    filtered_df = result_df[result_df["Segment"].isin(segments_selected)]
    st.dataframe(filtered_df, use_container_width=True, height=400)

    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Segmented Data (CSV)",
        data=csv_bytes,
        file_name="segmented_customers.csv",
        mime="text/csv"
    )
