"""
Employee Attrition Prediction Dashboard
----------------------------------------
A professional Streamlit dashboard built on top of a Decision Tree
classifier for the IBM HR Analytics Employee Attrition dataset.

Run locally:
    streamlit run app.py
"""

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

# --------------------------------------------------------------------------------------
# PAGE CONFIG & STYLE
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main > div {padding-top: 1.2rem;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .metric-card {
        background: #ffffff;
        border: 1px solid #e6e6e6;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #6b7280;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #111827;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #111827;
        margin-top: 0.5rem;
        margin-bottom: 0.6rem;
        border-left: 5px solid #6366f1;
        padding-left: 0.6rem;
    }
    .badge-risk-high {
        background:#fee2e2; color:#991b1b; padding:4px 10px; border-radius:999px; font-weight:600;
    }
    .badge-risk-low {
        background:#dcfce7; color:#166534; padding:4px 10px; border-radius:999px; font-weight:600;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

DROP_COLS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
TARGET = "Attrition"


# --------------------------------------------------------------------------------------
# HELPERS (cached)
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    return pd.read_csv(file)


@st.cache_data(show_spinner=False)
def preprocess(df: pd.DataFrame):
    """Drop unneeded columns and label-encode categorical columns.
    Returns the processed dataframe and a dict of fitted encoders."""
    df = df.copy()
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders


@st.cache_resource(show_spinner=False)
def train_model(df_hash_key, X_train, y_train, max_depth, criterion, min_samples_leaf):
    model = DecisionTreeClassifier(
        random_state=42,
        max_depth=max_depth,
        criterion=criterion,
        min_samples_leaf=min_samples_leaf,
    )
    model.fit(X_train, y_train)
    return model


def get_original_categories(raw_df: pd.DataFrame):
    """Return {column: sorted unique raw values} for object columns, used to
    build friendly selectboxes in the prediction form."""
    cats = {}
    for col in raw_df.select_dtypes(include="object").columns:
        if col == TARGET:
            continue
        cats[col] = sorted(raw_df[col].dropna().unique().tolist())
    return cats


# --------------------------------------------------------------------------------------
# SIDEBAR — DATA SOURCE & MODEL CONFIG
# --------------------------------------------------------------------------------------
st.sidebar.title("⚙️ Configuration")

st.sidebar.markdown("#### 1. Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload the IBM HR Attrition CSV", type=["csv"]
)
use_sample = st.sidebar.checkbox(
    "Use bundled sample dataset (data/WA_Fn-UseC_-HR-Employee-Attrition.csv)",
    value=False,
    help="Check this if you've placed the Kaggle IBM HR Attrition CSV in the data/ folder.",
)

st.sidebar.markdown("#### 2. Model hyperparameters")
max_depth = st.sidebar.slider("Max depth", min_value=2, max_value=15, value=5)
criterion = st.sidebar.selectbox("Split criterion", ["gini", "entropy", "log_loss"], index=0)
min_samples_leaf = st.sidebar.slider("Min samples per leaf", min_value=1, max_value=20, value=1)
test_size = st.sidebar.slider("Test set size", 0.1, 0.4, 0.2, step=0.05)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Built with Streamlit • Decision Tree classifier • "
    "IBM HR Analytics Employee Attrition dataset"
)

# --------------------------------------------------------------------------------------
# LOAD DATA
# --------------------------------------------------------------------------------------
raw_df = None
if uploaded_file is not None:
    raw_df = load_data(uploaded_file)
elif use_sample:
    try:
        raw_df = load_data("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    except FileNotFoundError:
        st.sidebar.error("Sample file not found in data/. Please upload a CSV instead.")

# --------------------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------------------
st.title("📊 Employee Attrition Prediction Dashboard")
st.markdown(
    "An end-to-end **Decision Tree** classification project — EDA, preprocessing, "
    "model evaluation, and a live attrition-risk predictor."
)

if raw_df is None:
    st.info(
        "👈 Upload the **IBM HR Analytics Employee Attrition** CSV from the sidebar to get started.\n\n"
        "Dataset link: [Kaggle — IBM HR Analytics Employee Attrition & Performance]"
        "(https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)"
    )
    st.stop()

if TARGET not in raw_df.columns:
    st.error(f"The uploaded file has no '{TARGET}' column. Please upload the correct dataset.")
    st.stop()

# --------------------------------------------------------------------------------------
# PREPROCESS + TRAIN
# --------------------------------------------------------------------------------------
df, encoders = preprocess(raw_df)

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

model = train_model(
    f"{raw_df.shape}-{max_depth}-{criterion}-{min_samples_leaf}-{test_size}",
    X_train,
    y_train,
    max_depth,
    criterion,
    min_samples_leaf,
)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

# --------------------------------------------------------------------------------------
# TABS
# --------------------------------------------------------------------------------------
tab_overview, tab_eda, tab_eval, tab_importance, tab_tree, tab_predict = st.tabs(
    [
        "🏠 Overview",
        "🔍 EDA",
        "📈 Model Evaluation",
        "⭐ Feature Importance",
        "🌳 Tree Visualization",
        "🧑‍💼 Predict",
    ]
)

# ---------------------------- OVERVIEW ----------------------------
with tab_overview:
    st.markdown('<div class="section-header">Dataset Snapshot</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Total Employees</div>'
            f'<div class="metric-value">{raw_df.shape[0]}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        attr_rate = (raw_df[TARGET] == "Yes").mean() * 100 if raw_df[TARGET].dtype == object else raw_df[TARGET].mean() * 100
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Attrition Rate</div>'
            f'<div class="metric-value">{attr_rate:.1f}%</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Features</div>'
            f'<div class="metric-value">{X.shape[1]}</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Missing Values</div>'
            f'<div class="metric-value">{raw_df.isnull().sum().sum()}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Raw Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(raw_df.head(20), use_container_width=True)

    with st.expander("Column data types & non-null counts"):
        buf = io.StringIO()
        raw_df.info(buf=buf)
        st.text(buf.getvalue())

    with st.expander("Statistical summary (describe)"):
        st.dataframe(raw_df.describe(include="all").transpose(), use_container_width=True)

# ---------------------------- EDA ----------------------------
with tab_eda:
    st.markdown('<div class="section-header">Attrition Distribution</div>', unsafe_allow_html=True)
    colA, colB = st.columns(2)

    with colA:
        counts = raw_df[TARGET].value_counts().reset_index()
        counts.columns = [TARGET, "Count"]
        fig = px.bar(
            counts, x=TARGET, y="Count", color=TARGET,
            color_discrete_sequence=["#6366f1", "#ef4444"],
            title="Attrition Count",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        fig = px.pie(
            counts, names=TARGET, values="Count", hole=0.45,
            color_discrete_sequence=["#6366f1", "#ef4444"],
            title="Attrition Share",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Explore a Feature vs Attrition</div>', unsafe_allow_html=True)
    numeric_cols = raw_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in raw_df.select_dtypes(include="object").columns if c != TARGET]

    colC, colD = st.columns(2)
    with colC:
        if numeric_cols:
            num_feature = st.selectbox("Numeric feature", numeric_cols, index=0)
            fig = px.histogram(
                raw_df, x=num_feature, color=TARGET, barmode="overlay",
                color_discrete_sequence=["#6366f1", "#ef4444"],
                opacity=0.7, title=f"{num_feature} distribution by Attrition",
            )
            st.plotly_chart(fig, use_container_width=True)

    with colD:
        if categorical_cols:
            cat_feature = st.selectbox("Categorical feature", categorical_cols, index=0)
            ct = pd.crosstab(raw_df[cat_feature], raw_df[TARGET], normalize="index") * 100
            ct = ct.reset_index().melt(id_vars=cat_feature, var_name=TARGET, value_name="Percent")
            fig = px.bar(
                ct, x=cat_feature, y="Percent", color=TARGET, barmode="stack",
                color_discrete_sequence=["#6366f1", "#ef4444"],
                title=f"Attrition rate by {cat_feature}",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Correlation Heatmap (encoded features)</div>', unsafe_allow_html=True)
    corr = df.corr(numeric_only=True)
    fig = px.imshow(
        corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        aspect="auto", title="Feature Correlation Matrix",
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------- MODEL EVALUATION ----------------------------
with tab_eval:
    st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    for col, label, val in zip(
        [m1, m2, m3, m4, m5],
        ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
        [acc, prec, rec, f1, roc_auc],
    ):
        col.markdown(
            f'<div class="metric-card"><div class="metric-title">{label}</div>'
            f'<div class="metric-value">{val:.3f}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    colE, colF = st.columns(2)

    with colE:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["No", "Yes"], y=["No", "Yes"],
        )
        st.plotly_chart(fig, use_container_width=True)

    with colF:
        st.markdown('<div class="section-header">ROC Curve</div>', unsafe_allow_html=True)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {roc_auc:.3f}", line=dict(color="#6366f1", width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash", color="gray")))
        fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)

    colG, colH = st.columns(2)
    with colG:
        st.markdown('<div class="section-header">Precision-Recall Curve</div>', unsafe_allow_html=True)
        precision_arr, recall_arr, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall_arr, precision_arr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recall_arr, y=precision_arr, mode="lines", name=f"PR AUC = {pr_auc:.3f}", line=dict(color="#ef4444", width=3)))
        fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig, use_container_width=True)

    with colH:
        st.markdown('<div class="section-header">Classification Report</div>', unsafe_allow_html=True)
        report = classification_report(y_test, y_pred, target_names=["No", "Yes"], output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(3)
        st.dataframe(report_df, use_container_width=True)

# ---------------------------- FEATURE IMPORTANCE ----------------------------
with tab_importance:
    st.markdown('<div class="section-header">Top Feature Importances</div>', unsafe_allow_html=True)
    fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_n = st.slider("Number of top features to show", 5, min(30, len(fi)), 15)
    fi_top = fi.head(top_n).sort_values()
    fig = px.bar(
        x=fi_top.values, y=fi_top.index, orientation="h",
        labels={"x": "Importance", "y": "Feature"},
        color=fi_top.values, color_continuous_scale="Viridis",
        title=f"Top {top_n} Feature Importances",
    )
    fig.update_layout(coloraxis_showscale=False, height=500)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View full importance table"):
        st.dataframe(fi.reset_index().rename(columns={"index": "Feature", 0: "Importance"}), use_container_width=True)

# ---------------------------- TREE VISUALIZATION ----------------------------
with tab_tree:
    st.markdown('<div class="section-header">Decision Tree Structure</div>', unsafe_allow_html=True)
    st.caption("Rendered from the currently trained model (depends on sidebar hyperparameters).")
    max_display_depth = st.slider("Depth to display (visual only, doesn't retrain)", 1, max_depth, min(3, max_depth))

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        model,
        feature_names=X.columns,
        class_names=["No", "Yes"],
        filled=True,
        fontsize=8,
        max_depth=max_display_depth,
        ax=ax,
    )
    st.pyplot(fig)

# ---------------------------- PREDICT ----------------------------
with tab_predict:
    st.markdown('<div class="section-header">Predict Attrition Risk for a New Employee</div>', unsafe_allow_html=True)
    st.caption("Fill in employee attributes to get a live attrition-risk prediction from the trained model.")

    raw_features_df = raw_df.drop(columns=[c for c in DROP_COLS if c in raw_df.columns] + [TARGET])
    categories = get_original_categories(raw_df)

    with st.form("prediction_form"):
        input_data = {}
        cols = st.columns(3)
        for i, col_name in enumerate(raw_features_df.columns):
            target_col = cols[i % 3]
            if col_name in categories:
                input_data[col_name] = target_col.selectbox(col_name, categories[col_name])
            else:
                col_min = int(raw_features_df[col_name].min())
                col_max = int(raw_features_df[col_name].max())
                col_mean = int(raw_features_df[col_name].mean())
                input_data[col_name] = target_col.slider(col_name, col_min, col_max, col_mean)

        submitted = st.form_submit_button("🔮 Predict Attrition Risk", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([input_data])
        # Encode categorical fields with the encoders fitted during preprocessing
        for col, le in encoders.items():
            if col in input_df.columns and col != TARGET:
                input_df[col] = le.transform(input_df[col])

        # Ensure the column order matches training data
        input_df = input_df[X.columns]

        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2 = st.columns([1, 2])
        with r1:
            if pred == 1:
                st.markdown(
                    f'<span class="badge-risk-high">⚠️ High Attrition Risk</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<span class="badge-risk-low">✅ Low Attrition Risk</span>',
                    unsafe_allow_html=True,
                )
        with r2:
            st.progress(min(max(prob, 0.0), 1.0), text=f"Predicted attrition probability: {prob:.1%}")

st.markdown("---")
st.caption("Employee Attrition Dashboard • Decision Tree Classifier • Streamlit")
