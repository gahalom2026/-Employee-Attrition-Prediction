"""
Employee Attrition Prediction Dashboard
----------------------------------------
An HR decision-support dashboard built on top of a pre-trained
Decision Tree model (model.pkl) trained on the IBM HR Analytics
Employee Attrition dataset.

Run with:  streamlit run app.py
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Employee Attrition Dashboard",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# CUSTOM CSS - professional look & feel
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main { background-color: #f5f7fa; }

    .kpi-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 20px 18px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border-left: 6px solid #4f8bf9;
        text-align: center;
    }
    .kpi-card.risk { border-left-color: #e74c3c; }
    .kpi-card.rate { border-left-color: #f39c12; }
    .kpi-card.income { border-left-color: #27ae60; }
    .kpi-card.satisfaction { border-left-color: #9b59b6; }

    .kpi-label {
        font-size: 13px;
        color: #6b7280;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 30px;
        font-weight: 700;
        color: #111827;
    }

    .section-header {
        font-size: 22px;
        font-weight: 700;
        color: #1f2937;
        margin-top: 25px;
        margin-bottom: 10px;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 6px;
    }

    .reco-box {
        background: #fff7ed;
        border-left: 5px solid #f97316;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .insight-box {
        background: #eff6ff;
        border-left: 5px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 10px;
    }

    div[data-testid="stMetricValue"] { font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------------
DROP_COLS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
TARGET_COL = "Attrition"

HIGH_RISK_THRESHOLD = 0.70
MEDIUM_RISK_THRESHOLD = 0.40


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model(path="model.pkl"):
    """Load the pre-trained Decision Tree model from disk."""
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model, None
    except FileNotFoundError:
        return None, f"Could not find '{path}'. Please place the trained model file in the app folder."
    except Exception as e:
        return None, f"Error loading model: {e}"


def preprocess_for_model(df: pd.DataFrame, model):
    """
    Replicate the training-time preprocessing:
    - drop identifier / constant columns
    - drop target column if present
    - label-encode categorical columns
    Returns the encoded feature matrix aligned (where possible) to the
    model's expected feature order.
    """
    data = df.copy()

    cols_to_drop = [c for c in DROP_COLS if c in data.columns]
    data = data.drop(columns=cols_to_drop)

    if TARGET_COL in data.columns:
        data = data.drop(columns=[TARGET_COL])

    for col in data.select_dtypes(include="object").columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))

    # Align column order with what the model expects, if that info is available
    expected_features = getattr(model, "feature_names_in_", None)
    if expected_features is not None:
        missing = [c for c in expected_features if c not in data.columns]
        for c in missing:
            data[c] = 0
        data = data[list(expected_features)]

    return data


def risk_level(prob):
    if prob >= HIGH_RISK_THRESHOLD:
        return "🔴 High"
    elif prob >= MEDIUM_RISK_THRESHOLD:
        return "🟠 Medium"
    else:
        return "🟢 Low"


def kpi_card(label, value, css_class=""):
    st.markdown(
        f"""
        <div class="kpi-card {css_class}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def generate_recommendation(row):
    recos = []
    if "OverTime" in row and str(row["OverTime"]).strip().lower() == "yes":
        recos.append("Review workload; frequent overtime may be driving burnout.")
    if "JobSatisfaction" in row and pd.notnull(row["JobSatisfaction"]) and row["JobSatisfaction"] <= 2:
        recos.append("Schedule a 1:1 to address low job satisfaction and engagement.")
    if "MonthlyIncome" in row and pd.notnull(row["MonthlyIncome"]):
        recos.append("Benchmark compensation against role/market to close potential pay gaps.")
    if "YearsSinceLastPromotion" in row and pd.notnull(row["YearsSinceLastPromotion"]) and row["YearsSinceLastPromotion"] >= 3:
        recos.append("Discuss career growth path; employee may be overdue for advancement.")
    if "WorkLifeBalance" in row and pd.notnull(row["WorkLifeBalance"]) and row["WorkLifeBalance"] <= 2:
        recos.append("Assess work-life balance concerns and flexible work options.")
    if not recos:
        recos.append("Conduct a stay interview to understand retention risk drivers.")
    return recos


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 👥 HR Control Panel")
    st.markdown("Upload employee data to generate attrition risk predictions.")
    uploaded_file = st.file_uploader("Upload Employee CSV", type=["csv"])

    st.markdown("---")
    st.markdown("### ⚙️ Risk Thresholds")
    st.caption(f"High risk: probability ≥ {HIGH_RISK_THRESHOLD:.0%}")
    st.caption(f"Medium risk: probability ≥ {MEDIUM_RISK_THRESHOLD:.0%}")

    st.markdown("---")
    st.caption("Model: Decision Tree Classifier")
    st.caption("Dataset: IBM HR Analytics Employee Attrition")

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.title("👥 Employee Attrition Prediction Dashboard")
st.markdown(
    "A decision-support tool for HR teams to identify **flight-risk employees** "
    "and take proactive retention action."
)
st.markdown("---")

# ----------------------------------------------------------------------------
# MAIN LOGIC
# ----------------------------------------------------------------------------
if uploaded_file is None:
    st.info("👈 Upload an employee CSV file from the sidebar to begin.")
    st.stop()

try:
    raw_df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read the uploaded file: {e}")
    st.stop()

if raw_df.empty:
    st.warning("The uploaded file is empty.")
    st.stop()

model, model_error = load_model("model.pkl")
if model_error:
    st.error(model_error)
    st.stop()

# Preprocess & predict
try:
    X = preprocess_for_model(raw_df, model)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
except Exception as e:
    st.error(f"Prediction failed. Please check that the uploaded data matches the model's expected schema.\n\nDetails: {e}")
    st.stop()

results = raw_df.copy()
results["Employee ID"] = (
    raw_df["EmployeeNumber"] if "EmployeeNumber" in raw_df.columns else raw_df.index + 1
)
results["Prediction"] = np.where(predictions == 1, "Leave", "Stay")
results["Probability"] = probabilities.round(3)
results["Risk Level"] = results["Probability"].apply(risk_level)

high_risk_df = results[results["Risk Level"] == "🔴 High"].copy()

# ----------------------------------------------------------------------------
# KPI CARDS
# ----------------------------------------------------------------------------
st.markdown('<div class="section-header">📊 Key Performance Indicators</div>', unsafe_allow_html=True)

total_employees = len(results)
high_risk_count = len(high_risk_df)
attrition_rate = (results["Prediction"] == "Leave").mean() * 100
avg_income = raw_df["MonthlyIncome"].mean() if "MonthlyIncome" in raw_df.columns else np.nan
avg_satisfaction = raw_df["JobSatisfaction"].mean() if "JobSatisfaction" in raw_df.columns else np.nan

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    kpi_card("Total Employees", f"{total_employees:,}")
with k2:
    kpi_card("High-Risk Employees", f"{high_risk_count:,}", "risk")
with k3:
    kpi_card("Attrition Rate", f"{attrition_rate:.1f}%", "rate")
with k4:
    kpi_card("Avg Monthly Income", f"${avg_income:,.0f}" if pd.notnull(avg_income) else "N/A", "income")
with k5:
    kpi_card("Avg Job Satisfaction", f"{avg_satisfaction:.2f}/4" if pd.notnull(avg_satisfaction) else "N/A", "satisfaction")

st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# PREDICTION RESULTS TABLE
# ----------------------------------------------------------------------------
st.markdown('<div class="section-header">📋 Prediction Results</div>', unsafe_allow_html=True)

display_cols = ["Employee ID", "Prediction", "Probability", "Risk Level"]
st.dataframe(results[display_cols], use_container_width=True, height=320)

csv_download = results.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Full Prediction Results (CSV)",
    data=csv_download,
    file_name="attrition_predictions.csv",
    mime="text/csv",
)

# ----------------------------------------------------------------------------
# HIGH-RISK EMPLOYEES TABLE
# ----------------------------------------------------------------------------
st.markdown('<div class="section-header">🚨 High-Risk Employees</div>', unsafe_allow_html=True)

if high_risk_df.empty:
    st.success("No high-risk employees identified in this dataset. 🎉")
else:
    hr_cols = [c for c in ["Employee ID", "Department", "JobRole", "MonthlyIncome",
                            "JobSatisfaction", "OverTime", "Probability", "Risk Level"]
               if c in high_risk_df.columns]
    st.dataframe(high_risk_df[hr_cols].sort_values("Probability", ascending=False),
                 use_container_width=True, height=280)

    high_risk_csv = high_risk_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download High-Risk Employees (CSV)",
        data=high_risk_csv,
        file_name="high_risk_employees.csv",
        mime="text/csv",
    )

# ----------------------------------------------------------------------------
# HR RECOMMENDATIONS
# ----------------------------------------------------------------------------
st.markdown('<div class="section-header">💡 HR Recommendations</div>', unsafe_allow_html=True)

if high_risk_df.empty:
    st.markdown(
        '<div class="reco-box">No immediate retention actions required — continue routine engagement check-ins.</div>',
        unsafe_allow_html=True,
    )
else:
    top_n = min(10, len(high_risk_df))
    for _, row in high_risk_df.sort_values("Probability", ascending=False).head(top_n).iterrows():
        recos = generate_recommendation(row)
        reco_text = " ".join(recos)
        emp_label = row.get("Employee ID", "N/A")
        dept = row.get("Department", "")
        role = row.get("JobRole", "")
        st.markdown(
            f"""
            <div class="reco-box">
                <b>Employee {emp_label}</b> {f"— {dept} / {role}" if dept or role else ""}
                (Risk: {row['Probability']:.0%})<br>
                {reco_text}
            </div>
            """,
            unsafe_allow_html=True,
        )
    if len(high_risk_df) > top_n:
        st.caption(f"Showing top {top_n} of {len(high_risk_df)} high-risk employees. Download the full list above.")

# ----------------------------------------------------------------------------
# BUSINESS INSIGHTS
# ----------------------------------------------------------------------------
st.markdown('<div class="section-header">📈 Business Insights</div>', unsafe_allow_html=True)

insights = []

if "Department" in raw_df.columns:
    dept_risk = results.groupby(raw_df["Department"])["Prediction"].apply(lambda s: (s == "Leave").mean() * 100)
    if not dept_risk.empty:
        top_dept = dept_risk.idxmax()
        insights.append(f"**{top_dept}** has the highest predicted attrition rate at **{dept_risk.max():.1f}%**.")

if "OverTime" in raw_df.columns:
    ot_rate = results.groupby(raw_df["OverTime"])["Prediction"].apply(lambda s: (s == "Leave").mean() * 100)
    if "Yes" in ot_rate.index and "No" in ot_rate.index:
        diff = ot_rate["Yes"] - ot_rate["No"]
        insights.append(
            f"Employees working overtime show a **{diff:.1f} percentage point** higher predicted "
            f"attrition rate than those who don't."
        )

if "MonthlyIncome" in raw_df.columns:
    leave_income = raw_df.loc[results["Prediction"] == "Leave", "MonthlyIncome"].mean()
    stay_income = raw_df.loc[results["Prediction"] == "Stay", "MonthlyIncome"].mean()
    if pd.notnull(leave_income) and pd.notnull(stay_income):
        insights.append(
            f"Employees predicted to leave earn on average **${leave_income:,.0f}**, compared to "
            f"**${stay_income:,.0f}** for those predicted to stay."
        )

if "JobSatisfaction" in raw_df.columns:
    leave_sat = raw_df.loc[results["Prediction"] == "Leave", "JobSatisfaction"].mean()
    if pd.notnull(leave_sat):
        insights.append(f"Average job satisfaction among predicted leavers is **{leave_sat:.2f}/4**.")

if not insights:
    insights.append("Upload a dataset with Department, OverTime, MonthlyIncome, and JobSatisfaction columns for richer insights.")

for ins in insights:
    st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# CHARTS
# ----------------------------------------------------------------------------
st.markdown('<div class="section-header">📊 Visual Analysis</div>', unsafe_allow_html=True)

plot_df = raw_df.copy()
plot_df["Prediction"] = results["Prediction"].values

chart_row1_col1, chart_row1_col2 = st.columns(2)

with chart_row1_col1:
    if "Department" in plot_df.columns:
        dept_counts = plot_df.groupby(["Department", "Prediction"]).size().reset_index(name="Count")
        fig = px.bar(
            dept_counts, x="Department", y="Count", color="Prediction",
            title="Attrition by Department", barmode="group",
            color_discrete_map={"Stay": "#4f8bf9", "Leave": "#e74c3c"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Column 'Department' not found in uploaded data.")

with chart_row1_col2:
    if "JobRole" in plot_df.columns:
        role_counts = plot_df.groupby(["JobRole", "Prediction"]).size().reset_index(name="Count")
        fig = px.bar(
            role_counts, x="JobRole", y="Count", color="Prediction",
            title="Attrition by Job Role", barmode="group",
            color_discrete_map={"Stay": "#4f8bf9", "Leave": "#e74c3c"},
        )
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Column 'JobRole' not found in uploaded data.")

chart_row2_col1, chart_row2_col2 = st.columns(2)

with chart_row2_col1:
    if "OverTime" in plot_df.columns:
        ot_counts = plot_df.groupby(["OverTime", "Prediction"]).size().reset_index(name="Count")
        fig = px.bar(
            ot_counts, x="OverTime", y="Count", color="Prediction",
            title="Overtime vs Attrition", barmode="group",
            color_discrete_map={"Stay": "#4f8bf9", "Leave": "#e74c3c"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Column 'OverTime' not found in uploaded data.")

with chart_row2_col2:
    if "MonthlyIncome" in plot_df.columns:
        fig = px.box(
            plot_df, x="Prediction", y="MonthlyIncome", color="Prediction",
            title="Monthly Income vs Attrition",
            color_discrete_map={"Stay": "#4f8bf9", "Leave": "#e74c3c"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Column 'MonthlyIncome' not found in uploaded data.")

chart_row3_col1, chart_row3_col2 = st.columns(2)

with chart_row3_col1:
    if "JobSatisfaction" in plot_df.columns:
        sat_counts = plot_df.groupby(["JobSatisfaction", "Prediction"]).size().reset_index(name="Count")
        fig = px.bar(
            sat_counts, x="JobSatisfaction", y="Count", color="Prediction",
            title="Job Satisfaction vs Attrition", barmode="group",
            color_discrete_map={"Stay": "#4f8bf9", "Leave": "#e74c3c"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Column 'JobSatisfaction' not found in uploaded data.")

with chart_row3_col2:
    if hasattr(model, "feature_importances_"):
        fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
        fig = px.bar(
            fi[::-1], x=fi[::-1].values, y=fi[::-1].index, orientation="h",
            title="Top 15 Feature Importance", labels={"x": "Importance", "y": "Feature"},
            color=fi[::-1].values, color_continuous_scale="Blues",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance is not available for this model.")

st.markdown("---")
st.caption("Employee Attrition Prediction Dashboard · Built with Streamlit & Plotly")
