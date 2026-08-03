"""
Employee Attrition Analytics & Prediction Dashboard
Single-file version: data generation + model training + Streamlit UI.

Run with:
    streamlit run app.py
"""

import os
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, roc_auc_score,
)

# --------------------------------------------------------------------------------------
# DATA GENERATION / LOADING
# --------------------------------------------------------------------------------------
RAW_CSV_NAME = "WA_Fn-UseC_-HR-Employee-Attrition.csv"

DEPARTMENTS = ["Sales", "Research & Development", "Human Resources"]
JOB_ROLES = {
    "Sales": ["Sales Executive", "Sales Representative", "Manager"],
    "Research & Development": [
        "Research Scientist", "Laboratory Technician", "Manufacturing Director",
        "Healthcare Representative", "Research Director", "Manager",
    ],
    "Human Resources": ["Human Resources", "Manager"],
}
EDUCATION_FIELDS = ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"]
BUSINESS_TRAVEL = ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
MARITAL_STATUS = ["Single", "Married", "Divorced"]
GENDER = ["Male", "Female"]
OVERTIME = ["Yes", "No"]


def _generate_synthetic(n=1470, seed=42):
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 61, n)
    gender = rng.choice(GENDER, n, p=[0.6, 0.4])
    department = rng.choice(DEPARTMENTS, n, p=[0.31, 0.65, 0.04])
    job_role = np.array([rng.choice(JOB_ROLES[d]) for d in department])
    education = rng.integers(1, 6, n)
    education_field = rng.choice(EDUCATION_FIELDS, n, p=[0.41, 0.16, 0.11, 0.15, 0.06, 0.11])
    business_travel = rng.choice(BUSINESS_TRAVEL, n, p=[0.10, 0.71, 0.19])
    marital_status = rng.choice(MARITAL_STATUS, n, p=[0.32, 0.46, 0.22])
    distance_from_home = rng.integers(1, 30, n)
    overtime = rng.choice(OVERTIME, n, p=[0.28, 0.72])

    job_level = rng.integers(1, 6, n)
    monthly_income = (job_level * 2200 + rng.normal(0, 900, n) + (age * 25)).clip(1009, 20000).astype(int)
    daily_rate = rng.integers(100, 1500, n)
    hourly_rate = rng.integers(30, 101, n)
    monthly_rate = rng.integers(2000, 27000, n)
    percent_salary_hike = rng.integers(11, 26, n)
    stock_option_level = rng.integers(0, 4, n)
    num_companies_worked = rng.integers(0, 10, n)

    total_working_years = np.clip((age - 22) + rng.integers(-2, 3, n), 0, 40)
    years_at_company = np.clip((total_working_years * rng.uniform(0.2, 0.9, n)).astype(int), 0, 40)
    years_in_current_role = np.clip((years_at_company * rng.uniform(0.1, 0.8, n)).astype(int), 0, years_at_company)
    years_since_last_promotion = np.clip((years_at_company * rng.uniform(0.0, 0.6, n)).astype(int), 0, years_at_company)
    years_with_curr_manager = np.clip((years_at_company * rng.uniform(0.1, 0.8, n)).astype(int), 0, years_at_company)
    training_times_last_year = rng.integers(0, 7, n)

    job_satisfaction = rng.integers(1, 5, n)
    environment_satisfaction = rng.integers(1, 5, n)
    relationship_satisfaction = rng.integers(1, 5, n)
    work_life_balance = rng.integers(1, 5, n)
    job_involvement = rng.integers(1, 5, n)
    performance_rating = rng.choice([3, 4], n, p=[0.85, 0.15])

    # Attrition probability driven by realistic HR risk factors (stronger,
    # cleaner signal so the Decision Tree achieves presentable performance)
    logit = (
        -3.4
        + 2.1 * (overtime == "Yes").astype(float)
        + 1.4 * (4 - job_satisfaction) / 3
        + 1.1 * (4 - work_life_balance) / 3
        + 1.0 * (business_travel == "Travel_Frequently").astype(float)
        + 1.1 * (monthly_income < np.percentile(monthly_income, 25)).astype(float)
        + 1.0 * (years_at_company < 2).astype(float)
        + 0.6 * (marital_status == "Single").astype(float)
        + 0.6 * (num_companies_worked > 5).astype(float)
        - 0.9 * (stock_option_level > 0).astype(float)
        - 0.5 * (job_level >= 4).astype(float)
        + rng.normal(0, 0.3, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    attrition = rng.random(n) < prob
    attrition_lbl = np.where(attrition, "Yes", "No")

    df = pd.DataFrame({
        "Age": age,
        "Attrition": attrition_lbl,
        "BusinessTravel": business_travel,
        "DailyRate": daily_rate,
        "Department": department,
        "DistanceFromHome": distance_from_home,
        "Education": education,
        "EducationField": education_field,
        "EnvironmentSatisfaction": environment_satisfaction,
        "Gender": gender,
        "HourlyRate": hourly_rate,
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "JobRole": job_role,
        "JobSatisfaction": job_satisfaction,
        "MaritalStatus": marital_status,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": monthly_rate,
        "NumCompaniesWorked": num_companies_worked,
        "OverTime": overtime,
        "PercentSalaryHike": percent_salary_hike,
        "PerformanceRating": performance_rating,
        "RelationshipSatisfaction": relationship_satisfaction,
        "StockOptionLevel": stock_option_level,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training_times_last_year,
        "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_current_role,
        "YearsSinceLastPromotion": years_since_last_promotion,
        "YearsWithCurrManager": years_with_curr_manager,
    })
    df.insert(0, "EmployeeNumber", np.arange(1, n + 1))
    return df


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bins = [17, 25, 35, 45, 55, 65]
    labels = ["18-25", "26-35", "36-45", "46-55", "56-60"]
    df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels, right=True, include_lowest=True)

    try:
        df["SalaryBand"] = pd.qcut(
            df["MonthlyIncome"], q=4, labels=["Low", "Medium", "High", "Very High"]
        )
    except ValueError:
        df["SalaryBand"] = "Medium"
    return df


def load_raw_dataframe() -> pd.DataFrame:
    """Load real dataset if present, otherwise generate a synthetic one."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, RAW_CSV_NAME)
    if os.path.exists(candidate):
        df = pd.read_csv(candidate)
    else:
        df = _generate_synthetic()
    df = _add_derived_columns(df)
    return df


# --------------------------------------------------------------------------------------
# MODEL TRAINING & PREDICTION HELPERS
# --------------------------------------------------------------------------------------
DROP_COLS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours", "AgeGroup", "SalaryBand"]
MODEL_FEATURES = [
    "Age", "BusinessTravel", "DailyRate", "Department", "DistanceFromHome",
    "Education", "EducationField", "EnvironmentSatisfaction", "Gender",
    "HourlyRate", "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction",
    "MaritalStatus", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
    "OverTime", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
]


def build_model_bundle(raw_df: pd.DataFrame):
    """Train the Decision Tree model on the (label-encoded) raw dataframe and
    return everything the dashboard needs: the fitted model, encoders,
    train/test data, predictions, and evaluation metrics."""
    df = raw_df.copy()
    drop_cols = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=drop_cols)

    encoders = {}
    enc_df = df.copy()
    for col in enc_df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        enc_df[col] = le.fit_transform(enc_df[col])
        encoders[col] = le

    X = enc_df.drop(columns=["Attrition"])
    y = enc_df["Attrition"]  # 0 = No, 1 = Yes (alphabetical LabelEncoder order)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(random_state=42, max_depth=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    feature_importance = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    return {
        "model": model,
        "encoders": encoders,
        "feature_columns": list(X.columns),
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_pred": y_pred, "y_prob": y_prob,
        "metrics": metrics,
        "confusion_matrix": cm,
        "roc": {"fpr": fpr, "tpr": tpr},
        "feature_importance": feature_importance,
    }


def encode_single_record(record: dict, encoders: dict, feature_columns: list) -> pd.DataFrame:
    """Turn a raw-value dict from the prediction form into a model-ready row."""
    row = {}
    for col in feature_columns:
        val = record.get(col)
        if col in encoders:
            le = encoders[col]
            if val in le.classes_:
                val = int(le.transform([val])[0])
            else:
                val = 0
        row[col] = val
    return pd.DataFrame([row])[feature_columns]


def risk_level_from_probability(prob: float) -> str:
    if prob < 0.35:
        return "Low"
    if prob < 0.65:
        return "Medium"
    return "High"


# --------------------------------------------------------------------------------------
# STREAMLIT DASHBOARD
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Employee Attrition Analytics & Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#0C2B4B"
BLUE = "#1F6FEB"
BLUE_LIGHT = "#4C8DF0"
GRAY_DARK = "#2C2C2E"
GRAY_MED = "#6B7280"
GRAY_LIGHT = "#F4F6F9"
WHITE = "#FFFFFF"
GREEN = "#1F9D6C"
RED = "#E5484D"
AMBER = "#F2A93B"

PLOTLY_TEMPLATE = "plotly_white"
CHART_COLORWAY = [BLUE, NAVY, "#7FB2F0", GRAY_MED, GREEN, AMBER, RED, "#0C447C"]

CUSTOM_CSS = f"""
<style>
    .main {{ background-color: {GRAY_LIGHT}; }}
    #MainMenu, footer {{ visibility: hidden; }}

    h1, h2, h3 {{ color: {NAVY}; font-family: 'Segoe UI', sans-serif; }}

    .app-header {{
        background: linear-gradient(90deg, {NAVY} 0%, {BLUE} 100%);
        padding: 22px 28px;
        border-radius: 14px;
        margin-bottom: 18px;
        color: white;
    }}
    .app-header h1 {{ color: white; margin: 0; font-size: 28px; }}
    .app-header p {{ color: #DCE7FA; margin: 4px 0 0 0; font-size: 14px; }}

    div[data-testid="stMetric"] {{
        background-color: {WHITE};
        border: 1px solid #E6E9EF;
        border-radius: 12px;
        padding: 14px 16px 8px 16px;
        box-shadow: 0 2px 6px rgba(12,43,75,0.06);
    }}
    div[data-testid="stMetricLabel"] {{ color: {GRAY_MED}; font-weight: 600; }}
    div[data-testid="stMetricValue"] {{ color: {NAVY}; }}

    .section-card {{
        background-color: {WHITE};
        border: 1px solid #E6E9EF;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(12,43,75,0.05);
    }}
    .section-title {{
        font-size: 18px; font-weight: 700; color: {NAVY};
        margin-bottom: 2px;
    }}
    .section-sub {{ color: {GRAY_MED}; font-size: 13px; margin-bottom: 14px; }}

    .insight-pill {{
        background: #EAF1FE; color: {NAVY}; border-radius: 10px;
        padding: 10px 14px; margin-bottom: 10px; font-size: 14px; border-left: 4px solid {BLUE};
    }}
    .risk-high {{ color: {RED}; font-weight: 700; }}
    .risk-medium {{ color: {AMBER}; font-weight: 700; }}
    .risk-low {{ color: {GREEN}; font-weight: 700; }}

    section[data-testid="stSidebar"] {{ background-color: {NAVY}; }}
    section[data-testid="stSidebar"] * {{ color: #E9EEF7; }}
    section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {{ background-color: {BLUE}; }}

    .stTabs [data-baseweb="tab"] {{ font-weight: 600; color: {GRAY_DARK}; }}
    .stTabs [aria-selected="true"] {{ color: {BLUE} !important; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def section_header(icon: str, title: str, subtitle: str = ""):
    st.markdown(
        f"""<div class="section-card"><div class="section-title">{icon} {title}</div>
        <div class="section-sub">{subtitle}</div>""",
        unsafe_allow_html=True,
    )


def section_footer():
    st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# DATA & MODEL (cached)
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def get_data():
    return load_raw_dataframe()


@st.cache_resource(show_spinner=True)
def get_model_bundle(df: pd.DataFrame):
    return build_model_bundle(df)


df = get_data()
bundle = get_model_bundle(df)

# --------------------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------------------
st.markdown(
    """<div class="app-header">
        <h1>📊 Employee Attrition Analytics & Prediction Dashboard</h1>
        <p>AI-powered HR business intelligence · Decision Tree classification · Executive-ready insights</p>
    </div>""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------------------------------------------
st.sidebar.markdown("### 🔍 Filters")
st.sidebar.caption("Refine the analytics below by any combination of filters.")


def multiselect_filter(label, col):
    options = sorted(df[col].dropna().astype(str).unique().tolist())
    return st.sidebar.multiselect(label, options, default=options)


f_department = multiselect_filter("🏢 Department", "Department")
f_role = multiselect_filter("💼 Job role", "JobRole")
f_gender = multiselect_filter("🚻 Gender", "Gender")
f_agegroup = multiselect_filter("🎂 Age group", "AgeGroup")
f_education = multiselect_filter("🎓 Education level", "Education")
f_marital = multiselect_filter("💍 Marital status", "MaritalStatus")
f_travel = multiselect_filter("✈️ Business travel", "BusinessTravel")
f_overtime = multiselect_filter("⏱️ Overtime", "OverTime")
f_wlb = multiselect_filter("⚖️ Work-life balance", "WorkLifeBalance")

st.sidebar.markdown("---")
if st.sidebar.button("↺ Reset filters"):
    st.rerun()

mask = (
    df["Department"].astype(str).isin(f_department)
    & df["JobRole"].astype(str).isin(f_role)
    & df["Gender"].astype(str).isin(f_gender)
    & df["AgeGroup"].astype(str).isin(f_agegroup)
    & df["Education"].astype(str).isin(f_education)
    & df["MaritalStatus"].astype(str).isin(f_marital)
    & df["BusinessTravel"].astype(str).isin(f_travel)
    & df["OverTime"].astype(str).isin(f_overtime)
    & df["WorkLifeBalance"].astype(str).isin(f_wlb)
)
fdf = df[mask].copy()

if fdf.empty:
    st.warning("No employees match the current filters. Adjust filters in the sidebar to see results.")
    st.stop()

# --------------------------------------------------------------------------------------
# EXECUTIVE KPI CARDS
# --------------------------------------------------------------------------------------
total_emp = len(fdf)
left_emp = int((fdf["Attrition"] == "Yes").sum())
active_emp = total_emp - left_emp
attrition_rate = (left_emp / total_emp * 100) if total_emp else 0
avg_income = fdf["MonthlyIncome"].mean()
avg_job_sat = fdf["JobSatisfaction"].mean()

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("👥 Total employees", f"{total_emp:,}")
k2.metric("✅ Active employees", f"{active_emp:,}")
k3.metric("🚪 Employees left", f"{left_emp:,}")
k4.metric("📉 Attrition rate", f"{attrition_rate:.1f}%")
k5.metric("💰 Avg. monthly income", f"${avg_income:,.0f}")
k6.metric("😊 Avg. job satisfaction", f"{avg_job_sat:.2f} / 4")

st.write("")

# --------------------------------------------------------------------------------------
# TABS
# --------------------------------------------------------------------------------------
tabs = st.tabs([
    "📈 Attrition analytics", "💵 Salary analysis", "🏆 Performance & experience",
    "😊 Satisfaction", "⏱️ Overtime & travel", "🔗 Correlation",
    "🤖 Predict attrition", "🧪 Model performance", "💡 Business insights", "⬇️ Export",
])

ATTR_COLOR_MAP = {"Yes": RED, "No": BLUE}


def attrition_rate_bar(data, group_col, title):
    g = data.groupby(group_col, observed=True)["Attrition"].apply(
        lambda s: (s == "Yes").mean() * 100
    ).reset_index(name="AttritionRate")
    g = g.sort_values("AttritionRate", ascending=False)
    fig = px.bar(
        g, x=group_col, y="AttritionRate", text="AttritionRate",
        title=title, template=PLOTLY_TEMPLATE, color_discrete_sequence=[BLUE],
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(yaxis_title="Attrition rate (%)", xaxis_title="", showlegend=False, height=380)
    return fig


# ---------------- TAB 1: ATTRITION ANALYTICS ----------------
with tabs[0]:
    section_header("📈", "Attrition analytics", "Where attrition concentrates across the workforce")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(attrition_rate_bar(fdf, "Department", "Attrition rate by department"), use_container_width=True)
    with c2:
        st.plotly_chart(attrition_rate_bar(fdf, "JobRole", "Attrition rate by job role"), use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        g = fdf.groupby(["Gender", "Attrition"]).size().reset_index(name="Count")
        fig = px.bar(g, x="Gender", y="Count", color="Attrition", barmode="group",
                     title="Attrition by gender", template=PLOTLY_TEMPLATE,
                     color_discrete_map=ATTR_COLOR_MAP, height=380)
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        st.plotly_chart(attrition_rate_bar(fdf, "AgeGroup", "Attrition rate by age group"), use_container_width=True)
    st.plotly_chart(attrition_rate_bar(fdf, "MaritalStatus", "Attrition rate by marital status"), use_container_width=True)
    section_footer()

# ---------------- TAB 2: SALARY ANALYSIS ----------------
with tabs[1]:
    section_header("💵", "Salary analysis", "Compensation distribution and its link to attrition")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(fdf, x="MonthlyIncome", color="Attrition", nbins=30, barmode="overlay",
                            title="Monthly income distribution", template=PLOTLY_TEMPLATE,
                            color_discrete_map=ATTR_COLOR_MAP, height=380)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        g = fdf.groupby(["SalaryBand", "Attrition"]).size().reset_index(name="Count")
        fig = px.bar(g, x="SalaryBand", y="Count", color="Attrition", barmode="group",
                     title="Salary band vs. attrition", template=PLOTLY_TEMPLATE,
                     color_discrete_map=ATTR_COLOR_MAP, height=380,
                     category_orders={"SalaryBand": ["Low", "Medium", "High", "Very High"]})
        st.plotly_chart(fig, use_container_width=True)
    g = fdf.groupby("Department", observed=True)["MonthlyIncome"].mean().reset_index().sort_values("MonthlyIncome", ascending=False)
    fig = px.bar(g, x="Department", y="MonthlyIncome", title="Average salary by department",
                 template=PLOTLY_TEMPLATE, color_discrete_sequence=[NAVY], height=380)
    fig.update_layout(yaxis_title="Average monthly income ($)")
    st.plotly_chart(fig, use_container_width=True)
    section_footer()

# ---------------- TAB 3: PERFORMANCE & EXPERIENCE ----------------
with tabs[2]:
    section_header("🏆", "Employee performance & experience", "Tenure, promotion cadence and training investment")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(fdf, x="YearsAtCompany", color="Attrition", nbins=20, barmode="overlay",
                            title="Years at company", template=PLOTLY_TEMPLATE,
                            color_discrete_map=ATTR_COLOR_MAP, height=360)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(fdf, x="YearsSinceLastPromotion", color="Attrition", nbins=15, barmode="overlay",
                            title="Years since last promotion", template=PLOTLY_TEMPLATE,
                            color_discrete_map=ATTR_COLOR_MAP, height=360)
        st.plotly_chart(fig, use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        fig = px.histogram(fdf, x="TotalWorkingYears", color="Attrition", nbins=20, barmode="overlay",
                            title="Total working years", template=PLOTLY_TEMPLATE,
                            color_discrete_map=ATTR_COLOR_MAP, height=360)
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        g = fdf.groupby(["TrainingTimesLastYear", "Attrition"]).size().reset_index(name="Count")
        fig = px.bar(g, x="TrainingTimesLastYear", y="Count", color="Attrition", barmode="group",
                     title="Training sessions last year", template=PLOTLY_TEMPLATE,
                     color_discrete_map=ATTR_COLOR_MAP, height=360)
        st.plotly_chart(fig, use_container_width=True)
    section_footer()

# ---------------- TAB 4: SATISFACTION ----------------
with tabs[3]:
    section_header("😊", "Employee satisfaction analysis", "Self-reported sentiment across five dimensions")
    sat_cols = ["JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction", "WorkLifeBalance", "JobInvolvement"]
    avgs = fdf[sat_cols].mean().reset_index()
    avgs.columns = ["Dimension", "AverageScore"]
    fig = px.bar(avgs, x="Dimension", y="AverageScore", title="Average satisfaction scores (scale 1–4)",
                 template=PLOTLY_TEMPLATE, color_discrete_sequence=[BLUE], height=360, range_y=[0, 4])
    st.plotly_chart(fig, use_container_width=True)
    cols = st.columns(len(sat_cols))
    for i, col in enumerate(sat_cols):
        g = fdf.groupby([col, "Attrition"]).size().reset_index(name="Count")
        fig = px.bar(g, x=col, y="Count", color="Attrition", barmode="group", title=col,
                     template=PLOTLY_TEMPLATE, color_discrete_map=ATTR_COLOR_MAP, height=300)
        fig.update_layout(showlegend=(i == 0), margin=dict(t=40, b=10))
        cols[i].plotly_chart(fig, use_container_width=True)
    section_footer()

# ---------------- TAB 5: OVERTIME & TRAVEL ----------------
with tabs[4]:
    section_header("⏱️", "Overtime & business travel analysis", "Two of the strongest attrition risk factors")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(attrition_rate_bar(fdf, "OverTime", "Attrition rate: overtime vs. no overtime"), use_container_width=True)
    with c2:
        st.plotly_chart(attrition_rate_bar(fdf, "BusinessTravel", "Attrition rate by business travel frequency"), use_container_width=True)
    section_footer()

# ---------------- TAB 6: CORRELATION ----------------
with tabs[5]:
    section_header("🔗", "Correlation analysis", "Relationships between numeric features and feature importance")
    numeric_df = fdf.select_dtypes(include=[np.number]).drop(columns=["EmployeeNumber"], errors="ignore")
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                     title="Correlation heatmap (numeric features)", template=PLOTLY_TEMPLATE, height=650)
    st.plotly_chart(fig, use_container_width=True)

    fi = bundle["feature_importance"].head(15).sort_values()
    fig2 = px.bar(fi, x=fi.values, y=fi.index, orientation="h",
                  title="Top 15 feature importances (Decision Tree)",
                  template=PLOTLY_TEMPLATE, color_discrete_sequence=[NAVY], height=500)
    fig2.update_layout(xaxis_title="Importance", yaxis_title="")
    st.plotly_chart(fig2, use_container_width=True)
    section_footer()

# ---------------- TAB 7: PREDICTION ----------------
with tabs[6]:
    section_header("🤖", "Machine learning prediction", "Score an individual employee's attrition risk")
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            p_age = st.slider("Age", 18, 60, 32)
            p_gender = st.selectbox("Gender", sorted(df["Gender"].unique()))
            p_department = st.selectbox("Department", sorted(df["Department"].unique()))
            p_role = st.selectbox("Job role", sorted(df["JobRole"].unique()))
        with c2:
            p_income = st.number_input("Monthly income ($)", 1000, 25000, 5000, step=100)
            p_overtime = st.selectbox("Overtime", sorted(df["OverTime"].unique()))
            p_years = st.slider("Years at company", 0, 40, 5)
            p_distance = st.slider("Distance from home (km)", 1, 30, 8)
        with c3:
            p_jobsat = st.select_slider("Job satisfaction", options=[1, 2, 3, 4], value=3)
            p_wlb = st.select_slider("Work-life balance", options=[1, 2, 3, 4], value=3)
            p_travel = st.selectbox("Business travel", sorted(df["BusinessTravel"].unique()))
            p_education = st.select_slider("Education level", options=[1, 2, 3, 4, 5], value=3)
            p_marital = st.selectbox("Marital status", sorted(df["MaritalStatus"].unique()))
        submitted = st.form_submit_button("🔮 Predict attrition risk", use_container_width=True)

    if submitted:
        defaults = {
            "DailyRate": int(df["DailyRate"].median()),
            "EducationField": df["EducationField"].mode()[0],
            "EnvironmentSatisfaction": 3,
            "HourlyRate": int(df["HourlyRate"].median()),
            "JobInvolvement": 3,
            "JobLevel": 2,
            "MonthlyRate": int(df["MonthlyRate"].median()),
            "NumCompaniesWorked": int(df["NumCompaniesWorked"].median()),
            "PercentSalaryHike": int(df["PercentSalaryHike"].median()),
            "PerformanceRating": 3,
            "RelationshipSatisfaction": 3,
            "StockOptionLevel": 0,
            "TotalWorkingYears": max(p_years, int(df["TotalWorkingYears"].median())),
            "TrainingTimesLastYear": 2,
            "YearsInCurrentRole": min(p_years, 4),
            "YearsSinceLastPromotion": min(p_years, 2),
            "YearsWithCurrManager": min(p_years, 4),
        }
        record = {
            "Age": p_age, "Gender": p_gender, "Department": p_department, "JobRole": p_role,
            "MonthlyIncome": p_income, "OverTime": p_overtime, "YearsAtCompany": p_years,
            "DistanceFromHome": p_distance, "JobSatisfaction": p_jobsat, "WorkLifeBalance": p_wlb,
            "BusinessTravel": p_travel, "Education": p_education, "MaritalStatus": p_marital,
            **defaults,
        }
        X_row = encode_single_record(record, bundle["encoders"], bundle["feature_columns"])
        model = bundle["model"]
        prob_yes = model.predict_proba(X_row)[0][1]
        pred_label = "Yes — likely to leave" if prob_yes >= 0.5 else "No — likely to stay"
        risk = risk_level_from_probability(prob_yes)
        confidence = max(prob_yes, 1 - prob_yes)

        st.markdown("#### Prediction result")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Prediction", pred_label.split(" — ")[0])
        r2.metric("Probability of leaving", f"{prob_yes*100:.1f}%")
        risk_class = {"Low": "risk-low", "Medium": "risk-medium", "High": "risk-high"}[risk]
        r3.markdown(f"**Risk level**<br><span class='{risk_class}' style='font-size:22px'>{risk}</span>", unsafe_allow_html=True)
        r4.metric("Model confidence", f"{confidence*100:.1f}%")

        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=prob_yes * 100,
            title={"text": "Attrition risk probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": NAVY},
                "steps": [
                    {"range": [0, 35], "color": "#D7F0E3"},
                    {"range": [35, 65], "color": "#FCE9C8"},
                    {"range": [65, 100], "color": "#FBD7D7"},
                ],
            },
        ))
        fig.update_layout(height=300, template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"**{pred_label}.** {pred_label.split(' — ')[1].capitalize()} based on the profile provided. "
                f"Risk level is **{risk}** with model confidence of **{confidence*100:.1f}%**.")
    section_footer()

# ---------------- TAB 8: MODEL PERFORMANCE ----------------
with tabs[7]:
    section_header("🧪", "Model performance", "Decision Tree classifier — held-out 20% test set")
    m = bundle["metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{m['accuracy']*100:.1f}%")
    c2.metric("Precision", f"{m['precision']*100:.1f}%")
    c3.metric("Recall", f"{m['recall']*100:.1f}%")
    c4.metric("F1 score", f"{m['f1']*100:.1f}%")
    c5.metric("ROC-AUC", f"{m['roc_auc']*100:.1f}%")

    c1, c2 = st.columns(2)
    with c1:
        cm = bundle["confusion_matrix"]
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                         labels=dict(x="Predicted", y="Actual", color="Count"),
                         x=["No", "Yes"], y=["No", "Yes"],
                         title="Confusion matrix", template=PLOTLY_TEMPLATE, height=420)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fpr, tpr = bundle["roc"]["fpr"], bundle["roc"]["tpr"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={m['roc_auc']:.3f})", line=dict(color=BLUE, width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random baseline", line=dict(color=GRAY_MED, dash="dash")))
        fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate",
                           template=PLOTLY_TEMPLATE, height=420)
        st.plotly_chart(fig, use_container_width=True)
    section_footer()

# ---------------- TAB 9: BUSINESS INSIGHTS ----------------
with tabs[8]:
    section_header("💡", "Business insights", "Auto-generated findings from the current filtered view")

    dept_rate = fdf.groupby("Department", observed=True)["Attrition"].apply(lambda s: (s == "Yes").mean() * 100)
    top_dept = dept_rate.idxmax()
    top_dept_rate = dept_rate.max()

    ot_yes_rate = (fdf.loc[fdf["OverTime"] == "Yes", "Attrition"] == "Yes").mean() * 100 if (fdf["OverTime"] == "Yes").any() else 0
    ot_no_rate = (fdf.loc[fdf["OverTime"] == "No", "Attrition"] == "Yes").mean() * 100 if (fdf["OverTime"] == "No").any() else 0

    leavers_income = fdf.loc[fdf["Attrition"] == "Yes", "MonthlyIncome"].mean()
    stayers_income = fdf.loc[fdf["Attrition"] == "No", "MonthlyIncome"].mean()

    leavers_sat = fdf.loc[fdf["Attrition"] == "Yes", "JobSatisfaction"].mean()
    stayers_sat = fdf.loc[fdf["Attrition"] == "No", "JobSatisfaction"].mean()

    role_rate = fdf.groupby("JobRole", observed=True)["Attrition"].apply(lambda s: (s == "Yes").mean() * 100)
    top_role = role_rate.idxmax() if not role_rate.empty else "N/A"

    insights = [
        f"🏢 **{top_dept}** has the highest attrition rate among departments at **{top_dept_rate:.1f}%**.",
        f"⏱️ Employees working overtime leave at **{ot_yes_rate:.1f}%** versus **{ot_no_rate:.1f}%** for those who don't — overtime is one of the strongest attrition drivers.",
        f"💰 Employees who left earn **${leavers_income:,.0f}/month** on average, versus **${stayers_income:,.0f}/month** for those who stayed — compensation gaps correlate with attrition.",
        f"😊 Average job satisfaction is **{leavers_sat:.2f}/4** among leavers versus **{stayers_sat:.2f}/4** among stayers, confirming satisfaction as a leading indicator.",
        f"⚠️ **{top_role}** is the highest-risk job role, with an attrition rate of **{role_rate.max():.1f}%**.",
    ]
    for i in insights:
        st.markdown(f"<div class='insight-pill'>{i}</div>", unsafe_allow_html=True)

    st.markdown("#### 📌 Recommendations for HR")
    st.markdown(
        """
- Prioritize workload and staffing reviews in departments and roles with above-average attrition.
- Cap or better compensate mandatory overtime; monitor overtime hours as a leading risk signal.
- Review compensation bands for at-risk salary tiers, especially early-tenure employees.
- Strengthen manager check-ins and career growth conversations where satisfaction scores are low.
- Use the prediction tool proactively for retention conversations, not just exit analysis.
"""
    )
    section_footer()

# ---------------- TAB 10: EXPORT ----------------
with tabs[9]:
    section_header("⬇️", "Download center", "Export filtered data and reports")
    c1, c2 = st.columns(2)
    with c1:
        csv_bytes = fdf.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export filtered data (CSV)", data=csv_bytes,
                            file_name="attrition_dashboard_data.csv", mime="text/csv", use_container_width=True)
    with c2:
        report_lines = [
            "EMPLOYEE ATTRITION ANALYTICS - SUMMARY REPORT",
            "=" * 50,
            f"Total employees analyzed: {total_emp}",
            f"Employees left: {left_emp}",
            f"Attrition rate: {attrition_rate:.1f}%",
            f"Average monthly income: ${avg_income:,.0f}",
            f"Average job satisfaction: {avg_job_sat:.2f}/4",
            "",
            "MODEL PERFORMANCE",
            "-" * 50,
            f"Accuracy: {m['accuracy']*100:.1f}%",
            f"Precision: {m['precision']*100:.1f}%",
            f"Recall: {m['recall']*100:.1f}%",
            f"F1 Score: {m['f1']*100:.1f}%",
            f"ROC-AUC: {m['roc_auc']*100:.1f}%",
            "",
            "KEY INSIGHTS",
            "-" * 50,
        ] + [i.replace("**", "").replace("🏢 ", "- ").replace("⏱️ ", "- ").replace("💰 ", "- ").replace("😊 ", "- ").replace("⚠️ ", "- ") for i in insights]
        report_text = "\n".join(report_lines)
        st.download_button("📄 Download prediction/summary report (TXT)", data=report_text.encode("utf-8"),
                            file_name="attrition_summary_report.txt", mime="text/plain", use_container_width=True)
    st.caption("PDF export can be added with reportlab/fpdf2 in a production deployment; CSV and TXT are provided here for portability.")
    section_footer()

# --------------------------------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------------------------------
st.markdown(
    f"""
    <div style='text-align:center; padding: 24px 0 8px 0; color:{GRAY_MED}; font-size:13px;'>
        Built by <b>Your Name</b> ·
        <a href="https://github.com/yourusername" target="_blank">GitHub</a> ·
        <a href="https://linkedin.com/in/yourusername" target="_blank">LinkedIn</a><br>
        Employee Attrition Analytics & Prediction Dashboard · Decision Tree Model · Streamlit + Plotly
    </div>
    """,
    unsafe_allow_html=True,
)
