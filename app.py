"""
app.py
Streamlit web app for the Student Performance Prediction project.

Loads the saved model pipeline and lets a user enter a student's details to
get a predicted final grade (G3, scale 0-20).

SAVE THIS FILE AT: app/app.py  (inside the "app" folder created in Step 1)

Run from the project root:
    streamlit run app/app.py
"""

import sys
from pathlib import Path

import streamlit as st
# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown(
    '<div class="main-title">🎓 Student Performance Predictor</div>',
    unsafe_allow_html=True
)
st.markdown("""
<style>

/* Whole application background */
.stApp {
    background-color: #ffffff !important;
    color: #222222 !important;
}
/* Light page and content areas */
[data-testid="stAppViewContainer"] {
    background-color: #ffffff !important;
}

[data-testid="stHeader"] {
    background-color: #ffffff !important;
}

[data-testid="stSidebar"] {
    background-color: #ffffff !important;
}

[data-testid="stToolbar"] {
    background-color: #ffffff !important;
}
}

div.main-title {
    text-align: center !important;
    font-size: 40px !important;
    font-weight: 700 !important;
    margin-top: 10px !important;
    margin-bottom: 10px !important;
    color: #1e293b !important;
    line-height: 1.2 !important;
}

@media (max-width: 768px) {
    div.main-title {
        font-size: 30px !important;
        text-align: center !important;
        color: #1e293b !important;
    }
}
}

/* =========================================================
   FINAL READABILITY FIX
   ========================================================= */

/* Page */
.stApp {
    background-color: #f8fafc !important;
    color: #222222 !important;
}

/* All normal text */
.stApp p,
.stApp span,
.stApp label,
.stApp div {
    color: #222222;
}

/* Section headings */
.section-title {
    color: #1e293b !important;
}

/* Streamlit widget labels */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span,
[data-testid="stWidgetLabel"] label {
    color: #334155 !important;
}

/* Select boxes */
[data-baseweb="select"] {
    background-color: #ffffff !important;
}

[data-baseweb="select"] * {
    color: #222222 !important;
}

/* Select dropdown menu */
[data-baseweb="popover"] {
    background-color: #ffffff !important;
}

[data-baseweb="popover"] * {
    color: #222222 !important;
}

/* Sliders */
[data-testid="stSlider"] label,
[data-testid="stSlider"] p,
[data-testid="stSlider"] span {
    color: #334155 !important;
}

/* Slider value */
[data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: #334155 !important;
}

/* Predict button */
.stButton > button {
    background-color: #dbeafe !important;
    color: #1e3a8a !important;
    border: 1px solid #93c5fd !important;
}

/* Predict button text */
.stButton > button p,
.stButton > button span {
    color: #1e3a8a !important;
}

/* Result */
.result-box {
    background-color: #ffffff !important;
    color: #222222 !important;
}

.result-title {
    color: #334155 !important;
}

.result-value {
    color: #1e3a8a !important;
    font-weight: 700 !important;
}

/* Mobile */
@media (max-width: 768px) {

    .main-title {
        font-size: 30px !important;
        color: #1e293b !important;
    }

    .subtitle {
        font-size: 16px !important;
        color: #475569 !important;
    }

    .section-title {
        font-size: 21px !important;
        color: #1e293b !important;
    }

    .result-title {
        color: #334155 !important;
    }

    .result-value {
        font-size: 30px !important;
        color: #1e3a8a !important;
    }

    .stButton > button {
        background-color: #dbeafe !important;
        color: #1e3a8a !important;
    }

    .stButton > button p,
    .stButton > button span {
        color: #1e3a8a !important;
    }
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# MAKE src/ IMPORTABLE
# This file lives in app/, but the model-loading code lives in src/.
# This line tells Python to also look inside src/ when importing.
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import src.features as features
import sys

sys.modules["features"] = features

from src.predict import InvalidStudentInput, get_metadata, predict_student
# ----------------------------------------------------------------------
# PAGE SETUP
# ----------------------------------------------------------------------
st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")

PASS_MARK = 10  # our own assumption, used throughout this project

# Friendlier text for the form labels only. These keys must be REAL column
# names from the trained model. Changing this dict changes how a field is
# DISPLAYED, never what is sent to the model.
LABELS = {
    "school": "School",
    "sex": "Sex",
    "age": "Age",
    "address": "Home address",
    "famsize": "Family size",
    "Pstatus": "Parents' cohabitation status",
    "Medu": "Mother's education (0 = none, 4 = higher education)",
    "Fedu": "Father's education (0 = none, 4 = higher education)",
    "Mjob": "Mother's job",
    "Fjob": "Father's job",
    "reason": "Reason for choosing this school",
    "guardian": "Main guardian",
    "traveltime": "Home-to-school travel time (1 = under 15 min, 4 = over 1 hour)",
    "studytime": "Weekly study time (1 = under 2h, 4 = over 10h)",
    "failures": "Past class failures",
    "schoolsup": "Extra school support",
    "famsup": "Family educational support",
    "paid": "Paid extra classes",
    "activities": "Extracurricular activities",
    "nursery": "Attended nursery school",
    "higher": "Wants higher education",
    "internet": "Internet access at home",
    "romantic": "In a romantic relationship",
    "famrel": "Family relationship quality (1-5)",
    "freetime": "Free time after school (1-5)",
    "goout": "Going out with friends (1-5)",
    "Dalc": "Weekday alcohol use (1-5)",
    "Walc": "Weekend alcohol use (1-5)",
    "health": "Current health status (1-5)",
    "absences": "Number of school absences",
    "G1": "First period grade (0-20)",
    "G2": "Second period grade (0-20)",
}

# Groups fields into sections for a cleaner form. Any model column NOT listed
# here still appears automatically in an "Other" section further down, so a
# retrained model with different columns can never be silently left out.
SECTIONS = {
    "Student background": ["school", "sex", "age", "address", "famsize", "Pstatus"],
    "Family": ["Medu", "Fedu", "Mjob", "Fjob", "guardian", "famsup", "famrel"],
    "Academics": ["studytime", "traveltime", "failures", "schoolsup", "paid",
                  "higher", "reason", "internet", "absences", "G1", "G2"],
    "Lifestyle": ["activities", "nursery", "romantic", "freetime", "goout",
                  "Dalc", "Walc", "health"],
}


# ----------------------------------------------------------------------
# LOAD METADATA (cached: read from disk only once per app session)
# ----------------------------------------------------------------------
@st.cache_data
def load_metadata():
    return get_metadata()


try:
    meta = load_metadata()
except FileNotFoundError as error:
    st.error(
        "The trained model was not found.\n\n"
        f"{error}\n\n"
        "Run these from the project root before starting the app:\n"
        "1. python src/clean_data.py\n"
        "2. python src/train_models.py\n"
        "3. python src/select_best_model.py\n"
        "4. python src/save_model.py"
    )
    st.stop()

required_fields = meta["input_columns"]
schema = meta["input_schema"]


# ----------------------------------------------------------------------
# BUILD ONE INPUT WIDGET FOR ONE FIELD, DIRECTLY FROM THE MODEL'S SCHEMA
# ----------------------------------------------------------------------
def render_field(col):
    """Draw the right widget for one real model column and return what the user picked."""
    rule = schema[col]
    label = LABELS.get(col, col)   # falls back to the raw column name if not in LABELS

    if rule["type"] == "categorical":
        return st.selectbox(label, rule["values"], key=col)

    # every numeric column in this dataset is a whole number (age, ratings, grades, absences)
    lo, hi = int(rule["min"]), int(rule["max"])
    default = (lo + hi) // 2
    return st.slider(label, min_value=lo, max_value=hi, value=default, step=1, key=col)


# ----------------------------------------------------------------------
# PAGE HEADER
# ----------------------------------------------------------------------

st.markdown(
    '<div class="subtitle">Machine Learning Based Student Performance Prediction</div>',
    unsafe_allow_html=True
)
st.write(
    "Estimate a student's final grade (G3, on a scale of 0 to 20) from their "
    "background, study habits and lifestyle."
)


st.divider()

# ----------------------------------------------------------------------
# THE INPUT FORM
# st.form groups every widget together so the page only reruns the prediction
# when the button is clicked, not after every single slider movement.
# ----------------------------------------------------------------------
with st.form("student_form"):
    student = {}
    placed_columns = set()

    for section_title, columns in SECTIONS.items():
        columns_here = [c for c in columns if c in required_fields]
        if not columns_here:
            continue   # e.g. G1/G2 are skipped entirely for an early-warning (Scenario B) model
        st.subheader(section_title)
        cols_layout = st.columns(2)
        for i, col in enumerate(columns_here):
            with cols_layout[i % 2]:
                student[col] = render_field(col)
            placed_columns.add(col)

    # Safety net: any model column not covered by a section above still gets a widget
    leftover = [c for c in required_fields if c not in placed_columns]
    if leftover:
        st.subheader("Other")
        cols_layout = st.columns(2)
        for i, col in enumerate(leftover):
            with cols_layout[i % 2]:
                student[col] = render_field(col)

    submitted = st.form_submit_button("Predict final grade", use_container_width=True)

# ----------------------------------------------------------------------
# HANDLE THE PREDICTION
# ----------------------------------------------------------------------
if submitted:
    try:
        result = predict_student(student, return_details=True)
    except InvalidStudentInput as error:
        st.error(f"Could not make a prediction:\n\n{error}")
        st.stop()

    grade = result["predicted_grade"]
    passed = grade >= PASS_MARK

    st.divider()
    st.subheader("Result")

    left, right = st.columns(2)
    with left:
        st.metric("Predicted final grade (G3)", f"{grade:.1f} / 20")
    with right:
        st.metric("Likely outcome", "Pass ✅" if passed else "At risk ⚠️")

    st.progress(min(max(grade / 20, 0.0), 1.0))

    if result["clipped"]:
        st.caption(
            f"Note: the model's raw output ({result['raw_prediction']}) was outside the "
            f"0-20 grade scale and was capped to a valid grade."
        )

    for warning in result["warnings"]:
        st.warning(warning, icon="⚠️")

    with st.expander("What did I enter?"):
        st.json(student)

    st.caption(
        f"Remember: this model's typical error on unseen students was about "
        f"{meta['test_metrics']['MAE']:.1f} grade points. Use this as a rough guide, not a guarantee."
    )
