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
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)
# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>

/* ================================
   MAIN APPLICATION
================================ */

.stApp {
    background-color: #f8fafc !important;
    color: #1e293b !important;
}

[data-testid="stAppViewContainer"] {
    background-color: #f8fafc !important;
}

[data-testid="stHeader"] {
    background-color: #f8fafc !important;
}


/* ================================
   TITLE
================================ */

.main-title {
    text-align: center !important;
    font-size: 40px !important;
    font-weight: 700 !important;
    margin-top: 10px !important;
    margin-bottom: 10px !important;
    color: #1e293b !important;
    line-height: 1.2 !important;
}

.subtitle {
    text-align: center !important;
    font-size: 17px !important;
    margin-bottom: 30px !important;
    color: #475569 !important;
}


/* ================================
   SECTION HEADINGS
================================ */

.section-title {
    font-size: 24px !important;
    font-weight: 600 !important;
    margin-top: 20px !important;
    margin-bottom: 15px !important;
    color: #1e293b !important;
}


/* ================================
   INPUT LABELS
================================ */

[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span,
[data-testid="stWidgetLabel"] label {
    color: #334155 !important;
}


/* ================================
   SELECT BOX
================================ */

[data-baseweb="select"] {
    background-color: #ffffff !important;
}

[data-baseweb="select"] div {
    color: #1e293b !important;
}


/* Dropdown */
[data-baseweb="popover"] {
    background-color: #ffffff !important;
}

[data-baseweb="popover"] * {
    color: #1e293b !important;
}


/* ================================
   SLIDERS
================================ */

[data-testid="stSlider"] label,
[data-testid="stSlider"] p,
[data-testid="stSlider"] span {
    color: #334155 !important;
}


/* ================================
   PREDICT BUTTON
================================ */

/* IMPORTANT:
   This targets st.form_submit_button
*/

[data-testid="stFormSubmitButton"] > button {
    width: 100% !important;
    min-height: 52px !important;
    border-radius: 10px !important;

    background-color: #2563eb !important;
    color: #ffffff !important;

    border: 2px solid #1d4ed8 !important;

    font-size: 18px !important;
    font-weight: 700 !important;

    padding: 12px 20px !important;

    cursor: pointer !important;
}

/* Make sure the text inside the button is white */

[data-testid="stFormSubmitButton"] > button p,
[data-testid="stFormSubmitButton"] > button span {
    color: #ffffff !important;
    font-weight: 700 !important;
}


/* Hover */

[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #1d4ed8 !important;
    color: #ffffff !important;
}


/* ================================
   RESULT CARD
================================ */

.result-container {
    background: #ffffff !important;

    border: 1px solid #d9dfe8 !important;
    border-radius: 16px !important;

    padding: 24px !important;

    margin-top: 20px !important;
    margin-bottom: 20px !important;

    text-align: center !important;

    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06) !important;
}

.result-heading {
    font-size: 25px !important;
    font-weight: 700 !important;

    color: #1e293b !important;

    margin-bottom: 22px !important;
}

.result-item {
    background: #f8fafc !important;

    border-radius: 12px !important;

    padding: 16px !important;

    margin: 10px 0 !important;

    text-align: center !important;
}

.result-label {
    font-size: 15px !important;
    font-weight: 600 !important;

    color: #475569 !important;

    margin-bottom: 6px !important;
}

.result-number {
    font-size: 28px !important;
    font-weight: 700 !important;

    color: #1d4ed8 !important;
}

.result-pass {
    font-size: 24px !important;
    font-weight: 700 !important;

    color: #15803d !important;
}

.result-risk {
    font-size: 24px !important;
    font-weight: 700 !important;

    color: #b45309 !important;
}


/* ================================
   MOBILE RESPONSIVE
================================ */

@media (max-width: 768px) {

    .main-title {
        font-size: 29px !important;
        line-height: 1.25 !important;
        margin-top: 5px !important;
    }

    .subtitle {
        font-size: 15px !important;
        line-height: 1.4 !important;
        margin-bottom: 20px !important;
    }

    .section-title {
        font-size: 20px !important;
    }

    /* Bigger mobile button */

    [data-testid="stFormSubmitButton"] > button {
        min-height: 54px !important;

        font-size: 18px !important;

        padding: 13px 16px !important;

        border-radius: 10px !important;
    }

    [data-testid="stFormSubmitButton"] > button p,
    [data-testid="stFormSubmitButton"] > button span {
        color: #ffffff !important;
        font-size: 18px !important;
        font-weight: 700 !important;
    }

    /* Mobile result */

    .result-container {
        padding: 16px !important;
        border-radius: 14px !important;
    }

    .result-heading {
        font-size: 22px !important;
        margin-bottom: 16px !important;
    }

    .result-item {
        padding: 14px !important;
        margin: 9px 0 !important;
    }

    .result-label {
        font-size: 14px !important;
    }

    .result-number {
        font-size: 26px !important;
    }

    .result-pass,
    .result-risk {
        font-size: 22px !important;
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

# ============================================================
# RESPONSIVE PREDICTION RESULT
# ============================================================

outcome_text = "PASS ✅" if passed else "AT RISK ⚠️"
outcome_class = "result-pass" if passed else "result-risk"

mae = meta["test_metrics"]["MAE"]

st.markdown(
    f"""
    <div class="result-container">

        <div class="result-heading">
            🎓 Prediction Result
        </div>

        <div class="result-item">
            <div class="result-label">
                Predicted Final Grade (G3)
            </div>

            <div class="result-number">
                {grade:.1f} / 20
            </div>
        </div>

        <div class="result-item">
            <div class="result-label">
                Likely Outcome
            </div>

            <div class="{outcome_class}">
                {outcome_text}
            </div>
        </div>

        <div class="result-item">
            <div class="result-label">
                Expected Test MAE
            </div>

            <div class="result-number">
                ± {mae:.1f} grade points
            </div>
        </div>

    </div>
    """,
    unsafe_allow_html=True
)

st.progress(
    min(max(grade / 20, 0.0), 1.0),
    text=f"Grade: {grade:.1f} / 20"
)

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
