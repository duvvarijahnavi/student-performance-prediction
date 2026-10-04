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
    background-color: #f5f7fb;
    color: #222222;
}

/* Main title */
.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: 700;
    margin-top: 10px;
    margin-bottom: 5px;
    color: #222222;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 17px;
    margin-bottom: 30px;
    color: #444444;
}

/* Section title */
.section-title {
    font-size: 24px;
    font-weight: 600;
    margin-top: 20px;
    margin-bottom: 15px;
    color: #222222;
}

/* Result box */
.result-box {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #d9dfe8;
    margin-top: 20px;
    color: #222222;
}

/* Result title */
.result-title {
    font-size: 18px;
    font-weight: 600;
    color: #222222;
}

/* Result value */
.result-value {
    font-size: 32px;
    font-weight: 700;
    color: #222222;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    padding: 10px;
    font-size: 17px;
    font-weight: 600;
    background-color: #dbeafe;
    color: #1e3a8a;
    border: 1px solid #93c5fd;
}

.stButton > button:hover {
    background-color: #bfdbfe;
    color: #1e3a8a;
}
}

/* Input spacing */
.stNumberInput,
.stSelectbox,
.stTextInput {
    margin-bottom: 10px;
}

/* Mobile screen */
@media (max-width: 768px) {
    .main-title {
        font-size: 30px;
        color: #222222;
    }

    .subtitle {
        font-size: 16px;
        color: #444444;
    }

    .section-title {
        font-size: 21px;
        color: #222222;
    }

    .result-value {
        font-size: 28px;
        color: #222222;
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
st.info(
    f"**Model in use:** {meta['model_name']}  |  **Scenario:** {meta['scenario']}\n\n"
    f"On unseen test students, predictions were typically off by about "
    f"**{meta['test_metrics']['MAE']:.1f} grade points** (Mean Absolute Error). "
    "Treat every prediction as an estimate, not a certainty.",
    icon="ℹ️",
)
st.warning(
    "**Educational project only.** This tool is a machine learning demonstration "
    "and must not be used to make real decisions about a student's education, "
    "placement or support.",
    icon="⚠️",
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
