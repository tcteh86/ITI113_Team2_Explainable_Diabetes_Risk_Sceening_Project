import json
import os
import time
from datetime import datetime, timezone
from io import StringIO
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import requests
import streamlit as st


# =====================================================================
# Team02 Diabetes Screening Demo
# Dataset contract:
# diabetes_012_health_indicators_BRFSS2015.csv
# 253,680 rows, 21 predictors + Diabetes_012 target.
#
# IMPORTANT MODEL SEMANTICS
# The deployed Team02 XGBoost model is BINARY:
#     0 = no diabetes
#     1 = prediabetes OR diabetes
# because the modelling pipeline converts Diabetes_012 > 0 to class 1.
# Therefore the UI must NOT claim that class 1 means confirmed diabetes.
# =====================================================================

FEATURE_COLUMNS = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]
TARGET_COLUMN = "Diabetes_012"
FULL_DATASET_COLUMNS = [TARGET_COLUMN] + FEATURE_COLUMNS

BINARY_COLUMNS = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex",
]

AGE_OPTIONS = {
    "18–24": 1, "25–29": 2, "30–34": 3, "35–39": 4, "40–44": 5,
    "45–49": 6, "50–54": 7, "55–59": 8, "60–64": 9, "65–69": 10,
    "70–74": 11, "75–79": 12, "80+": 13,
}
EDUCATION_OPTIONS = {
    "None / kindergarten": 1,
    "Elementary school": 2,
    "Some high school": 3,
    "High school graduate": 4,
    "Some college / technical school": 5,
    "College graduate": 6,
}
INCOME_OPTIONS = {
    "< $10,000": 1,
    "$10,000–$14,999": 2,
    "$15,000–$19,999": 3,
    "$20,000–$24,999": 4,
    "$25,000–$34,999": 5,
    "$35,000–$49,999": 6,
    "$50,000–$74,999": 7,
    "$75,000+": 8,
}
GENERAL_HEALTH_OPTIONS = {
    "Excellent": 1,
    "Very good": 2,
    "Good": 3,
    "Fair": 4,
    "Poor": 5,
}
YES_NO = {"No": 0, "Yes": 1}
SEX_OPTIONS = {"Female": 0, "Male": 1}

DATASET_LABELS = {
    0: "No diabetes",
    1: "Prediabetes",
    2: "Diabetes",
}

DEFAULT_RELAY_URL = os.getenv(
    "RELAY_PREDICT_URL",
    "http://127.0.0.1:8000/predict",
)
DEFAULT_RELAY_API_KEY = os.getenv("RELAY_API_KEY", "")

MAX_BATCH_ROWS = int(os.getenv("MAX_BATCH_ROWS", "100"))


# ---------------------------------------------------------------------
# Streamlit page + senior-friendly styling
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Team02 Diabetes Screening",
    page_icon="🩺",
    layout="wide",
)

st.markdown(
    """
<style>
html, body, [class*="css"] {
    font-size: 19px;
}
h1 {
    font-size: 2.35rem !important;
    line-height: 1.2 !important;
}
h2 {
    font-size: 1.75rem !important;
}
h3 {
    font-size: 1.38rem !important;
}
div[data-testid="stButton"] button,
div[data-testid="stFormSubmitButton"] button,
div[data-testid="stDownloadButton"] button {
    min-height: 3.2rem;
    font-size: 1.08rem;
    font-weight: 700;
}
div[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,0.35);
    border-radius: 0.75rem;
    padding: 0.7rem;
}
[data-testid="stSidebar"] {
    min-width: 320px;
}
.small-note {
    font-size: 0.92rem;
}
.big-result {
    font-size: 1.65rem;
    font-weight: 800;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🩺 Diabetes Screening Support")
st.write(
    "A simple Team02 academic demo using 21 BRFSS 2015 health indicators."
)

st.warning(
    "This is an academic screening-support prototype, not a medical diagnosis. "
    "A positive output means the model found a pattern associated with "
    "prediabetes or diabetes. Please consult a healthcare professional for diagnosis."
)


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def health_url_from_predict(predict_url: str) -> str:
    parts = urlsplit(predict_url.strip())
    path = parts.path or "/"
    if path.endswith("/predict"):
        path = path[:-len("/predict")] + "/health"
    else:
        path = "/health"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def request_headers(relay_api_key: str) -> dict:
    return {
        "Content-Type": "application/json",
        "x-api-key": relay_api_key,
        "ngrok-skip-browser-warning": "true",
    }


def yes_no_radio(
    label: str,
    *,
    default: str = "No",
    help_text: str | None = None,
    key: str,
) -> int:
    options = list(YES_NO.keys())
    return YES_NO[
        st.radio(
            label,
            options,
            index=options.index(default),
            horizontal=True,
            help=help_text,
            key=key,
        )
    ]


def dataset_label_text(value) -> str:
    try:
        key = int(float(value))
        return DATASET_LABELS.get(key, f"Unknown ({value})")
    except Exception:
        return ""


def binary_actual_from_dataset_label(value):
    """Match the deployed model target: Diabetes_012 > 0."""
    try:
        return int(float(value) > 0)
    except Exception:
        return None


def model_result_text(screening_class: int) -> str:
    """
    Plain-language screening result.
    IMPORTANT: class 1 combines prediabetes and diabetes in the Team02 model.
    """
    return (
        "Follow-up suggested"
        if int(screening_class) == 1
        else "No warning detected"
    )


def result_explanation(screening_class: int) -> str:
    if int(screening_class) == 1:
        return (
            "The screening model found a stronger pattern associated with "
            "prediabetes or diabetes. This does not mean the person has diabetes. "
            "A healthcare professional would be needed for diagnosis."
        )
    return (
        "The screening model did not find a strong warning pattern. "
        "This does not guarantee that the person does not have diabetes."
    )


def next_step_text(screening_class: int) -> str:
    return (
        "Consider follow-up screening"
        if int(screening_class) == 1
        else "No model follow-up flag"
    )


def normalise_payload(row: pd.Series) -> dict:
    payload = {}
    for col in FEATURE_COLUMNS:
        value = row[col]
        if col == "BMI":
            payload[col] = float(value)
        else:
            # Dataset columns are whole-number coded values.
            payload[col] = int(float(value))
    return payload


def validate_input_dataframe(input_df: pd.DataFrame):
    """Validate against the uploaded BRFSS2015 012 dataset contract."""
    errors = []
    df = input_df.copy()

    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        errors.append("Missing required columns: " + ", ".join(missing))
        return df, errors

    # Keep target only when supplied; ignore other extra columns for safety.
    keep = FEATURE_COLUMNS.copy()
    if TARGET_COLUMN in df.columns:
        keep = [TARGET_COLUMN] + keep
    df = df[keep].copy()

    # Convert everything to numeric.
    for col in keep:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[FEATURE_COLUMNS].isna().any().any():
        bad = [
            c for c in FEATURE_COLUMNS
            if df[c].isna().any()
        ]
        errors.append(
            "Blank or non-numeric values found in: " + ", ".join(bad)
        )

    for col in BINARY_COLUMNS:
        bad = df.loc[~df[col].isin([0, 1]) & df[col].notna(), col]
        if len(bad):
            errors.append(f"{col} must contain only 0 or 1.")

    range_rules = {
        "BMI": (12, 98),
        "GenHlth": (1, 5),
        "MentHlth": (0, 30),
        "PhysHlth": (0, 30),
        "Age": (1, 13),
        "Education": (1, 6),
        "Income": (1, 8),
    }
    for col, (low, high) in range_rules.items():
        bad = df.loc[
            ((df[col] < low) | (df[col] > high)) & df[col].notna(),
            col,
        ]
        if len(bad):
            errors.append(
                f"{col} must be between {low} and {high} "
                "to match the uploaded BRFSS2015 dataset."
            )

    if TARGET_COLUMN in df.columns:
        bad = df.loc[
            ~df[TARGET_COLUMN].isin([0, 1, 2]) & df[TARGET_COLUMN].notna(),
            TARGET_COLUMN,
        ]
        if len(bad):
            errors.append("Diabetes_012 must contain only 0, 1 or 2.")

    return df, errors


def parse_pasted_csv(text: str) -> pd.DataFrame:
    """
    Accept:
      A) header + one/many rows
      B) headerless 21-feature row(s)
      C) headerless 22-column original-dataset row(s), target first
    """
    text = text.strip()
    if not text:
        raise ValueError("No CSV text was entered.")

    # First try normal CSV with a header.
    candidate = pd.read_csv(StringIO(text))
    if set(FEATURE_COLUMNS).issubset(candidate.columns):
        return candidate

    # Otherwise interpret as headerless raw rows.
    candidate = pd.read_csv(StringIO(text), header=None)

    if candidate.shape[1] == len(FEATURE_COLUMNS):
        candidate.columns = FEATURE_COLUMNS
        return candidate

    if candidate.shape[1] == len(FULL_DATASET_COLUMNS):
        candidate.columns = FULL_DATASET_COLUMNS
        return candidate

    raise ValueError(
        f"Could not recognise CSV layout. Found {candidate.shape[1]} columns. "
        f"Expected 21 feature columns or 22 columns including {TARGET_COLUMN}."
    )


def call_relay(relay_url: str, relay_api_key: str, payload: dict):
    start = time.perf_counter()
    response = requests.post(
        relay_url.strip(),
        headers=request_headers(relay_api_key),
        json=payload,
        timeout=75,
    )
    latency = time.perf_counter() - start

    if not response.ok:
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        raise RuntimeError(
            f"Prediction request failed: HTTP {response.status_code} — {detail}"
        )

    result = response.json()
    if isinstance(result, list):
        if not result:
            raise RuntimeError("The endpoint returned an empty result list.")
        result = result[0]

    required = ["screening_class", "probability", "decision_threshold"]
    missing = [k for k in required if k not in result]
    if missing:
        raise RuntimeError(
            "Endpoint response is missing: " + ", ".join(missing)
        )

    return result, latency, response.status_code


def show_single_result(result: dict, latency: float):
    screening_class = int(result["screening_class"])
    probability = float(result["probability"])

    st.divider()
    st.subheader("Your screening result")

    if screening_class == 1:
        st.error("### ⚠️ Follow-up suggested")
        st.write(
            "**The model found signs linked with prediabetes or diabetes.**"
        )
    else:
        st.success("### ✅ No warning detected")
        st.write(
            "**The model did not find a strong diabetes warning pattern.**"
        )

    st.metric(
        "Screening score",
        f"{probability:.0%}",
        help=(
            "A higher score means the model found a stronger pattern linked with "
            "the positive screening group. It is not a medical diagnosis."
        ),
    )

    st.info(result_explanation(screening_class))

    with st.expander("Technical details for the project assessor"):
        st.caption(f"App round-trip time: {latency:.2f} seconds")
        st.json(result)


# ---------------------------------------------------------------------
# Technical connection — intentionally de-emphasised for senior users.
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("System")
    st.write("Team02 · XGBoost · SageMaker")

    with st.expander("Technical connection (demo operator)", expanded=False):
        relay_url = st.text_input(
            "Relay predict URL",
            value=DEFAULT_RELAY_URL,
        )
        relay_api_key = st.text_input(
            "Relay API key",
            value=DEFAULT_RELAY_API_KEY,
            type="password",
        )

        if st.button("Check system connection", use_container_width=True):
            if not relay_api_key:
                st.error("Set RELAY_API_KEY first.")
            else:
                try:
                    response = requests.get(
                        health_url_from_predict(relay_url),
                        headers=request_headers(relay_api_key),
                        timeout=20,
                    )
                    if response.ok:
                        health = response.json()
                        status = health.get("endpoint_status", "Unknown")
                        if status == "InService":
                            st.success("System is ready.")
                        else:
                            st.warning(f"Relay connected. Endpoint: {status}")
                        st.json(health)
                    else:
                        st.error(
                            f"Connection check failed: HTTP {response.status_code}"
                        )
                        st.code(response.text)
                except Exception as exc:
                    st.error(f"Connection check failed: {exc}")

    st.divider()
    st.caption("Academic screening support only.")
    st.caption("Frozen threshold: 0.45")


# ---------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------
tab_easy, tab_csv, tab_about = st.tabs(
    ["👤 Easy single-person form", "📄 CSV one / many rows", "ℹ️ About"]
)


# =====================================================================
# TAB 1 — Senior-friendly single person form
# =====================================================================
with tab_easy:
    st.header("Enter one person's information")
    st.write(
        "Questions are grouped into four short sections. "
        "Choose the answer that best matches the person."
    )

    with st.form("senior_friendly_diabetes_form"):
        with st.expander("1️⃣ About you", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                age_label = st.selectbox(
                    "Age group",
                    list(AGE_OPTIONS.keys()),
                    index=7,
                )
                sex_label = st.radio(
                    "Sex recorded in the dataset",
                    list(SEX_OPTIONS.keys()),
                    horizontal=True,
                )
            with c2:
                education_label = st.selectbox(
                    "Highest education",
                    list(EDUCATION_OPTIONS.keys()),
                    index=4,
                )
                income_label = st.selectbox(
                    "Household income",
                    list(INCOME_OPTIONS.keys()),
                    index=5,
                )

        with st.expander("2️⃣ Health conditions", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                high_bp = yes_no_radio(
                    "High blood pressure?",
                    key="high_bp",
                )
                high_chol = yes_no_radio(
                    "High cholesterol?",
                    key="high_chol",
                )
                chol_check = yes_no_radio(
                    "Cholesterol checked in the past 5 years?",
                    default="Yes",
                    key="chol_check",
                )
                bmi = st.number_input(
                    "Body Mass Index (BMI)",
                    min_value=12.0,
                    max_value=98.0,
                    value=25.0,
                    step=1.0,
                )
            with c2:
                stroke = yes_no_radio(
                    "History of stroke?",
                    key="stroke",
                )
                heart_disease = yes_no_radio(
                    "History of heart disease or heart attack?",
                    key="heart_disease",
                )
                diff_walk = yes_no_radio(
                    "Serious difficulty walking or climbing stairs?",
                    key="diff_walk",
                )
                gen_hlth_label = st.selectbox(
                    "Overall general health",
                    list(GENERAL_HEALTH_OPTIONS.keys()),
                    index=2,
                )

        with st.expander("3️⃣ Daily habits", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                smoker = yes_no_radio(
                    "Smoker indicator?",
                    key="smoker",
                )
                physical_activity = yes_no_radio(
                    "Physically active?",
                    default="Yes",
                    key="phys_activity",
                )
                fruits = yes_no_radio(
                    "Fruit daily?",
                    default="Yes",
                    key="fruits",
                )
            with c2:
                veggies = yes_no_radio(
                    "Vegetables daily?",
                    default="Yes",
                    key="veggies",
                )
                heavy_alcohol = yes_no_radio(
                    "Heavy alcohol consumption indicator?",
                    key="heavy_alcohol",
                )

        with st.expander("4️⃣ Healthcare & wellbeing", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                any_healthcare = yes_no_radio(
                    "Any healthcare coverage?",
                    default="Yes",
                    key="healthcare",
                )
                no_doc_cost = yes_no_radio(
                    "Could not see a doctor because of cost in the past 12 months?",
                    key="no_doc_cost",
                )
            with c2:
                ment_hlth = st.slider(
                    "Poor mental-health days in the past 30 days",
                    0, 30, 0,
                )
                phys_hlth = st.slider(
                    "Poor physical-health days in the past 30 days",
                    0, 30, 0,
                )

        st.markdown("#### Before you continue")
        confirm = st.checkbox(
            "I understand this is screening support and not a medical diagnosis."
        )

        submitted = st.form_submit_button(
            "Run diabetes screening",
            type="primary",
            use_container_width=True,
        )

    payload = {
        "HighBP": high_bp,
        "HighChol": high_chol,
        "CholCheck": chol_check,
        "BMI": float(bmi),
        "Smoker": smoker,
        "Stroke": stroke,
        "HeartDiseaseorAttack": heart_disease,
        "PhysActivity": physical_activity,
        "Fruits": fruits,
        "Veggies": veggies,
        "HvyAlcoholConsump": heavy_alcohol,
        "AnyHealthcare": any_healthcare,
        "NoDocbcCost": no_doc_cost,
        "GenHlth": GENERAL_HEALTH_OPTIONS[gen_hlth_label],
        "MentHlth": int(ment_hlth),
        "PhysHlth": int(phys_hlth),
        "DiffWalk": diff_walk,
        "Sex": SEX_OPTIONS[sex_label],
        "Age": AGE_OPTIONS[age_label],
        "Education": EDUCATION_OPTIONS[education_label],
        "Income": INCOME_OPTIONS[income_label],
    }

    if submitted:
        if not confirm:
            st.error(
                "Please tick the acknowledgement before running the screening."
            )
        elif not relay_url.strip():
            st.error("The demo operator must configure the relay URL.")
        elif not relay_api_key:
            st.error("The demo operator must configure RELAY_API_KEY.")
        else:
            try:
                result, latency, http_status = call_relay(
                    relay_url,
                    relay_api_key,
                    payload,
                )
                show_single_result(result, latency)

                st.success(
                    "End-to-end path verified: "
                    "Streamlit → FastAPI relay → SageMaker → result."
                )

                demo_evidence = {
                    "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                    "mode": "single_person",
                    "architecture_path": (
                        "Streamlit -> FastAPI relay -> "
                        "SageMaker Serverless Endpoint -> Streamlit"
                    ),
                    "http_status": int(http_status),
                    "client_round_trip_seconds": round(latency, 4),
                    "input_feature_count": len(payload),
                    "endpoint_result": result,
                    "credential_note": (
                        "Relay API key used at runtime; secret value not recorded."
                    ),
                }

                st.download_button(
                    "Download demo evidence",
                    data=json.dumps(demo_evidence, indent=2),
                    file_name="team02_streamlit_single_e2e_evidence.json",
                    mime="application/json",
                    use_container_width=True,
                )

            except requests.Timeout:
                st.error(
                    "The request timed out. The SageMaker Serverless endpoint "
                    "may be starting up. Please try once more."
                )
            except Exception as exc:
                st.error("Prediction failed.")
                st.exception(exc)


# =====================================================================
# TAB 2 — CSV input: one or multiple rows
# =====================================================================
with tab_csv:
    st.header("Use CSV data")
    st.write(
        "You can upload a CSV file or paste one or many CSV rows. "
        "The app accepts either the 21 model features or the original "
        "22-column dataset format containing `Diabetes_012`."
    )

    st.info(
        "If `Diabetes_012` is included, the app keeps it only for comparison. "
        "It is never sent to the model."
    )

    source_mode = st.radio(
        "Choose input method",
        ["Upload CSV file", "Paste CSV row(s)"],
        horizontal=True,
    )

    raw_df = None

    if source_mode == "Upload CSV file":
        uploaded = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
        )
        if uploaded is not None:
            try:
                raw_df = pd.read_csv(uploaded)
            except Exception as exc:
                st.error(f"Could not read the CSV file: {exc}")

    else:
        with st.expander("Show accepted CSV formats"):
            st.code(
                ",".join(FEATURE_COLUMNS),
                language="text",
            )
            st.write(
                "You may paste a header followed by data rows, or paste "
                "headerless rows copied directly from the source dataset."
            )

        pasted = st.text_area(
            "Paste one or more comma-separated rows",
            height=180,
            placeholder=(
                "Example headerless 22-column row:\n"
                "0,1,1,1,40,1,0,0,0,0,1,0,1,0,5,18,15,1,0,9,4,3"
            ),
        )
        if pasted.strip():
            try:
                raw_df = parse_pasted_csv(pasted)
            except Exception as exc:
                st.error(str(exc))

    if raw_df is not None:
        checked_df, validation_errors = validate_input_dataframe(raw_df)

        st.subheader("Rows ready for checking")
        st.write(f"Rows received: **{len(checked_df):,}**")
        st.dataframe(checked_df.head(20), use_container_width=True)

        if len(checked_df) > 20:
            st.caption("Showing the first 20 rows only.")

        if validation_errors:
            st.error("Please fix these CSV problems:")
            for err in validation_errors:
                st.write("• " + err)
        elif len(checked_df) == 0:
            st.error("No data rows found.")
        elif len(checked_df) > MAX_BATCH_ROWS:
            st.error(
                f"This demo limits each batch to {MAX_BATCH_ROWS} rows "
                "to avoid accidental large endpoint workloads. "
                "Split the file into smaller batches or change MAX_BATCH_ROWS."
            )
        else:
            st.success("CSV structure matches the Team02 model input contract.")

            confirm_batch = st.checkbox(
                "I understand these are screening-support model outputs, "
                "not medical diagnoses.",
                key="confirm_batch",
            )

            if st.button(
                f"Run screening for {len(checked_df)} row(s)",
                type="primary",
                use_container_width=True,
            ):
                if not confirm_batch:
                    st.error("Please tick the acknowledgement first.")
                elif not relay_url.strip():
                    st.error("The demo operator must configure the relay URL.")
                elif not relay_api_key:
                    st.error("The demo operator must configure RELAY_API_KEY.")
                else:
                    progress = st.progress(0)
                    status_box = st.empty()
                    output_rows = []
                    total = len(checked_df)

                    for position, (_, row) in enumerate(
                        checked_df.iterrows(),
                        start=1,
                    ):
                        status_box.write(
                            f"Checking row {position} of {total}..."
                        )

                        try:
                            payload = normalise_payload(row)
                            result, latency, http_status = call_relay(
                                relay_url,
                                relay_api_key,
                                payload,
                            )

                            screening_class = int(result["screening_class"])
                            probability = float(result["probability"])
                            threshold = float(result["decision_threshold"])

                            output = {
                                "Person": position,
                                "Screening result": model_result_text(screening_class),
                                "Screening score": round(probability * 100, 1),
                                "What to do": next_step_text(screening_class),
                                "Status": "Success",
                                # Technical fields are kept for evidence/download,
                                # but are hidden from the default senior-facing table.
                                "_model_class": screening_class,
                                "_probability": probability,
                                "_threshold": threshold,
                                "_latency_seconds": round(latency, 3),
                            }

                            if TARGET_COLUMN in checked_df.columns:
                                actual_value = row[TARGET_COLUMN]
                                actual_binary = binary_actual_from_dataset_label(
                                    actual_value
                                )
                                output["Known status in sample data"] = dataset_label_text(
                                    actual_value
                                )
                                output["Matches known status?"] = (
                                    "Yes"
                                    if screening_class == actual_binary
                                    else "No"
                                )

                                if screening_class == 1 and actual_binary == 0:
                                    output["Comparison note"] = (
                                        "Model suggested follow-up, but sample label says no diabetes"
                                    )
                                elif screening_class == 0 and actual_binary == 1:
                                    output["Comparison note"] = (
                                        "Model gave no warning, but sample label is prediabetes/diabetes"
                                    )
                                else:
                                    output["Comparison note"] = "Model and sample label agree"

                            output_rows.append(output)

                        except Exception as exc:
                            output_rows.append(
                                {
                                    "Person": position,
                                    "Screening result": "Could not check",
                                    "Screening score": None,
                                    "What to do": "Please retry",
                                    "Status": f"Error: {exc}",
                                    "_model_class": None,
                                    "_probability": None,
                                    "_threshold": None,
                                    "_latency_seconds": None,
                                }
                            )

                        progress.progress(position / total)

                    status_box.empty()
                    results_df = pd.DataFrame(output_rows)

                    st.divider()
                    st.header("Screening results")

                    successful = (
                        results_df["Status"].eq("Success").sum()
                        if "Status" in results_df.columns
                        else 0
                    )
                    follow_up = (
                        results_df["Screening result"]
                        .eq("Follow-up suggested")
                        .sum()
                        if "Screening result" in results_df.columns
                        else 0
                    )
                    no_warning = (
                        results_df["Screening result"]
                        .eq("No warning detected")
                        .sum()
                        if "Screening result" in results_df.columns
                        else 0
                    )

                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("People checked", successful)
                    with m2:
                        st.metric("Follow-up suggested", follow_up)
                    with m3:
                        st.metric("No warning detected", no_warning)

                    # Senior-facing table: show only plain-language fields.
                    friendly_columns = [
                        "Person",
                        "Screening result",
                        "Screening score",
                        "What to do",
                    ]
                    if "Known status in sample data" in results_df.columns:
                        friendly_columns += [
                            "Known status in sample data",
                            "Matches known status?",
                            "Comparison note",
                        ]

                    friendly_columns = [
                        c for c in friendly_columns if c in results_df.columns
                    ]

                    friendly_df = results_df[friendly_columns].copy()
                    if "Screening score" in friendly_df.columns:
                        friendly_df["Screening score"] = (
                            friendly_df["Screening score"]
                            .apply(
                                lambda x: f"{x:.0f}%"
                                if pd.notna(x)
                                else ""
                            )
                        )

                    st.dataframe(
                        friendly_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.caption(
                        "Screening score: higher means the model found a stronger "
                        "pattern linked with prediabetes/diabetes. It is not a diagnosis."
                    )

                    if "Known status in sample data" in results_df.columns:
                        comparable = results_df[
                            results_df["Status"].eq("Success")
                        ].copy()

                        matches = (
                            comparable["Matches known status?"].eq("Yes").sum()
                        )
                        extra_warnings = (
                            (
                                comparable["Screening result"].eq("Follow-up suggested")
                                & comparable["Known status in sample data"].eq("No diabetes")
                            )
                            .sum()
                        )
                        missed_warnings = (
                            (
                                comparable["Screening result"].eq("No warning detected")
                                & comparable["Known status in sample data"].isin(
                                    ["Prediabetes", "Diabetes"]
                                )
                            )
                            .sum()
                        )

                        st.subheader("Comparison with the sample's known labels")
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric("Matched sample label", int(matches))
                        with c2:
                            st.metric("Extra follow-up warnings", int(extra_warnings))
                        with c3:
                            st.metric("Missed warnings", int(missed_warnings))

                        st.info(
                            "A mismatch does not mean the Streamlit app is sending the "
                            "wrong data. It means the model prediction differs from the "
                            "known label for that row. Screening models can make both "
                            "extra warnings and missed warnings."
                        )

                    with st.expander("Technical validation details for the project assessor"):
                        technical_cols = [
                            c for c in results_df.columns
                            if c.startswith("_") or c in ["Person", "Status"]
                        ]
                        st.dataframe(
                            results_df[technical_cols],
                            use_container_width=True,
                            hide_index=True,
                        )

                    st.download_button(
                        "Download full results CSV",
                        data=results_df.to_csv(index=False),
                        file_name="team02_diabetes_batch_results.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                    evidence = {
                        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                        "mode": "csv_batch",
                        "rows_submitted": int(total),
                        "rows_successful": int(successful),
                        "follow_up_suggested": int(follow_up),
                        "no_warning_detected": int(no_warning),
                        "feature_count": len(FEATURE_COLUMNS),
                        "source_target_present": bool(
                            TARGET_COLUMN in checked_df.columns
                        ),
                        "architecture_path": (
                            "Streamlit -> FastAPI relay -> "
                            "SageMaker Serverless Endpoint -> Streamlit"
                        ),
                        "credential_note": (
                            "Relay API key used at runtime; secret value not recorded."
                        ),
                    }

                    st.download_button(
                        "Download batch evidence JSON",
                        data=json.dumps(evidence, indent=2),
                        file_name="team02_streamlit_batch_e2e_evidence.json",
                        mime="application/json",
                        use_container_width=True,
                    )


# =====================================================================
# TAB 3 — Dataset/model semantics and governance
# =====================================================================
with tab_about:
    st.header("About this model and dataset")

    st.subheader("Source dataset")
    st.markdown(
        """
The application input contract matches:

**`diabetes_012_health_indicators_BRFSS2015.csv`**

- 21 predictor fields
- Original target: `Diabetes_012`
- `0` = No diabetes
- `1` = Prediabetes
- `2` = Diabetes
"""
    )

    st.subheader("Why the model output has two groups")
    st.warning(
        "The deployed Team02 model does not perform the original 3-class prediction. "
        "During modelling, the target was converted to binary using "
        "`Diabetes_012 > 0`. Therefore class 1 combines prediabetes and diabetes."
    )

    st.markdown(
        """
**Deployed-model meaning**

| Model class | Correct UI wording |
|---|---|
| 0 | No diabetes signal |
| 1 | Prediabetes / diabetes signal |

A strict **'Diabetes' vs 'No diabetes'** prediction would require a different
target definition and model retraining. The current application deliberately
does not mislabel prediabetes as diabetes.
"""
    )

    st.subheader("Responsible-AI controls")
    st.markdown(
        """
- Plain-language intended-use warning.
- Human acknowledgement before prediction.
- Input validation against the BRFSS coding domains.
- CSV target column is excluded from inference to prevent leakage.
- Errors are surfaced instead of silently generating a result.
- The relay API key is never written into evidence files.
- No claim of medical diagnosis is made.
- No claim of per-prediction SHAP/LIME explanations is made.
"""
    )

    st.subheader("Deployment")
    st.markdown(
        """
**Runtime path**

`User → Streamlit → FastAPI relay → SageMaker Serverless Endpoint → XGBoost → Result`
"""
    )

