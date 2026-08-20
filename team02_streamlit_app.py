import json
import os
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st

from team02_frontend_core import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    RelayClientError,
    binary_actual_from_dataset_label,
    call_relay,
    check_relay_health,
    dataset_label_text,
    model_result_text,
    next_step_text,
    normalise_payload,
    parse_pasted_csv,
    parse_uploaded_csv,
    validate_input_dataframe,
)


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
    font-size: 20px;
    line-height: 1.55;
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
div[role="radiogroup"] label {
    min-height: 2.8rem;
    padding-right: 1.2rem;
}
div[data-testid="stAlert"] {
    border-radius: 0.8rem;
    padding: 1rem;
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

st.title("🩺 Health Screening Check")
st.write(
    "Answer a few health questions for one person, or check several records "
    "from a CSV file."
)

st.info(
    "**Please remember:** This academic screening-support prototype does not "
    "provide a medical diagnosis. It can only suggest whether a follow-up "
    "conversation with a healthcare professional may be helpful.",
    icon="ℹ️",
)


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
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


def show_relay_failure(exc: RelayClientError) -> None:
    """Show a clear error first and keep diagnostics under an expander."""
    st.error(str(exc))
    if exc.detail:
        with st.expander("Technical error details for the demo operator"):
            st.code(exc.detail)


def show_single_result(result: dict, latency: float):
    screening_class = int(result["screening_class"])
    probability = float(result["probability"])

    st.divider()
    st.subheader("Screening result")

    if screening_class == 1:
        st.warning("### Follow-up suggested", icon="📋")
        st.write(
            "This screening check found a pattern linked with prediabetes or "
            "diabetes. It does not mean that the person has diabetes."
        )
        st.write(
            "**Next step:** Consider discussing this result with a healthcare "
            "professional."
        )
    else:
        st.success("### No warning detected", icon="✅")
        st.write(
            "This screening check did not find a strong warning pattern. This "
            "does not guarantee that the person does not have diabetes."
        )

        st.write(
            "**Next step:** Continue usual health checks and speak with a "
            "healthcare professional if you have concerns."
        )

    with st.expander("Technical details for the project assessor"):
        st.metric("Screening score", f"{probability:.0%}")
        st.caption(
            "This score is technical model information. It is not a diagnosis "
            "or a measured chance of having diabetes."
        )
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
                    health = check_relay_health(
                        relay_url,
                        relay_api_key,
                    )
                    status = health.get("endpoint_status", "Unknown")
                    if status == "InService":
                        st.success("System is ready.")
                    else:
                        st.warning(f"Relay connected. Endpoint: {status}")
                    st.json(health)
                except requests.Timeout:
                    st.error(
                        "The health check timed out. The relay or SageMaker "
                        "endpoint may still be starting."
                    )
                except RelayClientError as exc:
                    show_relay_failure(exc)

    st.divider()
    st.caption("Academic screening support only.")
    st.caption("The decision threshold is supplied by the deployed endpoint.")


# ---------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------
tab_easy, tab_csv, tab_about = st.tabs(
    ["👤 Check one person", "📄 Check a CSV file", "ℹ️ About this check"]
)


# =====================================================================
# TAB 1 — Senior-friendly single person form
# =====================================================================
with tab_easy:
    st.header("Check one person")
    st.write(
        "Complete the four short sections below. Choose the answer that best "
        "matches the person being checked."
    )

    with st.form("senior_friendly_diabetes_form"):
        with st.expander("1️⃣ About the person", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                age_label = st.selectbox(
                    "Age group",
                    list(AGE_OPTIONS.keys()),
                    index=7,
                )
                sex_label = st.radio(
                    "Sex recorded in the health survey",
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
                    "Have you been told you have high blood pressure?",
                    key="high_bp",
                )
                high_chol = yes_no_radio(
                    "Have you been told you have high cholesterol?",
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
                    "Have you ever had a stroke?",
                    key="stroke",
                )
                heart_disease = yes_no_radio(
                    "Have you ever had heart disease or a heart attack?",
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

        with st.expander("3️⃣ Everyday health habits", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                smoker = yes_no_radio(
                    "Have you smoked at least 100 cigarettes in your lifetime?",
                    key="smoker",
                )
                physical_activity = yes_no_radio(
                    "Were you physically active in the past 30 days?",
                    default="Yes",
                    key="phys_activity",
                )
                fruits = yes_no_radio(
                    "Do you eat fruit at least once a day?",
                    default="Yes",
                    key="fruits",
                )
            with c2:
                veggies = yes_no_radio(
                    "Do you eat vegetables at least once a day?",
                    default="Yes",
                    key="veggies",
                )
                heavy_alcohol = yes_no_radio(
                    "Does the health survey classify your alcohol use as heavy?",
                    help_text=(
                        "This follows the original survey coding. Ask the demo "
                        "operator if you are checking a survey record."
                    ),
                    key="heavy_alcohol",
                )

        with st.expander("4️⃣ Healthcare & wellbeing", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                any_healthcare = yes_no_radio(
                    "Do you have any healthcare coverage?",
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
            "Show screening result",
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
            except RelayClientError as exc:
                show_relay_failure(exc)
            except Exception as exc:
                st.error("Prediction failed because of an unexpected app error.")
                with st.expander("Technical error details for the demo operator"):
                    st.exception(exc)


# =====================================================================
# TAB 2 — CSV input: one or multiple rows
# =====================================================================
with tab_csv:
    st.header("Check one or more rows from a CSV file")
    st.write(
        "Upload a CSV file or paste CSV rows below. You can check one row or "
        "several rows at the same time."
    )

    st.info(
        "For project assessors: the file may contain the 21 required health "
        "fields, with optional `Diabetes_012` reference labels. Reference labels "
        "are used only for comparison and are never sent for screening."
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
                raw_df = parse_uploaded_csv(uploaded)
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

        st.subheader("Review the rows")
        st.write(f"Rows received: **{len(checked_df):,}**")
        preview_df = checked_df.head(20).copy()
        if TARGET_COLUMN in preview_df.columns:
            preview_df[TARGET_COLUMN] = preview_df[TARGET_COLUMN].apply(
                dataset_label_text
            )
            preview_df = preview_df.rename(
                columns={TARGET_COLUMN: "Known sample/reference status"}
            )
        st.dataframe(preview_df, use_container_width=True)

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
            st.success("The CSV is ready to check.")

            confirm_batch = st.checkbox(
                "I understand these are screening-support results, "
                "not medical diagnoses.",
                key="confirm_batch",
            )

            if st.button(
                f"Show results for {len(checked_df)} row(s)",
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
                                        "Follow-up suggested, but sample label says no diabetes"
                                    )
                                elif screening_class == 0 and actual_binary == 1:
                                    output["Comparison note"] = (
                                        "No warning detected, but sample label is prediabetes/diabetes"
                                    )
                                else:
                                    output["Comparison note"] = "Result and sample label agree"

                            output_rows.append(output)

                        except requests.Timeout:
                            output_rows.append(
                                {
                                    "Person": position,
                                    "Screening result": "Could not check",
                                    "Screening score": None,
                                    "What to do": "Retry this row",
                                    "Status": "Error: request timed out",
                                    "_model_class": None,
                                    "_probability": None,
                                    "_threshold": None,
                                    "_latency_seconds": None,
                                }
                            )
                        except RelayClientError as exc:
                            output_rows.append(
                                {
                                    "Person": position,
                                    "Screening result": "Could not check",
                                    "Screening score": None,
                                    "What to do": "Ask the demo operator to check the relay",
                                    "Status": f"Error: {exc}",
                                    "_model_class": None,
                                    "_probability": None,
                                    "_threshold": None,
                                    "_latency_seconds": None,
                                }
                            )
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
                        "What to do",
                    ]
                    friendly_columns = [
                        c for c in friendly_columns if c in results_df.columns
                    ]

                    friendly_df = results_df[friendly_columns].copy()
                    st.dataframe(
                        friendly_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    if "Known status in sample data" in results_df.columns:
                        with st.expander("Sample-label comparison for the project assessor"):
                            comparable = results_df[
                                results_df["Status"].eq("Success")
                            ].copy()

                            st.dataframe(
                                comparable[[
                                    "Person",
                                    "Known status in sample data",
                                    "Matches known status?",
                                    "Comparison note",
                                ]],
                                use_container_width=True,
                                hide_index=True,
                            )

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

                            c1, c2, c3 = st.columns(3)
                            with c1:
                                st.metric("Matched sample label", int(matches))
                            with c2:
                                st.metric("Extra follow-up warnings", int(extra_warnings))
                            with c3:
                                st.metric("Missed warnings", int(missed_warnings))

                            st.info(
                                "A mismatch means the screening result differs from the "
                                "known label for that sample row."
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
    st.header("About this screening check")
    st.write(
        "This academic project looks for health patterns associated with "
        "prediabetes or diabetes. It supports screening conversations; it does "
        "not confirm or rule out a medical condition."
    )
    st.info(
        "A result of **Follow-up suggested** includes patterns associated with "
        "either prediabetes or diabetes. It must not be read as a diabetes diagnosis."
    )

    with st.expander("Technical information for the project assessor"):
        st.subheader("Dataset and output meaning")
        st.markdown(
            """
The application input contract matches
**`diabetes_012_health_indicators_BRFSS2015.csv`** with 21 predictor fields.
The optional original target is `Diabetes_012`: `0` = no diabetes,
`1` = prediabetes, and `2` = diabetes.

The deployed binary target uses `Diabetes_012 > 0`, so class `1` combines
prediabetes and diabetes.

| Model class | User-facing wording |
|---|---|
| 0 | No warning detected |
| 1 | Follow-up suggested |
"""
        )

        st.subheader("Project controls")
        st.markdown(
            """
- The intended-use warning remains visible.
- Acknowledgement is required before a prediction.
- Inputs are validated against the BRFSS coding domains.
- `Diabetes_012` is never sent for inference.
- Errors are shown instead of silently generating a result.
- The relay API key is not written into evidence files.
"""
        )

        st.subheader("Deployment path")
        st.code(
            "User → Streamlit → FastAPI relay → SageMaker Serverless Endpoint "
            "→ XGBoost → Result",
            language="text",
        )
