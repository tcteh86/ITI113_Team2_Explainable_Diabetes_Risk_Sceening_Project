"""Pure CSV and relay-client logic for the Team02 Streamlit frontend.

This module deliberately has no Streamlit dependency, so its behavior can be
unit tested without rendering the application or contacting AWS.
"""

import math
import time
from io import StringIO
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import requests


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
INTEGER_CODED_COLUMNS = [column for column in FEATURE_COLUMNS if column != "BMI"]

DATASET_LABELS = {
    0: "No diabetes",
    1: "Prediabetes",
    2: "Diabetes",
}


class RelayClientError(RuntimeError):
    """A relay failure with a safe operator message and optional diagnostics."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        detail: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


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


def _connection_error(exc: requests.ConnectionError) -> RelayClientError:
    return RelayClientError(
        "Could not reach the relay. The ngrok tunnel may be offline or expired.",
        detail=str(exc),
    )


def _http_error(response, action: str) -> RelayClientError:
    messages = {
        401: "Authentication failed (HTTP 401). Check RELAY_API_KEY.",
        422: (
            "The relay rejected the input (HTTP 422). Check all 21 feature "
            "names and values."
        ),
        502: (
            "The relay could not obtain a SageMaker response (HTTP 502). "
            "Check the relay and endpoint logs."
        ),
        503: (
            "The relay service is unavailable (HTTP 503). Confirm the SageMaker "
            "relay notebook is still running."
        ),
    }
    try:
        detail = str(response.json())
    except ValueError:
        detail = response.text
    return RelayClientError(
        messages.get(
            response.status_code,
            f"The relay {action} failed (HTTP {response.status_code}).",
        ),
        status_code=response.status_code,
        detail=detail,
    )


def check_relay_health(relay_url: str, relay_api_key: str):
    """Call the authenticated health route and return its JSON object."""
    try:
        response = requests.get(
            health_url_from_predict(relay_url),
            headers=request_headers(relay_api_key),
            timeout=20,
        )
    except requests.Timeout:
        raise
    except requests.ConnectionError as exc:
        raise _connection_error(exc) from exc
    except requests.RequestException as exc:
        raise RelayClientError(
            "The relay health check could not be completed.",
            detail=str(exc),
        ) from exc

    if not response.ok:
        raise _http_error(response, "health check")
    try:
        health = response.json()
    except ValueError as exc:
        raise RelayClientError(
            "The relay health check returned a non-JSON response.",
            detail=response.text,
        ) from exc
    if not isinstance(health, dict):
        raise RelayClientError(
            "The relay health check returned an unexpected JSON response."
        )
    missing = [key for key in ("relay", "endpoint_status") if key not in health]
    if missing:
        raise RelayClientError(
            "Relay health response is missing: " + ", ".join(missing)
        )
    return health


def dataset_label_text(value) -> str:
    try:
        key = int(float(value))
        return DATASET_LABELS.get(key, f"Unknown ({value})")
    except Exception:
        return ""


def binary_actual_from_dataset_label(value):
    """Match the deployed binary model target: Diabetes_012 > 0."""
    try:
        return int(float(value) > 0)
    except Exception:
        return None


def model_result_text(screening_class: int) -> str:
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
    """Build exactly the approved 21-feature payload, excluding the target."""
    payload = {}
    for col in FEATURE_COLUMNS:
        value = row[col]
        payload[col] = float(value) if col == "BMI" else int(float(value))
    return payload


def validate_input_dataframe(input_df: pd.DataFrame):
    """Validate and select the model features plus optional reference label."""
    errors = []
    df = input_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if df.empty:
        errors.append("No data rows found.")
        return df, errors

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        errors.append("Missing required columns: " + ", ".join(missing))
        return df, errors

    allowed = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    unexpected = [column for column in df.columns if column not in allowed]
    if unexpected:
        errors.append("Unexpected columns: " + ", ".join(unexpected))

    keep = FEATURE_COLUMNS.copy()
    if TARGET_COLUMN in df.columns:
        keep = [TARGET_COLUMN] + keep
    df = df[keep].copy()

    for col in keep:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[FEATURE_COLUMNS].isna().any().any():
        bad = [c for c in FEATURE_COLUMNS if df[c].isna().any()]
        errors.append("Blank or non-numeric values found in: " + ", ".join(bad))

    for col in BINARY_COLUMNS:
        bad = df.loc[~df[col].isin([0, 1]) & df[col].notna(), col]
        if len(bad):
            errors.append(f"{col} must contain only 0 or 1.")

    for col in INTEGER_CODED_COLUMNS:
        values = df[col].dropna()
        if (values % 1 != 0).any():
            errors.append(f"{col} must contain whole-number coded values.")

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
        if df[TARGET_COLUMN].isna().any():
            errors.append(
                "Diabetes_012 contains blank or non-numeric reference values."
            )
        bad = df.loc[
            ~df[TARGET_COLUMN].isin([0, 1, 2]) & df[TARGET_COLUMN].notna(),
            TARGET_COLUMN,
        ]
        if len(bad):
            errors.append("Diabetes_012 must contain only 0, 1 or 2.")

    return df, errors


def parse_pasted_csv(text: str) -> pd.DataFrame:
    """Parse headered or headerless 21/22-column CSV text."""
    text = text.strip()
    if not text:
        raise ValueError("No CSV text was entered.")

    try:
        candidate = pd.read_csv(StringIO(text))
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read pasted CSV: {exc}") from exc
    candidate.columns = [
        str(column).lstrip("\ufeff").strip()
        for column in candidate.columns
    ]
    if set(FEATURE_COLUMNS).issubset(candidate.columns):
        return candidate

    try:
        candidate = pd.read_csv(StringIO(text), header=None)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read pasted CSV: {exc}") from exc
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


def parse_uploaded_csv(source) -> pd.DataFrame:
    """Read a headered uploaded CSV in model-only or original BRFSS format."""
    try:
        return pd.read_csv(source)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read uploaded CSV: {exc}") from exc


def call_relay(relay_url: str, relay_api_key: str, payload: dict):
    """Send one prediction request while preserving the endpoint contract."""
    start = time.perf_counter()
    try:
        response = requests.post(
            relay_url.strip(),
            headers=request_headers(relay_api_key),
            json=payload,
            timeout=75,
        )
    except requests.Timeout:
        raise
    except requests.ConnectionError as exc:
        raise _connection_error(exc) from exc
    except requests.RequestException as exc:
        raise RelayClientError(
            "The prediction request could not be completed.",
            detail=str(exc),
        ) from exc
    latency = time.perf_counter() - start

    if not response.ok:
        raise _http_error(response, "prediction request")

    try:
        result = response.json()
    except ValueError as exc:
        raise RelayClientError(
            "The prediction relay returned a non-JSON response.",
            detail=response.text,
        ) from exc
    if isinstance(result, list):
        if not result:
            raise RelayClientError("The endpoint returned an empty result list.")
        result = result[0]

    return validate_prediction_response(result), latency, response.status_code


def validate_prediction_response(result) -> dict:
    """Validate the deployed endpoint's required three-field response contract."""
    if not isinstance(result, dict):
        raise RelayClientError("The endpoint returned an unexpected JSON response.")

    required = ["screening_class", "probability", "decision_threshold"]
    missing = [key for key in required if key not in result]
    if missing:
        raise RelayClientError(
            "Endpoint response is missing: " + ", ".join(missing)
        )

    try:
        screening_class = int(result["screening_class"])
        probability = float(result["probability"])
        threshold = float(result["decision_threshold"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise RelayClientError(
            "The endpoint returned invalid prediction values."
        ) from exc

    original_class = result["screening_class"]
    if (
        isinstance(original_class, bool)
        or screening_class not in {0, 1}
        or float(original_class) != screening_class
    ):
        raise RelayClientError("screening_class must be 0 or 1.")
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise RelayClientError("probability must be between 0 and 1.")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise RelayClientError("decision_threshold must be between 0 and 1.")

    return {
        **result,
        "screening_class": screening_class,
        "probability": probability,
        "decision_threshold": threshold,
    }
