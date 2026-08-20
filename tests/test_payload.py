"""Tests for Team02 payload construction and model semantics."""

import pandas as pd
import pytest

import team02_frontend_core as app


def valid_feature_row() -> dict:
    return {
        "HighBP": 1,
        "HighChol": 0,
        "CholCheck": 1,
        "BMI": 27.5,
        "Smoker": 0,
        "Stroke": 0,
        "HeartDiseaseorAttack": 0,
        "PhysActivity": 1,
        "Fruits": 1,
        "Veggies": 1,
        "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1,
        "NoDocbcCost": 0,
        "GenHlth": 2,
        "MentHlth": 3,
        "PhysHlth": 4,
        "DiffWalk": 0,
        "Sex": 1,
        "Age": 7,
        "Education": 5,
        "Income": 6,
    }


def test_normalise_payload_contains_exact_21_features_in_approved_order():
    row = pd.Series(valid_feature_row())

    payload = app.normalise_payload(row)

    assert list(payload.keys()) == app.FEATURE_COLUMNS
    assert len(payload) == 21


def test_normalise_payload_never_includes_target():
    source = {"Diabetes_012": 2, **valid_feature_row()}
    row = pd.Series(source)

    payload = app.normalise_payload(row)

    assert app.TARGET_COLUMN not in payload


def test_multiple_original_rows_never_leak_target_into_prediction_payloads():
    rows = pd.DataFrame([
        {app.TARGET_COLUMN: 0, **valid_feature_row()},
        {app.TARGET_COLUMN: 1, **valid_feature_row()},
        {app.TARGET_COLUMN: 2, **valid_feature_row()},
    ])

    checked, errors = app.validate_input_dataframe(rows)
    payloads = [app.normalise_payload(row) for _, row in checked.iterrows()]

    assert errors == []
    assert len(payloads) == 3
    assert all(list(payload) == app.FEATURE_COLUMNS for payload in payloads)
    assert all(app.TARGET_COLUMN not in payload for payload in payloads)


def test_normalise_payload_uses_float_for_bmi_and_int_for_coded_fields():
    row = pd.Series(valid_feature_row())

    payload = app.normalise_payload(row)

    assert isinstance(payload["BMI"], float)

    for key, value in payload.items():
        if key != "BMI":
            assert isinstance(value, int), f"{key} was not converted to int"


@pytest.mark.parametrize(
    "source_label,expected_binary",
    [
        (0, 0),
        (0.0, 0),
        (1, 1),
        (1.0, 1),
        (2, 1),
        (2.0, 1),
    ],
)
def test_original_dataset_label_maps_to_deployed_binary_target(
    source_label,
    expected_binary,
):
    assert app.binary_actual_from_dataset_label(source_label) == expected_binary


def test_invalid_dataset_label_mapping_returns_none():
    assert app.binary_actual_from_dataset_label("not-a-number") is None


@pytest.mark.parametrize(
    "screening_class,expected",
    [
        (0, "No warning detected"),
        (1, "Follow-up suggested"),
    ],
)
def test_user_facing_result_wording_is_non_diagnostic(screening_class, expected):
    assert app.model_result_text(screening_class) == expected


def test_positive_explanation_does_not_claim_confirmed_diabetes():
    text = app.result_explanation(1).lower()

    assert "does not mean the person has diabetes" in text
    assert "healthcare professional" in text


def test_negative_explanation_does_not_claim_diabetes_is_ruled_out():
    text = app.result_explanation(0).lower()

    assert "does not guarantee" in text


def test_next_step_text_is_plain_language():
    assert app.next_step_text(1) == "Consider follow-up screening"
    assert app.next_step_text(0) == "No model follow-up flag"


def test_request_headers_include_relay_auth_and_ngrok_skip_header():
    headers = app.request_headers("temporary-test-key")

    assert headers["Content-Type"] == "application/json"
    assert headers["x-api-key"] == "temporary-test-key"
    assert headers["ngrok-skip-browser-warning"] == "true"


@pytest.mark.parametrize(
    "predict_url,expected",
    [
        (
            "https://example.ngrok-free.dev/predict",
            "https://example.ngrok-free.dev/health",
        ),
        (
            "http://127.0.0.1:8000/predict",
            "http://127.0.0.1:8000/health",
        ),
        (
            "https://example.ngrok-free.dev/anything",
            "https://example.ngrok-free.dev/health",
        ),
    ],
)
def test_health_url_from_predict(predict_url, expected):
    assert app.health_url_from_predict(predict_url) == expected
