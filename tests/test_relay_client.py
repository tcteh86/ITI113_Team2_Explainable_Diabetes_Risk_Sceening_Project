"""Tests for the Streamlit-to-FastAPI relay client.

These tests mock HTTP requests and must not require:
- AWS credentials
- a live SageMaker endpoint
- ngrok
- FastAPI running locally
"""

import requests
import pytest

import team02_frontend_core as app


def valid_payload() -> dict:
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


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    @property
    def ok(self):
        return 200 <= self.status_code < 400

    def json(self):
        if isinstance(self._json_data, Exception):
            raise self._json_data
        return self._json_data


def test_check_relay_health_sends_derived_url_auth_and_timeout(monkeypatch):
    captured = {}

    def fake_get(url, headers, timeout):
        captured.update(url=url, headers=headers, timeout=timeout)
        return FakeResponse(
            200,
            {"relay": "ok", "endpoint_status": "InService"},
        )

    monkeypatch.setattr(app.requests, "get", fake_get)

    health = app.check_relay_health(
        "https://example.ngrok-free.dev/predict",
        "temporary-test-key",
    )

    assert captured["url"] == "https://example.ngrok-free.dev/health"
    assert captured["headers"]["x-api-key"] == "temporary-test-key"
    assert captured["timeout"] == 20
    assert health["endpoint_status"] == "InService"


@pytest.mark.parametrize("status_code", [401, 422, 502, 503])
def test_check_relay_health_raises_clear_http_failure(monkeypatch, status_code):
    monkeypatch.setattr(
        app.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(status_code, text="relay error"),
    )

    with pytest.raises(app.RelayClientError) as exc:
        app.check_relay_health(
            "https://example.ngrok-free.dev/predict",
            "key",
        )

    assert exc.value.status_code == status_code
    assert f"HTTP {status_code}" in str(exc.value)


def test_check_relay_health_propagates_timeout(monkeypatch):
    def fake_get(*args, **kwargs):
        raise requests.Timeout("simulated health timeout")

    monkeypatch.setattr(app.requests, "get", fake_get)

    with pytest.raises(requests.Timeout):
        app.check_relay_health(
            "https://example.ngrok-free.dev/predict",
            "key",
        )


def test_check_relay_health_surfaces_malformed_success_json(monkeypatch):
    monkeypatch.setattr(
        app.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(200, ValueError("bad health json")),
    )

    with pytest.raises(app.RelayClientError, match="non-JSON"):
        app.check_relay_health(
            "https://example.ngrok-free.dev/predict",
            "key",
        )


@pytest.mark.parametrize("missing_field", ["relay", "endpoint_status"])
def test_check_relay_health_rejects_missing_required_field(
    monkeypatch,
    missing_field,
):
    body = {"relay": "ok", "endpoint_status": "InService"}
    body.pop(missing_field)
    monkeypatch.setattr(
        app.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(200, body),
    )

    with pytest.raises(app.RelayClientError, match="health response is missing"):
        app.check_relay_health(
            "https://example.ngrok-free.dev/predict",
            "key",
        )


def test_call_relay_sends_exact_url_headers_and_payload(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse(
            200,
            {
                "screening_class": 0,
                "probability": 0.12,
                "decision_threshold": 0.45,
            },
        )

    monkeypatch.setattr(app.requests, "post", fake_post)

    payload = valid_payload()
    result, latency, status_code = app.call_relay(
        " https://example.ngrok-free.dev/predict ",
        "temporary-test-key",
        payload,
    )

    assert captured["url"] == "https://example.ngrok-free.dev/predict"
    assert captured["json"] == payload
    assert captured["headers"]["x-api-key"] == "temporary-test-key"
    assert captured["headers"]["Content-Type"] == "application/json"
    assert captured["headers"]["ngrok-skip-browser-warning"] == "true"
    assert captured["timeout"] == 75

    assert result["screening_class"] == 0
    assert result["probability"] == 0.12
    assert result["decision_threshold"] == 0.45
    assert status_code == 200
    assert latency >= 0


def test_call_relay_accepts_single_result_wrapped_in_list(monkeypatch):
    def fake_post(url, headers, json, timeout):
        return FakeResponse(
            200,
            [
                {
                    "screening_class": 1,
                    "probability": 0.80,
                    "decision_threshold": 0.45,
                }
            ],
        )

    monkeypatch.setattr(app.requests, "post", fake_post)

    result, _, _ = app.call_relay(
        "https://example.ngrok-free.dev/predict",
        "key",
        valid_payload(),
    )

    assert result["screening_class"] == 1
    assert result["probability"] == 0.80


def test_call_relay_rejects_empty_result_list(monkeypatch):
    monkeypatch.setattr(
        app.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(200, []),
    )

    with pytest.raises(RuntimeError, match="empty result list"):
        app.call_relay(
            "https://example.ngrok-free.dev/predict",
            "key",
            valid_payload(),
        )


@pytest.mark.parametrize("missing_field", [
    "screening_class",
    "probability",
    "decision_threshold",
])
def test_call_relay_rejects_missing_required_response_field(
    monkeypatch,
    missing_field,
):
    body = {
        "screening_class": 0,
        "probability": 0.12,
        "decision_threshold": 0.45,
    }
    body.pop(missing_field)

    monkeypatch.setattr(
        app.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(200, body),
    )

    with pytest.raises(RuntimeError, match="Endpoint response is missing"):
        app.call_relay(
            "https://example.ngrok-free.dev/predict",
            "key",
            valid_payload(),
        )


@pytest.mark.parametrize(
    "status_code,detail",
    [
        (401, {"detail": "Invalid relay API key."}),
        (422, {"detail": "Input validation failed."}),
        (502, {"detail": "SageMaker invocation failed."}),
        (503, {"detail": "AWS credentials are not available to the relay."}),
    ],
)
def test_call_relay_surfaces_http_error_details(
    monkeypatch,
    status_code,
    detail,
):
    monkeypatch.setattr(
        app.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(
            status_code,
            detail,
            text=str(detail),
        ),
    )

    with pytest.raises(RuntimeError) as exc:
        app.call_relay(
            "https://example.ngrok-free.dev/predict",
            "key",
            valid_payload(),
        )

    message = str(exc.value)
    assert f"HTTP {status_code}" in message


def test_call_relay_surfaces_non_json_error_body(monkeypatch):
    bad_json = ValueError("not json")

    monkeypatch.setattr(
        app.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(
            502,
            bad_json,
            text="upstream gateway error",
        ),
    )

    with pytest.raises(RuntimeError) as exc:
        app.call_relay(
            "https://example.ngrok-free.dev/predict",
            "key",
            valid_payload(),
        )

    assert "HTTP 502" in str(exc.value)
    assert "upstream gateway error" in exc.value.detail


def test_call_relay_propagates_timeout(monkeypatch):
    def fake_post(*args, **kwargs):
        raise requests.Timeout("simulated timeout")

    monkeypatch.setattr(app.requests, "post", fake_post)

    with pytest.raises(requests.Timeout):
        app.call_relay(
            "https://example.ngrok-free.dev/predict",
            "key",
            valid_payload(),
        )


def test_call_relay_surfaces_success_response_with_invalid_json(monkeypatch):
    monkeypatch.setattr(
        app.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(
            200,
            ValueError("malformed json"),
        ),
    )

    with pytest.raises(app.RelayClientError, match="non-JSON"):
        app.call_relay(
            "https://example.ngrok-free.dev/predict",
            "key",
            valid_payload(),
        )


@pytest.mark.parametrize(
    "field,value,expected_message",
    [
        ("screening_class", 2, "screening_class must be 0 or 1"),
        ("screening_class", 0.5, "screening_class must be 0 or 1"),
        ("probability", -0.01, "probability must be between 0 and 1"),
        ("probability", 1.01, "probability must be between 0 and 1"),
        ("probability", float("nan"), "probability must be between 0 and 1"),
        ("decision_threshold", -0.01, "decision_threshold must be between 0 and 1"),
        ("decision_threshold", 1.01, "decision_threshold must be between 0 and 1"),
    ],
)
def test_call_relay_rejects_invalid_prediction_values(
    monkeypatch,
    field,
    value,
    expected_message,
):
    body = {
        "screening_class": 0,
        "probability": 0.12,
        "decision_threshold": 0.45,
    }
    body[field] = value
    monkeypatch.setattr(
        app.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(200, body),
    )

    with pytest.raises(app.RelayClientError, match=expected_message):
        app.call_relay(
            "https://example.ngrok-free.dev/predict",
            "key",
            valid_payload(),
        )


@pytest.mark.parametrize("method_name", ["get", "post"])
def test_relay_client_explains_ngrok_offline_connection_error(
    monkeypatch,
    method_name,
):
    def offline(*args, **kwargs):
        raise requests.ConnectionError("name resolution failed")

    monkeypatch.setattr(app.requests, method_name, offline)

    with pytest.raises(app.RelayClientError, match="ngrok tunnel may be offline"):
        if method_name == "get":
            app.check_relay_health(
                "https://offline.ngrok-free.dev/predict",
                "key",
            )
        else:
            app.call_relay(
                "https://offline.ngrok-free.dev/predict",
                "key",
                valid_payload(),
            )
