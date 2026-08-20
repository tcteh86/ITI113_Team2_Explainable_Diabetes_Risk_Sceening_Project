"""Senior-friendly Streamlit presentation smoke tests."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def render_app():
    app_path = Path(__file__).parents[1] / "team02_streamlit_app.py"
    return AppTest.from_file(str(app_path), default_timeout=15).run()


def test_app_has_clear_modes_and_non_diagnostic_notice():
    app = render_app()

    assert not app.exception
    assert app.title[0].value == "🩺 Health Screening Check"
    assert [tab.label for tab in app.tabs] == [
        "👤 Check one person",
        "📄 Check a CSV file",
        "ℹ️ About this check",
    ]
    assert "does not provide a medical diagnosis" in app.info[0].value


def test_single_person_action_and_yes_no_controls_are_plain_language():
    app = render_app()

    assert "Show screening result" in [button.label for button in app.button]
    yes_no_radios = [
        radio
        for radio in app.radio
        if list(radio.options) == ["No", "Yes"]
    ]
    assert len(yes_no_radios) == 13


def test_technical_content_is_collapsed_by_default():
    app = render_app()

    assessor_expanders = [
        expander
        for expander in app.expander
        if "assessor" in expander.label.lower()
    ]
    assert assessor_expanders
    assert all(not expander.proto.expanded for expander in assessor_expanders)


def test_app_reads_relay_url_and_key_from_expected_environment_variables(
    monkeypatch,
):
    monkeypatch.setenv(
        "RELAY_PREDICT_URL",
        "https://placeholder.ngrok-free.dev/predict",
    )
    monkeypatch.setenv("RELAY_API_KEY", "test-placeholder-only")

    app = render_app()
    inputs = {item.label: item.value for item in app.text_input}

    assert inputs["Relay predict URL"] == (
        "https://placeholder.ngrok-free.dev/predict"
    )
    assert inputs["Relay API key"] == "test-placeholder-only"


def test_app_does_not_present_a_hard_coded_decision_threshold():
    app = render_app()
    captions = [caption.value for caption in app.caption]

    assert any(
        "decision threshold is supplied by the deployed endpoint" in value
        for value in captions
    )
    assert all("Frozen threshold: 0.45" not in value for value in captions)
