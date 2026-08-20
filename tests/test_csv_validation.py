"""Tests for Team02 CSV parsing and validation.

These tests target the public helpers in team02_frontend_core.py.

The pure core module is imported directly, so importing these tests does not
render Streamlit or require a live application session.
"""

from io import StringIO

import pandas as pd
import pytest

import team02_frontend_core as app


EXPECTED_FEATURE_COLUMNS = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]


def valid_feature_row() -> dict:
    return {
        "HighBP": 1,
        "HighChol": 1,
        "CholCheck": 1,
        "BMI": 30,
        "Smoker": 0,
        "Stroke": 0,
        "HeartDiseaseorAttack": 0,
        "PhysActivity": 1,
        "Fruits": 1,
        "Veggies": 1,
        "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1,
        "NoDocbcCost": 0,
        "GenHlth": 3,
        "MentHlth": 0,
        "PhysHlth": 0,
        "DiffWalk": 0,
        "Sex": 0,
        "Age": 8,
        "Education": 5,
        "Income": 6,
    }


def test_feature_contract_has_21_unique_columns():
    assert app.FEATURE_COLUMNS == EXPECTED_FEATURE_COLUMNS
    assert len(app.FEATURE_COLUMNS) == 21
    assert len(set(app.FEATURE_COLUMNS)) == 21
    assert app.TARGET_COLUMN == "Diabetes_012"
    assert app.TARGET_COLUMN not in app.FEATURE_COLUMNS


def test_validate_accepts_model_only_21_feature_dataframe():
    df = pd.DataFrame([valid_feature_row()])

    checked, errors = app.validate_input_dataframe(df)

    assert errors == []
    assert list(checked.columns) == app.FEATURE_COLUMNS
    assert len(checked) == 1


def test_validate_accepts_original_22_column_dataset_format():
    row = {"Diabetes_012": 2, **valid_feature_row()}
    df = pd.DataFrame([row])

    checked, errors = app.validate_input_dataframe(df)

    assert errors == []
    assert list(checked.columns) == [app.TARGET_COLUMN] + app.FEATURE_COLUMNS
    assert checked.loc[0, app.TARGET_COLUMN] == 2


def test_validate_reports_and_drops_unexpected_extra_columns():
    row = {"Diabetes_012": 0, **valid_feature_row(), "UnexpectedColumn": 999}
    df = pd.DataFrame([row])

    checked, errors = app.validate_input_dataframe(df)

    assert any("Unexpected columns" in error for error in errors)
    assert "UnexpectedColumn" not in checked.columns
    assert list(checked.columns) == [app.TARGET_COLUMN] + app.FEATURE_COLUMNS


def test_validate_reports_missing_required_column():
    row = valid_feature_row()
    row.pop("BMI")
    df = pd.DataFrame([row])

    _, errors = app.validate_input_dataframe(df)

    assert errors
    assert any("Missing required columns" in e and "BMI" in e for e in errors)


@pytest.mark.parametrize(
    "column,bad_value",
    [
        ("HighBP", 2),
        ("HighChol", -1),
        ("CholCheck", 3),
        ("Smoker", 9),
        ("Stroke", 2),
        ("HeartDiseaseorAttack", -1),
        ("PhysActivity", 2),
        ("Fruits", 4),
        ("Veggies", 7),
        ("HvyAlcoholConsump", 2),
        ("AnyHealthcare", 3),
        ("NoDocbcCost", -1),
        ("DiffWalk", 5),
        ("Sex", 2),
    ],
)
def test_validate_rejects_invalid_binary_codes(column, bad_value):
    row = valid_feature_row()
    row[column] = bad_value

    _, errors = app.validate_input_dataframe(pd.DataFrame([row]))

    assert any(column in e and "0 or 1" in e for e in errors)


@pytest.mark.parametrize(
    "column,bad_value",
    [
        ("BMI", 11),
        ("BMI", 99),
        ("GenHlth", 0),
        ("GenHlth", 6),
        ("MentHlth", -1),
        ("MentHlth", 31),
        ("PhysHlth", -1),
        ("PhysHlth", 31),
        ("Age", 0),
        ("Age", 14),
        ("Education", 0),
        ("Education", 7),
        ("Income", 0),
        ("Income", 9),
    ],
)
def test_validate_rejects_values_outside_dataset_contract(column, bad_value):
    row = valid_feature_row()
    row[column] = bad_value

    _, errors = app.validate_input_dataframe(pd.DataFrame([row]))

    assert any(column in e and "must be between" in e for e in errors)


def test_validate_rejects_blank_or_non_numeric_value():
    row = valid_feature_row()
    row["BMI"] = "not-a-number"

    _, errors = app.validate_input_dataframe(pd.DataFrame([row]))

    assert any("Blank or non-numeric" in e and "BMI" in e for e in errors)


@pytest.mark.parametrize(
    "column",
    ["GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"],
)
def test_validate_rejects_fractional_coded_values(column):
    row = valid_feature_row()
    row[column] = float(row[column]) + 0.5

    _, errors = app.validate_input_dataframe(pd.DataFrame([row]))

    assert any(
        column in error and "whole-number coded values" in error
        for error in errors
    )


@pytest.mark.parametrize("target", [-1, 3, 99])
def test_validate_rejects_invalid_diabetes_012_label(target):
    row = {"Diabetes_012": target, **valid_feature_row()}

    _, errors = app.validate_input_dataframe(pd.DataFrame([row]))

    assert any("Diabetes_012 must contain only 0, 1 or 2" in e for e in errors)


def test_parse_pasted_csv_accepts_header_plus_rows():
    row1 = valid_feature_row()
    row2 = valid_feature_row()
    row2["BMI"] = 35

    df = pd.DataFrame([row1, row2])
    pasted = df.to_csv(index=False)

    parsed = app.parse_pasted_csv(pasted)

    assert list(parsed.columns) == app.FEATURE_COLUMNS
    assert len(parsed) == 2
    assert parsed.iloc[1]["BMI"] == 35


def test_parse_pasted_csv_normalises_whitespace_and_bom_in_header():
    row = valid_feature_row()
    header = ",".join(
        ("\ufeff " if index == 0 else " ") + column + " "
        for index, column in enumerate(app.FEATURE_COLUMNS)
    )
    values = ",".join(str(row[column]) for column in app.FEATURE_COLUMNS)

    parsed = app.parse_pasted_csv(header + "\n" + values)
    checked, errors = app.validate_input_dataframe(parsed)

    assert errors == []
    assert list(checked.columns) == app.FEATURE_COLUMNS


def test_parse_pasted_csv_accepts_headerless_21_feature_row():
    row = valid_feature_row()
    values = [row[c] for c in app.FEATURE_COLUMNS]
    pasted = ",".join(str(v) for v in values)

    parsed = app.parse_pasted_csv(pasted)

    assert list(parsed.columns) == app.FEATURE_COLUMNS
    assert len(parsed) == 1


def test_parse_pasted_csv_accepts_headerless_22_column_original_row():
    row = valid_feature_row()
    values = [2] + [row[c] for c in app.FEATURE_COLUMNS]
    pasted = ",".join(str(v) for v in values)

    parsed = app.parse_pasted_csv(pasted)

    assert list(parsed.columns) == [app.TARGET_COLUMN] + app.FEATURE_COLUMNS
    assert len(parsed) == 1
    assert parsed.iloc[0][app.TARGET_COLUMN] == 2


def test_parse_pasted_csv_rejects_empty_text():
    with pytest.raises(ValueError, match="No CSV text"):
        app.parse_pasted_csv("   ")


def test_parse_pasted_csv_rejects_wrong_number_of_columns():
    with pytest.raises(ValueError, match="Expected 21 feature columns or 22 columns"):
        app.parse_pasted_csv("1,2,3,4,5")


def test_validate_rejects_header_only_csv_with_no_rows():
    empty = pd.DataFrame(columns=app.FEATURE_COLUMNS)

    _, errors = app.validate_input_dataframe(empty)

    assert errors == ["No data rows found."]


def test_parse_uploaded_csv_accepts_one_model_only_row():
    csv_file = StringIO(pd.DataFrame([valid_feature_row()]).to_csv(index=False))

    parsed = app.parse_uploaded_csv(csv_file)
    checked, errors = app.validate_input_dataframe(parsed)

    assert errors == []
    assert len(checked) == 1
    assert list(checked.columns) == app.FEATURE_COLUMNS


def test_parse_uploaded_csv_accepts_multiple_original_brfss_rows():
    rows = [
        {app.TARGET_COLUMN: 0, **valid_feature_row()},
        {app.TARGET_COLUMN: 2, **valid_feature_row()},
    ]
    csv_file = StringIO(pd.DataFrame(rows).to_csv(index=False))

    parsed = app.parse_uploaded_csv(csv_file)
    checked, errors = app.validate_input_dataframe(parsed)

    assert errors == []
    assert len(checked) == 2
    assert list(checked.columns) == [app.TARGET_COLUMN] + app.FEATURE_COLUMNS
    assert checked[app.TARGET_COLUMN].tolist() == [0, 2]


def test_parse_pasted_csv_accepts_multiple_headerless_original_rows():
    row = valid_feature_row()
    first = [0] + [row[column] for column in app.FEATURE_COLUMNS]
    second = [2] + [row[column] for column in app.FEATURE_COLUMNS]
    pasted = "\n".join(
        ",".join(str(value) for value in values)
        for values in [first, second]
    )

    parsed = app.parse_pasted_csv(pasted)
    checked, errors = app.validate_input_dataframe(parsed)

    assert errors == []
    assert len(checked) == 2
    assert checked[app.TARGET_COLUMN].tolist() == [0, 2]


@pytest.mark.parametrize(
    "parser,value",
    [
        (app.parse_pasted_csv, 'HighBP,"unterminated'),
        (app.parse_uploaded_csv, StringIO('HighBP,"unterminated')),
    ],
)
def test_csv_parsers_reject_malformed_csv(parser, value):
    with pytest.raises(ValueError, match="Could not read"):
        parser(value)
