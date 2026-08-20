# Team02 Diabetes Screening-Support Prototype

This repository contains the Team02 ITI113 academic demonstration for a
diabetes **screening-support** workflow. It is not a medical device and does not
provide a diagnosis.

The deployed XGBoost model is binary:

- `0` → **No warning detected**
- `1` → **Follow-up suggested**

The positive group combines the original BRFSS prediabetes and diabetes labels.
It must never be described as confirmed diabetes.

## Architecture

```text
User
  → Streamlit
  → HTTPS / ngrok with x-api-key
  → FastAPI relay running in SageMaker Studio/Jupyter
  → boto3
  → SageMaker Serverless Endpoint
  → XGBoost
  → prediction response returned to Streamlit
```

Streamlit does not need or store AWS credentials. The authoritative AWS demo
procedure is
`JupyterNotebook/team02_fastapi_ngrok_relay_sagemaker_endpoint.ipynb`; run it
manually inside SageMaker Studio/Jupyter. Normal frontend work must not modify
that notebook or replace its relay architecture.

More detail is available in:

- `docs/ARCHITECTURE.md`
- `docs/AWS_STREAMLIT_INTEGRATION.md`
- `docs/DEMO_RUNBOOK.md`

## Repository layout

| Path | Purpose |
|---|---|
| `team02_streamlit_app.py` | Senior-friendly Streamlit interface and result rendering |
| `team02_frontend_core.py` | Testable CSV validation, payload construction, and relay client |
| `tests/` | Mocked unit and Streamlit smoke tests; no AWS required |
| `Test_Data/` | Small validation sample for the academic demo |
| `JupyterNotebook/` | Authoritative manual SageMaker/FastAPI/ngrok procedure |
| `aws_reference/` | Reference relay implementation; not a replacement deployment |

## Local installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configuration

Streamlit reads these runtime environment variables:

| Variable | Required | Meaning |
|---|---:|---|
| `RELAY_PREDICT_URL` | Yes for a real demo | Temporary ngrok HTTPS URL ending in `/predict` |
| `RELAY_API_KEY` | Yes for a real demo | Temporary shared secret used in the `x-api-key` header |
| `MAX_BATCH_ROWS` | No | Maximum CSV rows per run; defaults to `100` |

Linux/macOS example:

```bash
export RELAY_PREDICT_URL="https://<temporary-ngrok-host>/predict"
export RELAY_API_KEY="<same-temporary-shared-secret>"
streamlit run team02_streamlit_app.py
```

PowerShell example:

```powershell
$env:RELAY_PREDICT_URL="https://<temporary-ngrok-host>/predict"
$env:RELAY_API_KEY="<same-temporary-shared-secret>"
streamlit run team02_streamlit_app.py
```

Never place actual secrets in source code, documentation, tests, screenshots,
sample environment files, or Git history.

## Using the application

### Check one person

1. Open **Check one person**.
2. Complete the four groups of questions.
3. Acknowledge that the result is screening support, not a diagnosis.
4. Select **Show screening result**.
5. Read the result and suggested next action.
6. Expand technical details only when assessor evidence is required.

### Check CSV rows

The CSV mode supports file upload and pasted CSV text, with one or many rows.
Two schemas are accepted:

1. Model-only: the exact 21 predictor columns.
2. Original BRFSS: `Diabetes_012` plus the exact 21 predictors.

The approved predictor order is:

```text
HighBP,HighChol,CholCheck,BMI,Smoker,Stroke,HeartDiseaseorAttack,
PhysActivity,Fruits,Veggies,HvyAlcoholConsump,AnyHealthcare,NoDocbcCost,
GenHlth,MentHlth,PhysHlth,DiffWalk,Sex,Age,Education,Income
```

`Diabetes_012` is optional reference/test-label information. It may be shown as
known sample status for evaluation, but it is never included in `/predict`
payloads. Empty rows, missing features, unexpected columns, non-numeric values,
invalid binary fields, invalid coded ranges, and malformed CSV are rejected.

## Relay API contract

### Health

```http
GET /health
x-api-key: <temporary-shared-secret>
```

A successful health response is JSON containing at least `relay` and
`endpoint_status`. The endpoint should report `InService` before prediction.

### Prediction

```http
POST /predict
Content-Type: application/json
x-api-key: <temporary-shared-secret>
```

The JSON request contains exactly the 21 predictors. A successful response must
contain:

```json
{
  "screening_class": 0,
  "probability": 0.1034,
  "decision_threshold": 0.45
}
```

Additional metadata is kept in technical sections. Streamlit uses the class and
threshold supplied by the deployed endpoint; it does not choose a separate
operating threshold.

## Error handling

The frontend provides specific guidance for:

- `401` — check that Streamlit and the relay use the same `RELAY_API_KEY`;
- `422` — verify the exact 21 feature names and allowed values;
- `502` — inspect the relay and SageMaker endpoint logs;
- `503` — confirm the SageMaker relay notebook is still running;
- timeout — allow for Serverless Endpoint startup and retry once;
- connection failure — recreate an expired or offline ngrok tunnel;
- non-JSON response — inspect the relay/ngrok response in technical details.

## Automated testing

Frontend tests use mocked HTTP responses. They do not require AWS credentials,
ngrok, a running FastAPI relay, or a live SageMaker endpoint.

```bash
python -m py_compile team02_frontend_core.py team02_streamlit_app.py tests/*.py
python -m pytest -q
```

The suite covers CSV formats and validation, exact payload construction, target
leakage prevention, health and prediction requests, authentication headers,
HTTP failures, timeouts, ngrok-offline behavior, malformed JSON, required
response values, and Streamlit rendering.

Passing mocked tests must not be reported as proof of real AWS integration.

## Manual AWS integration

1. Open SageMaker Studio/Jupyter.
2. Run `JupyterNotebook/team02_fastapi_ngrok_relay_sagemaker_endpoint.ipynb`.
3. Configure a fresh temporary relay key using hidden input.
4. Confirm the SageMaker endpoint is `InService`.
5. Confirm direct SageMaker invocation returns the required prediction fields.
6. Start FastAPI and verify authenticated local `/health` and `/predict`.
7. Start ngrok and verify authenticated public `/health` and `/predict`.
8. Copy the public `/predict` URL to `RELAY_PREDICT_URL` on the Streamlit machine.
9. Configure the same temporary secret as `RELAY_API_KEY` on Streamlit.
10. Start Streamlit and select **Check system connection**.
11. Run one real single-person prediction and one small CSV prediction.
12. Capture sanitized evidence without any secret values.
13. Stop ngrok and FastAPI, clear the temporary environment variables, and
    discard the relay key after the demonstration.

See `docs/DEMO_RUNBOOK.md` for the detailed evidence and troubleshooting steps.

## Security reminders

- Never configure AWS access keys in Streamlit.
- Never commit AWS credentials, ngrok tokens, or relay keys.
- Use a fresh temporary relay key for each demonstration.
- Treat the ngrok URL as temporary and stop the tunnel after use.
- Do not include secrets in downloaded evidence or screenshots.
- Do not report a mocked test as a completed AWS integration test.
