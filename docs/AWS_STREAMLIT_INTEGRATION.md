# AWS–Streamlit Integration Guide

## 1. Purpose

This document defines how the separately developed Streamlit application connects to the Team02 AWS SageMaker inference system.

The key architectural rule is:

> `team02_fastapi_ngrok_relay_sagemaker_endpoint.ipynb` is an AWS deployment/integration notebook and is not part of normal Streamlit coding work in Codex.

It should be retained in the repository because it documents and demonstrates the real backend integration used by the Streamlit application.

---

## 2. Scope Boundary for Codex

### Codex may normally modify

- `team02_streamlit_app.py`
- frontend helper modules
- CSV validation
- API client code
- tests
- frontend documentation
- senior-friendly UX

### Codex should not normally modify

`JupyterNotebook/team02_fastapi_ngrok_relay_sagemaker_endpoint.ipynb`

The notebook is executed in AWS SageMaker where:

- the AWS execution role is available;
- the SageMaker endpoint can be invoked;
- FastAPI is started;
- ngrok is created;
- a temporary public relay URL is produced.

Codex may inspect the notebook to verify the integration contract.

---

## 3. Integration Architecture

```text
Developer / End User
        |
        v
team02_streamlit_app.py
        |
        | RELAY_PREDICT_URL
        | x-api-key = RELAY_API_KEY
        v
https://<temporary-ngrok-host>/predict
        |
        v
ngrok tunnel
        |
        v
http://127.0.0.1:8000/predict
        |
        v
FastAPI relay in SageMaker
        |
        | boto3 invoke_endpoint()
        v
team02-diabetes-risk-test
        |
        v
XGBoost model
```

---

## 4. Required Environment Variables

### On the Streamlit machine

```text
RELAY_PREDICT_URL=https://<ngrok-host>/predict
RELAY_API_KEY=<temporary-shared-secret>
MAX_BATCH_ROWS=100
```

### In the AWS SageMaker relay session

```text
AWS_REGION=ap-southeast-1
SAGEMAKER_ENDPOINT=team02-diabetes-risk-test
RELAY_API_KEY=<same-temporary-shared-secret>
NGROK_AUTHTOKEN=<your-ngrok-token>
```

The exact endpoint name may change between project environments. Use the endpoint verified by the AWS notebook.

---

## 5. Generate a Relay API Key

`RELAY_API_KEY` is not obtained from AWS.

Generate it yourself.

Example PowerShell:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Use the same value:

- in the FastAPI relay session; and
- in the Streamlit process.

Do not commit the generated value.

---

## 6. Run the AWS Backend

Open:

`JupyterNotebook/team02_fastapi_ngrok_relay_sagemaker_endpoint.ipynb`

inside AWS SageMaker Studio/Jupyter.

Execute the notebook in order.

The notebook should prove the following before Streamlit is connected:

1. AWS identity is available.
2. SageMaker endpoint exists.
3. SageMaker endpoint status is `InService`.
4. Direct endpoint inference works.
5. FastAPI starts locally.
6. Local `GET /health` returns HTTP 200.
7. Local `POST /predict` returns HTTP 200.
8. ngrok process is alive.
9. ngrok tunnel points to local FastAPI.
10. Public FastAPI root is reachable.
11. Public `GET /health` returns HTTP 200.
12. Public `POST /predict` returns HTTP 200.

Only after these checks pass should the ngrok prediction URL be copied into the Streamlit environment.

---

## 7. Expected ngrok URLs

Example only:

```text
Public root:
https://example.ngrok-free.dev

Health:
https://example.ngrok-free.dev/health

Prediction:
https://example.ngrok-free.dev/predict
```

The ngrok hostname is temporary.

Do not hard-code a previous hostname into source control.

---

## 8. Streamlit Connection

### PowerShell example

```powershell
$env:RELAY_PREDICT_URL="https://example.ngrok-free.dev/predict"
$env:RELAY_API_KEY="<TEMPORARY_SHARED_SECRET>"
$env:MAX_BATCH_ROWS="100"

streamlit run team02_streamlit_app.py
```

### Command Prompt example

```cmd
set RELAY_PREDICT_URL=https://example.ngrok-free.dev/predict
set RELAY_API_KEY=<TEMPORARY_SHARED_SECRET>
set MAX_BATCH_ROWS=100

streamlit run team02_streamlit_app.py
```

---

## 9. Health Check

The Streamlit application derives:

```text
/predict -> /health
```

and calls:

```http
GET /health
x-api-key: <RELAY_API_KEY>
```

Expected:

```json
{
  "relay": "ok",
  "endpoint_name": "team02-diabetes-risk-test",
  "endpoint_status": "InService",
  "region": "ap-southeast-1"
}
```

The application should treat `InService` as ready.

---

## 10. Prediction Payload

The request must contain exactly the 21 approved features.

Example:

```json
{
  "HighBP": 1,
  "HighChol": 1,
  "CholCheck": 1,
  "BMI": 30.0,
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
  "Income": 6
}
```

Never include:

```text
Diabetes_012
```

in the prediction request.

If it is present in a CSV, use it only to compare the model prediction with known sample data.

---

## 11. Prediction Response

Minimum fields expected by Streamlit:

```json
{
  "screening_class": 0,
  "probability": 0.1034,
  "decision_threshold": 0.45
}
```

The endpoint may return additional metadata such as:

- model family
- candidate/version
- data version
- intended-use information

Do not require optional metadata unless it is part of the agreed contract.

---

## 12. Senior-Friendly Rendering

Do not show the raw class as the primary result.

Recommended mapping:

```text
screening_class = 0
-> No warning detected

screening_class = 1
-> Follow-up suggested
```

The positive class combines prediabetes and diabetes.

Do not change it to "You have diabetes."

---

## 13. Batch CSV Integration

Accepted frontend formats:

### Original dataset format

```text
Diabetes_012 + 21 predictors
```

### Model-only format

```text
21 predictors
```

For every row:

1. validate all 21 features;
2. remove `Diabetes_012` from the request;
3. construct the payload in approved feature order;
4. call `/predict`;
5. collect the response;
6. optionally compare against the source label;
7. show senior-friendly results;
8. preserve technical evidence in downloadable output if required.

---

## 14. Error Handling Contract

### HTTP 401

Meaning:

```text
RELAY_API_KEY mismatch
```

Action:

- verify the same key is used by Streamlit and FastAPI;
- do not change AWS credentials.

### HTTP 422

Meaning:

```text
FastAPI input validation rejected the request
```

Action:

- inspect missing/extra/invalid fields;
- verify 21-feature payload.

### HTTP 502

Meaning:

```text
FastAPI reached AWS but SageMaker invocation/response failed
```

Action:

- inspect AWS endpoint logs/status;
- verify endpoint contract.

### HTTP 503

Possible meaning:

```text
AWS credentials unavailable to relay
```

Action:

- confirm relay is running inside the intended SageMaker environment;
- confirm IAM execution role is available.

### ngrok `ERR_NGROK_3200`

Meaning:

```text
ngrok public endpoint is offline
```

Action:

- confirm local FastAPI is still HTTP 200;
- confirm ngrok process is alive;
- confirm active tunnel target;
- recreate the temporary tunnel if necessary.

### Timeout

Possible cause:

- SageMaker Serverless cold start;
- ngrok/network delay.

Action:

- check `/health`;
- retry once;
- inspect backend logs if repeated.

---

## 15. Development Testing Without AWS

Codex / pytest should mock `requests.post` and `requests.get`.

Unit tests should verify:

- URL construction;
- headers;
- exact feature payload;
- target leakage prevention;
- success response handling;
- list response handling if supported;
- missing response fields;
- HTTP errors;
- timeouts;
- malformed CSV;
- invalid feature values.

Automated frontend development should not require real AWS credentials.

---

## 16. Real Integration Evidence

For ITI113 evidence, capture a real end-to-end run:

```text
Streamlit
-> public ngrok endpoint
-> FastAPI relay
-> SageMaker endpoint
-> real XGBoost response
-> Streamlit result
```

Useful evidence:

- SageMaker endpoint `InService`
- local FastAPI health HTTP 200
- public relay health HTTP 200
- public prediction HTTP 200
- Streamlit result screenshot
- non-secret evidence JSON
- latency
- endpoint/model/data metadata

Never include secret values in evidence.

---

## 17. Shutdown

After the demo:

1. stop Streamlit if no longer needed;
2. stop ngrok tunnel;
3. stop FastAPI/Uvicorn process;
4. optionally delete temporary endpoint resources only if the approved MLOps procedure requires it;
5. invalidate/rotate temporary relay secrets where appropriate.

The AWS notebook remains the authoritative backend run procedure.
