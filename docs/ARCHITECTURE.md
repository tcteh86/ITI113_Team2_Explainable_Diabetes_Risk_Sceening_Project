# Team02 System Architecture

## 1. Purpose

This document describes the runtime and development architecture for the Team02 ITI113 diabetes screening-support prototype.

The system intentionally separates:

1. **Streamlit application development**
2. **AWS SageMaker model serving**
3. **FastAPI relay execution inside the AWS environment**
4. **Temporary ngrok exposure used for demonstration**

This separation is important for security, reproducibility, and MLOps traceability.

---

## 2. System Boundary

### Streamlit development scope

The normal ChatGPT Codex / GitHub development scope is:

- `team02_streamlit_app.py`
- Streamlit UI/UX
- senior-friendly wording
- CSV upload and pasted-row handling
- one-row and multi-row prediction workflows
- input validation
- response rendering
- frontend error handling
- tests
- frontend documentation

### AWS backend scope

The AWS integration notebook is:

`JupyterNotebook/team02_fastapi_ngrok_relay_sagemaker_endpoint.ipynb`

This notebook is specifically intended to be executed in **AWS SageMaker Studio/Jupyter**.

It is not part of normal Streamlit frontend development in Codex.

Codex may read the notebook to understand the API and deployment contract, but it should not change AWS deployment behaviour unless an AWS-backend task is explicitly requested.

---

## 3. Runtime Architecture

```text
User
  |
  v
Streamlit application
  |
  | HTTPS
  | x-api-key: <temporary RELAY_API_KEY>
  v
ngrok temporary public URL
  |
  v
FastAPI relay
(running in the AWS SageMaker notebook environment)
  |
  | boto3 / SageMaker Runtime API
  v
AWS SageMaker Serverless Endpoint
  |
  v
XGBoost model
  |
  v
Prediction JSON
  |
  v
FastAPI -> ngrok -> Streamlit -> User
```

The Streamlit application does **not** require AWS access keys.

The FastAPI relay uses the AWS IAM execution context available in SageMaker.

---

## 4. Component Responsibilities

| Component | Runtime | Responsibility |
|---|---|---|
| Streamlit | Developer/demo PC | User interaction, CSV parsing, validation, result presentation |
| ngrok | SageMaker notebook session | Temporary HTTPS bridge to FastAPI |
| FastAPI relay | SageMaker notebook environment | Authentication, input contract enforcement, SageMaker invocation |
| boto3 | SageMaker notebook environment | Calls the SageMaker Runtime API |
| SageMaker Serverless Endpoint | AWS | Hosts the deployed inference artefact |
| XGBoost | SageMaker endpoint | Produces model probability and screening class |
| MLflow / Model Registry | AWS MLOps environment | Experiment and model lifecycle evidence |

---

## 5. Model Input Contract

The deployed model accepts exactly 21 features, in the approved order:

1. `HighBP`
2. `HighChol`
3. `CholCheck`
4. `BMI`
5. `Smoker`
6. `Stroke`
7. `HeartDiseaseorAttack`
8. `PhysActivity`
9. `Fruits`
10. `Veggies`
11. `HvyAlcoholConsump`
12. `AnyHealthcare`
13. `NoDocbcCost`
14. `GenHlth`
15. `MentHlth`
16. `PhysHlth`
17. `DiffWalk`
18. `Sex`
19. `Age`
20. `Education`
21. `Income`

The original dataset column `Diabetes_012` is a **reference/test label only**.

It must never be sent to `/predict`.

---

## 6. Model Output Semantics

The original BRFSS dataset uses:

- `Diabetes_012 = 0`: No diabetes
- `Diabetes_012 = 1`: Prediabetes
- `Diabetes_012 = 2`: Diabetes

The deployed Team02 model is binary because the modelling pipeline maps:

```text
Diabetes_012 > 0 -> positive class
```

Therefore:

- model class `0`: no diabetes warning pattern
- model class `1`: prediabetes/diabetes screening group

The application must not present model class `1` as confirmed "Diabetes".

Preferred senior-friendly wording:

- **No warning detected**
- **Follow-up suggested**

The application must also state that the output is screening support, not a medical diagnosis.

---

## 7. API Contract

### Health

```text
GET /health
```

Authentication:

```text
x-api-key: <RELAY_API_KEY>
```

Expected successful response:

```json
{
  "relay": "ok",
  "endpoint_name": "team02-diabetes-risk-test",
  "endpoint_status": "InService",
  "region": "ap-southeast-1"
}
```

### Prediction

```text
POST /predict
```

Headers:

```text
Content-Type: application/json
x-api-key: <RELAY_API_KEY>
ngrok-skip-browser-warning: true
```

Minimum expected response fields:

```json
{
  "screening_class": 0,
  "probability": 0.1034,
  "decision_threshold": 0.45
}
```

Additional model/data metadata may also be returned.

The Streamlit app should preserve and display extra metadata only in an assessor/technical section.

---

## 8. Environment Separation

### Streamlit environment

Typical dependencies:

- Streamlit
- pandas
- requests

The Streamlit process uses:

```text
RELAY_PREDICT_URL
RELAY_API_KEY
MAX_BATCH_ROWS
```

### AWS relay environment

Typical dependencies:

- FastAPI
- Uvicorn
- boto3
- pydantic
- pyngrok
- requests

The AWS relay process uses:

```text
AWS_REGION
SAGEMAKER_ENDPOINT
RELAY_API_KEY
NGROK_AUTHTOKEN
```

AWS credentials should come from the SageMaker IAM role/session, not from the Streamlit machine.

---

## 9. Security Boundary

The Streamlit application must never contain or require:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_SESSION_TOKEN
```

Do not commit:

```text
RELAY_API_KEY
NGROK_AUTHTOKEN
AWS credentials
```

The temporary relay key is shared only between Streamlit and FastAPI.

It is not an AWS credential.

---

## 10. Threshold Governance

The operating threshold belongs to the deployed model lifecycle.

Streamlit should normally respect:

```text
decision_threshold
```

returned by the endpoint.

Do not independently change the frontend threshold merely to improve a few visible examples.

A threshold change should be justified using evaluation evidence such as:

- precision
- recall
- F1
- F2
- false positives
- false negatives
- business/screening risk

and then propagated through evaluation, model registration, approval, and deployment.

---

## 11. Development Architecture

Automated frontend tests should not require AWS.

```text
Codex / pytest
     |
     v
Streamlit/core logic
     |
     v
Mocked FastAPI responses
```

Real integration testing is separate:

```text
Streamlit
     |
     v
ngrok
     |
     v
FastAPI in SageMaker
     |
     v
Real SageMaker endpoint
```

A mocked test must never be reported as evidence that the real SageMaker integration was executed.

---

## 12. MLOps / Governance Relevance

This architecture supports:

### ITI113 C — MLOps & Deployment

- clear deployment architecture
- separation of frontend and inference serving
- explicit API contract
- AWS-hosted model serving
- reproducible integration procedure

### ITI113 E — AI Governance

- no AWS credentials in the user application
- human-facing intended-use notice
- explicit model limitations
- prevention of target leakage
- controlled model threshold
- clear human oversight / follow-up wording

### Reproducibility and auditability

The system can be traced as:

```text
Versioned data
-> preprocessing
-> training
-> evaluation
-> MLflow experiment
-> model package
-> approval
-> SageMaker endpoint
-> FastAPI relay
-> Streamlit application
-> user-visible result
```
