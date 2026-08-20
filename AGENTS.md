# Team02 ITI113 Codex Development Instructions

## Project

This repository contains the Team02 ITI113 Machine Learning & Operations
diabetes screening-support demonstration.

The deployed ML model is an XGBoost binary classifier hosted as an
AWS SageMaker Serverless Endpoint.

The original BRFSS target is:

- Diabetes_012 = 0: No diabetes
- Diabetes_012 = 1: Prediabetes
- Diabetes_012 = 2: Diabetes

The deployed Team02 model converts the target to binary:

- 0: No diabetes
- 1: Prediabetes OR diabetes

Therefore the application MUST NOT describe model class 1 as confirmed
"Diabetes".

Preferred plain-language UI wording:

- Class 0: "No warning detected"
- Class 1: "Follow-up suggested"

Always make clear that this is an academic screening-support prototype
and not a medical diagnosis.


# Development Scope

## Primary Codex development scope

Codex may normally develop, refactor and test:

- team02_streamlit_app.py
- Streamlit UI/UX
- CSV input handling
- one-row and multi-row prediction workflows
- input validation
- display formatting
- error handling
- non-technical / senior-friendly wording
- automated tests
- README documentation
- docs/
- Streamlit-specific dependency files

The Streamlit application should remain easy for non-technical and
older users to understand.


# AWS Backend Boundary

The following notebook is an AWS deployment/integration component:

JupyterNotebook/team02_fastapi_ngrok_relay_sagemaker_endpoint.ipynb

IMPORTANT:

This notebook is specifically designed to be executed manually inside
AWS SageMaker Studio/Jupyter where the Team02 AWS IAM role and SageMaker
resources are available.

It is NOT part of normal Streamlit application development in Codex.

Do NOT:

- redesign the SageMaker deployment when working on Streamlit UI tasks;
- assume Codex has AWS credentials;
- attempt to create or replace the SageMaker endpoint;
- replace the FastAPI relay architecture;
- put AWS credentials into Streamlit;
- put AWS access keys in GitHub;
- put RELAY_API_KEY values in source control;
- change AWS infrastructure merely to simplify frontend development.

Codex may READ the AWS notebook to understand the integration contract.

Codex may document inconsistencies or recommend changes, but must not
modify AWS deployment behaviour unless the user explicitly requests an
AWS-backend task.


# Backend Source of Truth

For the submitted architecture,
JupyterNotebook/team02_fastapi_ngrok_relay_sagemaker_endpoint.ipynb
is the authoritative AWS demo deployment procedure.

It performs the following:

1. Runs inside AWS SageMaker.
2. Uses the AWS execution role available in SageMaker.
3. Verifies the deployed endpoint.
4. Starts an authenticated FastAPI relay.
5. FastAPI invokes the SageMaker Serverless Endpoint using boto3.
6. ngrok exposes the FastAPI relay temporarily using HTTPS.
7. Streamlit calls the ngrok URL.

The root-level team02_sagemaker_relay.py may be treated as a reference
implementation of the relay contract.

Do not assume it replaces the SageMaker notebook deployment process.


# End-to-End Runtime Architecture

User
  ↓
Streamlit
  ↓ HTTPS + x-api-key
ngrok
  ↓
FastAPI relay running inside SageMaker environment
  ↓ boto3
AWS SageMaker Serverless Endpoint
  ↓
XGBoost
  ↓
Prediction JSON
  ↓
FastAPI
  ↓
Streamlit
  ↓
User


# Streamlit / Backend Contract

Streamlit obtains configuration through environment variables:

RELAY_PREDICT_URL
RELAY_API_KEY

Example:

RELAY_PREDICT_URL=https://<temporary-ngrok-host>/predict

Never commit the actual RELAY_API_KEY.

The corresponding health endpoint is:

GET /health

The prediction endpoint is:

POST /predict

Authentication header:

x-api-key: <temporary shared secret>


# Prediction Input Contract

The model receives exactly these 21 features:

HighBP
HighChol
CholCheck
BMI
Smoker
Stroke
HeartDiseaseorAttack
PhysActivity
Fruits
Veggies
HvyAlcoholConsump
AnyHealthcare
NoDocbcCost
GenHlth
MentHlth
PhysHlth
DiffWalk
Sex
Age
Education
Income

If Diabetes_012 exists in an uploaded CSV, it is reference/test-label
information only.

NEVER send Diabetes_012 to the prediction API.


# Prediction Response Contract

Streamlit expects the SageMaker/relay response to contain at least:

screening_class
probability
decision_threshold

The application may display additional metadata under a technical
details section.

Do not silently invent missing response fields.


# Threshold Governance

The operating threshold belongs to the trained/deployed model lifecycle.

Do NOT independently change the application threshold to make individual
examples appear more accurate.

Streamlit should normally respect:

decision_threshold

returned by the deployed endpoint.

Any proposal to change the operating threshold must be evaluated using
model evidence such as:

- precision
- recall
- F1
- F2
- false positives
- false negatives

and then changed through the model/deployment lifecycle.


# Senior-Friendly UI Requirements

Prioritise:

- simple wording;
- large readable text;
- clear Yes/No choices;
- minimal technical terminology;
- logical grouping of questions;
- clear result hierarchy;
- clear next action;
- accessibility;
- visible intended-use warning.

Technical metadata should be hidden under an expandable assessor /
technical section where possible.

Avoid user-facing terms such as:

- screening_class
- binary target
- threshold optimisation
- API payload
- model class
- inference
- endpoint

unless displayed inside technical details.


# CSV Requirements

Support:

1. original BRFSS CSV rows with Diabetes_012 + 21 predictors;
2. 21-feature model-only CSV;
3. one row;
4. multiple rows;
5. pasted CSV rows where practical.

Validate:

- required features;
- numeric values;
- allowed coded ranges;
- missing values;
- unexpected columns.

Never use Diabetes_012 as an inference feature.


# Testing Rules

Frontend development should not require a live AWS endpoint.

Use mocks for automated tests of:

- health endpoint handling;
- prediction endpoint handling;
- successful response;
- timeout;
- HTTP 401;
- HTTP 422;
- HTTP 502/503;
- malformed JSON;
- CSV validation;
- payload construction;
- one-row prediction;
- batch prediction.

A real AWS integration test is separate and manual.


# Real End-to-End Test

When AWS testing is required, instruct the developer to:

1. Open SageMaker Studio.
2. Run team02_fastapi_ngrok_relay_sagemaker_endpoint.ipynb.
3. Confirm SageMaker endpoint is InService.
4. Confirm local FastAPI /health works.
5. Confirm local FastAPI /predict works.
6. Start ngrok.
7. Confirm public /health works.
8. Copy the public /predict URL.
9. Use the SAME temporary RELAY_API_KEY in Streamlit.
10. Start Streamlit.
11. Run one real prediction.
12. Capture evidence.
13. Stop ngrok and FastAPI after the demo.

Do not pretend that AWS integration has been tested if Codex has only
run mocked tests.


# Security Rules

Never commit:

- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_SESSION_TOKEN
- NGROK_AUTHTOKEN
- RELAY_API_KEY

Do not write secrets into:

- code;
- README;
- screenshots;
- tests;
- sample environment files.

Provide .env.example only with placeholders.


# Quality Rules

Before proposing a completed change:

- run syntax validation;
- run available tests;
- inspect git diff;
- report files changed;
- report tests run;
- report limitations;
- identify anything requiring a real AWS manual test.

Keep changes focused.

Do not modify unrelated ML/MLOps implementation merely while improving
the Streamlit frontend.