# Team02 Demo Runbook

## 1. Objective

This runbook provides the exact sequence for demonstrating:

```text
Streamlit -> ngrok -> FastAPI -> AWS SageMaker -> XGBoost -> Streamlit result
```

It is intended for the ITI113 final demo and for controlled end-to-end testing.

---

## 2. Roles

### Streamlit side

Runs on the developer/demo computer.

Responsible for:

- user form
- CSV input
- result display
- frontend evidence

### AWS side

Runs inside AWS SageMaker Studio/Jupyter using:

`JupyterNotebook/team02_fastapi_ngrok_relay_sagemaker_endpoint.ipynb`

Responsible for:

- AWS identity
- SageMaker endpoint connectivity
- FastAPI relay
- ngrok tunnel

---

## 3. Before the Demo

Confirm:

- [ ] GitHub repository is up to date.
- [ ] `team02_streamlit_app.py` is the approved version.
- [ ] No secrets are committed.
- [ ] `.env`, Streamlit secrets, AWS keys and ngrok token are excluded from Git.
- [ ] AWS SageMaker endpoint exists.
- [ ] You know the expected AWS region.
- [ ] ngrok authentication is available.
- [ ] A new temporary `RELAY_API_KEY` can be generated.
- [ ] Test CSV is available.
- [ ] Browser is available for Streamlit.

---

## 4. Generate a Fresh Relay API Key

On a trusted machine:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Copy the generated value temporarily.

Do not paste it into:

- GitHub
- screenshots
- report
- README
- chat
- committed notebook output

Use the same value on AWS and Streamlit.

---

## 5. Start the AWS Side

Open SageMaker Studio/Jupyter.

Open:

`JupyterNotebook/team02_fastapi_ngrok_relay_sagemaker_endpoint.ipynb`

Run the setup cells in order.

---

## 6. Verify AWS Identity

Expected evidence:

- AWS account/session identity resolves;
- region is correct;
- SageMaker execution role/session is available.

Do not continue if AWS credentials are unavailable.

---

## 7. Verify SageMaker Endpoint

Check the configured endpoint.

Expected:

```text
Endpoint status: InService
```

Recommended endpoint:

```text
team02-diabetes-risk-test
```

Use the actual deployed endpoint if the project naming changes.

---

## 8. Test Direct SageMaker Inference

Run the direct endpoint smoke test in the AWS notebook.

Expected:

- valid JSON response;
- probability returned;
- screening class returned;
- decision threshold returned;
- model/data metadata consistent.

This proves:

```text
SageMaker notebook -> SageMaker endpoint
```

before FastAPI/ngrok is introduced.

---

## 9. Start FastAPI Locally in SageMaker

Start the relay on:

```text
http://127.0.0.1:8000
```

The same temporary `RELAY_API_KEY` must be configured.

---

## 10. Test Local FastAPI Root

Expected:

```text
GET http://127.0.0.1:8000/
HTTP 200
```

Typical response:

```json
{
  "status": "running",
  "service": "Team02 Diabetes SageMaker Relay"
}
```

---

## 11. Test Local Health

Call:

```text
GET http://127.0.0.1:8000/health
x-api-key: <temporary key>
```

Expected:

```text
HTTP 200
endpoint_status = InService
```

This proves:

```text
FastAPI -> AWS SageMaker control plane
```

---

## 12. Test Local Prediction

Call:

```text
POST http://127.0.0.1:8000/predict
```

with the valid 21-feature payload.

Expected:

```text
HTTP 200
```

and a real prediction.

This proves:

```text
FastAPI -> SageMaker Runtime -> model
```

before ngrok is introduced.

---

## 13. Start ngrok

Start a tunnel to:

```text
http://127.0.0.1:8000
```

Example output:

```text
https://example.ngrok-free.dev
```

---

## 14. Verify ngrok Process

Confirm:

- [ ] ngrok process is running.
- [ ] one active tunnel exists.
- [ ] tunnel target is port 8000.
- [ ] public root returns HTTP 200.

Expected architecture:

```text
https://example.ngrok-free.dev
-> http://localhost:8000
```

---

## 15. Test Public Health

Call:

```text
https://example.ngrok-free.dev/health
```

with:

```text
x-api-key: <temporary key>
ngrok-skip-browser-warning: true
```

Expected:

```text
HTTP 200
```

---

## 16. Test Public Prediction

Call:

```text
https://example.ngrok-free.dev/predict
```

Expected:

```text
HTTP 200
```

and a real SageMaker prediction.

At this point the backend integration is ready.

---

## 17. Configure Streamlit

On the Streamlit machine:

```powershell
$env:RELAY_PREDICT_URL="https://example.ngrok-free.dev/predict"
$env:RELAY_API_KEY="<SAME_TEMPORARY_KEY>"
$env:MAX_BATCH_ROWS="100"
```

Do not configure AWS credentials on the Streamlit machine.

---

## 18. Install Frontend Dependencies

Recommended:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements-streamlit.txt
```

---

## 19. Run Automated Tests

Before the live demo:

```powershell
pytest -q
```

Expected:

```text
all frontend/unit tests pass
```

These tests are not a substitute for the live AWS integration test.

---

## 20. Start Streamlit

```powershell
streamlit run team02_streamlit_app.py
```

Open the URL displayed by Streamlit.

---

## 21. Test Connection from Streamlit

Use the technical/operator connection control.

Expected:

```text
System is ready
Endpoint status: InService
```

If this fails, check `/health` directly before changing Streamlit code.

---

## 22. Single-Person Demo

Use one representative input.

Explain to the assessor:

- 21 BRFSS-derived features are sent;
- `Diabetes_012` is not sent;
- the app calls the FastAPI relay;
- FastAPI calls SageMaker;
- SageMaker returns probability/class/threshold;
- the UI converts the raw class into senior-friendly wording.

Expected user wording:

```text
No warning detected
```

or:

```text
Follow-up suggested
```

Do not say:

```text
You have diabetes
```

because the positive class combines prediabetes and diabetes.

---

## 23. CSV Batch Demo

Use:

`Test_Data/team02_validation_sample_12rows_balanced_labels.csv`

Demonstrate:

- file upload;
- validation;
- multiple rows;
- real endpoint requests;
- result table;
- optional comparison with known `Diabetes_012` labels;
- downloadable result evidence.

Explain that model-label mismatches are model errors, not automatically application bugs.

---

## 24. Evidence to Capture

Capture only non-secret evidence.

Recommended:

- [ ] SageMaker endpoint `InService`
- [ ] FastAPI local `/health` = 200
- [ ] FastAPI local `/predict` = 200
- [ ] ngrok diagnostic showing active tunnel
- [ ] public `/health` = 200
- [ ] public `/predict` = 200
- [ ] Streamlit single prediction
- [ ] Streamlit batch results
- [ ] result/evidence JSON
- [ ] architecture diagram
- [ ] Git commit / repository link

Do not show the relay key.

---

## 25. Demo Talking Points

A concise explanation:

> The Streamlit application is developed independently from the AWS serving environment. It does not contain AWS credentials. During the demo, the AWS SageMaker notebook starts an authenticated FastAPI relay and exposes it temporarily through ngrok. Streamlit sends only the 21 approved model features over HTTPS to the relay. FastAPI uses the SageMaker execution role to invoke the deployed XGBoost Serverless Endpoint and returns the prediction to Streamlit.

Governance explanation:

> The UI is intentionally labelled as screening support rather than diagnosis. The deployed binary model combines prediabetes and diabetes into the positive class. The model threshold is controlled by the deployed model lifecycle rather than being independently changed in the frontend.

---

## 26. Troubleshooting Matrix

| Symptom | Likely cause | First action |
|---|---|---|
| Local FastAPI root fails | Uvicorn/FastAPI not running | Restart relay process |
| Local `/health` = 401 | Wrong relay key | Make AWS and client key identical |
| Local `/health` = 503 | AWS credentials unavailable | Confirm SageMaker IAM session |
| `/predict` = 422 | Invalid payload | Check exact 21 features/ranges |
| `/predict` = 502 | SageMaker invocation error | Inspect endpoint/logs |
| ngrok `ERR_NGROK_3200` | Tunnel offline | Confirm ngrok process/tunnel |
| Public root = 404 ngrok error | stale/offline tunnel | Recreate ngrok |
| Streamlit connection fails but public health works | client URL/key issue | Check env vars |
| First request is slow | Serverless cold start | Retry once after health check |
| Batch row mismatch | model prediction differs from label | Treat as model error analysis |

---

## 27. Shutdown Procedure

After the demo:

- [ ] stop Streamlit if no longer required;
- [ ] stop ngrok;
- [ ] stop FastAPI/Uvicorn;
- [ ] clear temporary environment variables;
- [ ] rotate/discard temporary relay secret;
- [ ] avoid leaving unnecessary AWS endpoints running if project policy requires cleanup.

Do not delete registered models or project evidence merely as part of routine shutdown.

---

## 28. Final Assessment Evidence Statement

If all live checks pass, the project may state that it demonstrated:

```text
Streamlit -> authenticated FastAPI/ngrok relay
-> SageMaker Serverless Endpoint -> real XGBoost inference
```

Do not describe mocked unit tests as production integration evidence.
