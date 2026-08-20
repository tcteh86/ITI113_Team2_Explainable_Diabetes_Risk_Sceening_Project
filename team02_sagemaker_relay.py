import hmac
import json
import os
import time
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field

AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")
SAGEMAKER_ENDPOINT = os.getenv(
    "SAGEMAKER_ENDPOINT",
    "team02-diabetes-risk-test",
)
RELAY_API_KEY = os.getenv("RELAY_API_KEY", "")

if not RELAY_API_KEY:
    raise RuntimeError(
        "RELAY_API_KEY is required. Set a strong random value before starting the relay."
    )

app = FastAPI(
    title="Team02 SageMaker Relay",
    version="1.0",
    description=(
        "Minimal authenticated relay for the Team02 ITI113 diabetes "
        "screening-support SageMaker endpoint."
    ),
)

api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

class DiabetesFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    HighBP: int = Field(ge=0, le=1)
    HighChol: int = Field(ge=0, le=1)
    CholCheck: int = Field(ge=0, le=1)
    BMI: float = Field(gt=0)
    Smoker: int = Field(ge=0, le=1)
    Stroke: int = Field(ge=0, le=1)
    HeartDiseaseorAttack: int = Field(ge=0, le=1)
    PhysActivity: int = Field(ge=0, le=1)
    Fruits: int = Field(ge=0, le=1)
    Veggies: int = Field(ge=0, le=1)
    HvyAlcoholConsump: int = Field(ge=0, le=1)
    AnyHealthcare: int = Field(ge=0, le=1)
    NoDocbcCost: int = Field(ge=0, le=1)
    GenHlth: int = Field(ge=1, le=5)
    MentHlth: int = Field(ge=0, le=30)
    PhysHlth: int = Field(ge=0, le=30)
    DiffWalk: int = Field(ge=0, le=1)
    Sex: int = Field(ge=0, le=1)
    Age: int = Field(ge=1, le=13)
    Education: int = Field(ge=1, le=6)
    Income: int = Field(ge=1, le=8)

def require_api_key(api_key: str | None = Depends(api_key_header)) -> None:
    if api_key is None or not hmac.compare_digest(api_key, RELAY_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid relay API key.")

runtime = boto3.client(
    "sagemaker-runtime",
    region_name=AWS_REGION,
    config=Config(
        connect_timeout=10,
        read_timeout=70,
        retries={"max_attempts": 2, "mode": "standard"},
    ),
)
sagemaker = boto3.client(
    "sagemaker",
    region_name=AWS_REGION,
    config=Config(
        connect_timeout=10,
        read_timeout=20,
        retries={"max_attempts": 2, "mode": "standard"},
    ),
)

@app.get("/health")
def health(_: None = Depends(require_api_key)) -> dict[str, Any]:
    try:
        desc = sagemaker.describe_endpoint(EndpointName=SAGEMAKER_ENDPOINT)
        return {
            "relay": "ok",
            "endpoint_name": SAGEMAKER_ENDPOINT,
            "endpoint_status": desc.get("EndpointStatus"),
            "region": AWS_REGION,
        }
    except NoCredentialsError:
        raise HTTPException(
            status_code=503,
            detail="AWS credentials are not available to the relay.",
        )
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "ClientError")
        raise HTTPException(
            status_code=502,
            detail=f"SageMaker health check failed: {code}",
        )

@app.post("/predict")
def predict(
    features: DiabetesFeatures,
    _: None = Depends(require_api_key),
) -> Any:
    payload = features.model_dump()

    try:
        started = time.perf_counter()
        response = runtime.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(payload).encode("utf-8"),
        )
        elapsed_ms = (time.perf_counter() - started) * 1000

        raw = response["Body"].read().decode("utf-8")
        result = json.loads(raw)

        # Keep the deployed endpoint's JSON contract intact.
        # Add only a response header for operational visibility.
        from fastapi.responses import JSONResponse
        return JSONResponse(
            content=result,
            headers={"x-team02-sagemaker-latency-ms": f"{elapsed_ms:.1f}"},
        )

    except NoCredentialsError:
        raise HTTPException(
            status_code=503,
            detail="AWS credentials are not available to the relay.",
        )
    except (BotoCoreError, ClientError) as exc:
        if isinstance(exc, ClientError):
            code = exc.response.get("Error", {}).get("Code", "ClientError")
        else:
            code = type(exc).__name__
        raise HTTPException(
            status_code=502,
            detail=f"SageMaker invocation failed: {code}",
        )
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=502,
            detail="SageMaker returned a non-JSON response.",
        )
