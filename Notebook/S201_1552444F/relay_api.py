"""Authenticated Team02 relay for the final SageMaker champion endpoint."""

import hmac
import json
import math
import os
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, NoCredentialsError
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field

REGION = os.environ.get("AWS_REGION", "ap-southeast-1")
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "team02-diabetes-risk")
RELAY_API_KEY = os.environ.get("RELAY_API_KEY", "")

EXPECTED_MODEL_TYPE = os.environ.get("EXPECTED_MODEL_TYPE", "XGBoost")
EXPECTED_MODEL_RUN_ID = os.environ.get(
    "EXPECTED_MODEL_RUN_ID",
    "057bb58d81564c0db9a628b224eab8f9",
)
EXPECTED_DECISION_THRESHOLD = float(
    os.environ.get("EXPECTED_DECISION_THRESHOLD", "0.50")
)
EXPECTED_CONFIG_VERSION = os.environ.get(
    "EXPECTED_CONFIG_VERSION",
    "2026-08-22-cross-model-champion-v5",
)
EXPECTED_DATA_VERSION = os.environ.get(
    "EXPECTED_DATA_VERSION",
    "brfss2015-diabetes-binary-6244bec277fe",
)

EXPECTED_RESPONSE_FIELDS = {
    "screening_class",
    "screening_category",
    "probability",
    "decision_threshold",
    "model_type",
    "data_version",
    "model_run_id",
    "config_version",
    "intended_use_notice",
}

if not RELAY_API_KEY:
    raise RuntimeError(
        "RELAY_API_KEY is missing. Set it before starting the FastAPI relay."
    )

app = FastAPI(
    title="Team02 Diabetes Final-Champion SageMaker Relay",
    version="2.0",
    description=(
        "Authenticated FastAPI relay for the Team02 ITI113 diabetes "
        "screening-support final SageMaker champion endpoint."
    ),
)

api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

runtime = boto3.client(
    "sagemaker-runtime",
    region_name=REGION,
    config=Config(
        connect_timeout=10,
        read_timeout=70,
        retries={"max_attempts": 2, "mode": "standard"},
    ),
)

sagemaker = boto3.client(
    "sagemaker",
    region_name=REGION,
    config=Config(
        connect_timeout=10,
        read_timeout=20,
        retries={"max_attempts": 2, "mode": "standard"},
    ),
)


class DiabetesFeatures(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

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


def require_api_key(
    supplied_key: str | None = Depends(api_key_header),
) -> None:
    if supplied_key is None or not hmac.compare_digest(
        supplied_key,
        RELAY_API_KEY,
    ):
        raise HTTPException(status_code=401, detail="Invalid relay API key.")


def validate_champion_response(payload: Any) -> list[dict[str, Any]]:
    """Fail closed if SageMaker serves stale or unexpected model lineage."""
    if not isinstance(payload, list) or not payload:
        raise HTTPException(
            status_code=502,
            detail="SageMaker returned an unexpected response shape.",
        )

    for row in payload:
        if not isinstance(row, dict):
            raise HTTPException(
                status_code=502,
                detail="SageMaker returned a non-object prediction row.",
            )

        missing = EXPECTED_RESPONSE_FIELDS - set(row)
        if missing:
            raise HTTPException(
                status_code=502,
                detail=(
                    "SageMaker response is missing expected fields: "
                    + ", ".join(sorted(missing))
                ),
            )

        lineage_mismatches = {}

        if row.get("model_type") != EXPECTED_MODEL_TYPE:
            lineage_mismatches["model_type"] = {
                "expected": EXPECTED_MODEL_TYPE,
                "actual": row.get("model_type"),
            }

        if row.get("model_run_id") != EXPECTED_MODEL_RUN_ID:
            lineage_mismatches["model_run_id"] = {
                "expected": EXPECTED_MODEL_RUN_ID,
                "actual": row.get("model_run_id"),
            }

        if row.get("config_version") != EXPECTED_CONFIG_VERSION:
            lineage_mismatches["config_version"] = {
                "expected": EXPECTED_CONFIG_VERSION,
                "actual": row.get("config_version"),
            }

        if row.get("data_version") != EXPECTED_DATA_VERSION:
            lineage_mismatches["data_version"] = {
                "expected": EXPECTED_DATA_VERSION,
                "actual": row.get("data_version"),
            }

        try:
            actual_threshold = float(row.get("decision_threshold"))
        except (TypeError, ValueError):
            actual_threshold = math.nan

        if not math.isclose(
            actual_threshold,
            EXPECTED_DECISION_THRESHOLD,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            lineage_mismatches["decision_threshold"] = {
                "expected": EXPECTED_DECISION_THRESHOLD,
                "actual": row.get("decision_threshold"),
            }

        try:
            probability = float(row.get("probability"))
        except (TypeError, ValueError):
            probability = math.nan

        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            lineage_mismatches["probability"] = {
                "expected": "finite value between 0 and 1",
                "actual": row.get("probability"),
            }
        else:
            expected_class = int(probability >= EXPECTED_DECISION_THRESHOLD)
            if int(row.get("screening_class", -1)) != expected_class:
                lineage_mismatches["screening_class"] = {
                    "expected": expected_class,
                    "actual": row.get("screening_class"),
                }

        if lineage_mismatches:
            raise HTTPException(
                status_code=502,
                detail={
                    "message": (
                        "Deployed endpoint metadata does not match the frozen "
                        "Notebook 03 v5 champion contract."
                    ),
                    "mismatches": lineage_mismatches,
                },
            )

    return payload


@app.get("/")
def home() -> dict[str, Any]:
    return {
        "status": "running",
        "service": "Team02 Diabetes Final-Champion SageMaker Relay",
        "relay_version": "2.0",
        "endpoint": ENDPOINT_NAME,
        "region": REGION,
        "expected_model_type": EXPECTED_MODEL_TYPE,
        "expected_model_run_id": EXPECTED_MODEL_RUN_ID,
        "expected_config_version": EXPECTED_CONFIG_VERSION,
        "routes": [
            "GET /health",
            "POST /predict",
            "GET /docs",
        ],
    }


@app.get("/health")
def health(_: None = Depends(require_api_key)) -> dict[str, Any]:
    try:
        desc = sagemaker.describe_endpoint(EndpointName=ENDPOINT_NAME)
        return {
            "relay": "ok",
            "relay_version": "2.0",
            "endpoint_name": ENDPOINT_NAME,
            "endpoint_status": desc.get("EndpointStatus"),
            "endpoint_config_name": desc.get("EndpointConfigName"),
            "region": REGION,
            "expected_champion": {
                "model_type": EXPECTED_MODEL_TYPE,
                "model_run_id": EXPECTED_MODEL_RUN_ID,
                "decision_threshold": EXPECTED_DECISION_THRESHOLD,
                "config_version": EXPECTED_CONFIG_VERSION,
                "data_version": EXPECTED_DATA_VERSION,
            },
            "prediction_lineage_check": (
                "enforced on POST /predict"
            ),
        }

    except NoCredentialsError:
        raise HTTPException(
            status_code=503,
            detail=(
                "AWS credentials are not available to the relay. "
                "Run this relay inside the Team02 SageMaker Studio/Jupyter environment."
            ),
        )

    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "ClientError")
        message = exc.response.get("Error", {}).get("Message", "")
        raise HTTPException(
            status_code=502,
            detail=f"SageMaker health check failed: {code}: {message}",
        )


@app.post("/predict")
def predict(
    features: DiabetesFeatures,
    _: None = Depends(require_api_key),
) -> Any:
    payload = features.model_dump()

    try:
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(payload).encode("utf-8"),
        )

        raw = response["Body"].read().decode("utf-8")

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=502,
                detail="SageMaker returned a non-JSON response.",
            )

        return validate_champion_response(parsed)

    except NoCredentialsError:
        raise HTTPException(
            status_code=503,
            detail=(
                "AWS credentials are not available to the relay. "
                "Run this relay inside the Team02 SageMaker Studio/Jupyter environment."
            ),
        )

    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "ClientError")
        message = exc.response.get("Error", {}).get("Message", "")
        raise HTTPException(
            status_code=502,
            detail=f"SageMaker invocation failed: {code}: {message}",
        )
