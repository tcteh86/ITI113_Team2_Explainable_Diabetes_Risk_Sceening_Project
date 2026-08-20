import io
import json
import tarfile
import time
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError, WaiterError


def _error_code(exc):
    return exc.response.get("Error", {}).get("Code", "Unknown")


def _is_not_found(exc):
    return _error_code(exc) in {
        "ValidationException",
        "ResourceNotFound",
        "ResourceNotFoundException",
    }


def get_pipeline_execution_summary(sm, execution_arn):
    execution = sm.describe_pipeline_execution(
        PipelineExecutionArn=execution_arn
    )
    response = sm.list_pipeline_execution_steps(
        PipelineExecutionArn=execution_arn,
        SortOrder="Ascending",
        MaxResults=100,
    )

    steps = []
    for step in response.get("PipelineExecutionSteps", []):
        metadata = step.get("Metadata", {}) or {}
        processing = metadata.get("ProcessingJob", {}) or {}
        training = metadata.get("TrainingJob", {}) or {}
        model = metadata.get("Model", {}) or {}
        register_model = metadata.get("RegisterModel", {}) or {}

        steps.append({
            "step_name": step.get("StepName"),
            "step_status": step.get("StepStatus"),
            "failure_reason": step.get("FailureReason"),
            "processing_job_arn": processing.get("Arn"),
            "training_job_arn": training.get("Arn"),
            "model_arn": model.get("Arn"),
            "register_model_arn": register_model.get("Arn"),
            "start_time": step.get("StartTime"),
            "end_time": step.get("EndTime"),
        })

    return {
        "execution_arn": execution_arn,
        "pipeline_status": execution.get("PipelineExecutionStatus"),
        "failure_reason": execution.get("FailureReason"),
        "creation_time": execution.get("CreationTime"),
        "last_modified_time": execution.get("LastModifiedTime"),
        "steps": steps,
    }


def get_recent_pipeline_executions(sm, pipeline_name, max_results=10):
    response = sm.list_pipeline_executions(
        PipelineName=pipeline_name,
        SortOrder="Descending",
        MaxResults=max_results,
    )
    return response.get("PipelineExecutionSummaries", [])


def get_model_registry_summary(sm, model_package_group, max_results=20):
    response = sm.list_model_packages(
        ModelPackageGroupName=model_package_group,
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=max_results,
    )
    rows = []
    for item in response.get("ModelPackageSummaryList", []):
        rows.append({
            "model_package_arn": item.get("ModelPackageArn"),
            "model_package_version": item.get("ModelPackageVersion"),
            "approval_status": item.get("ModelApprovalStatus"),
            "creation_time": item.get("CreationTime"),
        })
    return rows



def _parse_s3_uri(uri):
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Expected s3:// URI, received: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def get_model_package_model_data_url(sm, model_package_arn):
    package = sm.describe_model_package(ModelPackageName=model_package_arn)
    containers = package.get("InferenceSpecification", {}).get("Containers", [])
    for container in containers:
        if container.get("ModelDataUrl"):
            return container["ModelDataUrl"]
        s3_source = container.get("ModelDataSource", {}).get("S3DataSource", {})
        if s3_source.get("S3Uri"):
            return s3_source["S3Uri"]
    raise RuntimeError("No model artifact S3 URI found in Model Package.")


def validate_model_package_artifact(sm, s3, model_package_arn):
    required_members = [
        "model.pkl",
        "model_metadata.json",
        "code/inference.py",
        "code/requirements.txt",
    ]
    model_data_url = get_model_package_model_data_url(sm, model_package_arn)
    bucket, key = _parse_s3_uri(model_data_url)
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()

    with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tar:
        members = sorted(
            member.name.lstrip("./")
            for member in tar.getmembers()
            if member.isfile()
        )

    missing = [x for x in required_members if x not in members]
    return {
        "valid": not missing,
        "model_package_arn": model_package_arn,
        "model_data_url": model_data_url,
        "required_members": required_members,
        "missing_members": missing,
        "artifact_members": members,
    }

def approve_model_package(sm, model_package_arn, approval_description):
    before = sm.describe_model_package(
        ModelPackageName=model_package_arn
    )
    sm.update_model_package(
        ModelPackageArn=model_package_arn,
        ModelApprovalStatus="Approved",
        ApprovalDescription=approval_description,
    )
    after = sm.describe_model_package(
        ModelPackageName=model_package_arn
    )
    return {
        "model_package_arn": model_package_arn,
        "status_before": before.get("ModelApprovalStatus"),
        "status_after": after.get("ModelApprovalStatus"),
    }


def get_endpoint_summary(sm, endpoint_name):
    try:
        desc = sm.describe_endpoint(EndpointName=endpoint_name)
    except ClientError as exc:
        if _is_not_found(exc):
            return {
                "exists": False,
                "endpoint_name": endpoint_name,
                "endpoint_status": "NotCreated",
            }
        raise

    return {
        "exists": True,
        "endpoint_name": endpoint_name,
        "endpoint_status": desc.get("EndpointStatus"),
        "endpoint_arn": desc.get("EndpointArn"),
        "endpoint_config_name": desc.get("EndpointConfigName"),
        "creation_time": desc.get("CreationTime"),
        "last_modified_time": desc.get("LastModifiedTime"),
        "failure_reason": desc.get("FailureReason"),
    }



def get_endpoint_cloudwatch_log_tail(region_name, endpoint_name, max_events=100):
    logs = boto3.client("logs", region_name=region_name)
    log_group = f"/aws/sagemaker/Endpoints/{endpoint_name}"
    try:
        response = logs.describe_log_streams(
            logGroupName=log_group,
            orderBy="LastEventTime",
            descending=True,
            limit=20,
        )
    except ClientError as exc:
        if _is_not_found(exc):
            return []
        raise

    events = []
    for stream in response.get("logStreams", []):
        stream_name = stream["logStreamName"]
        data = logs.get_log_events(
            logGroupName=log_group,
            logStreamName=stream_name,
            startFromHead=False,
            limit=max_events,
        )
        for event in data.get("events", []):
            events.append({
                "timestamp": event.get("timestamp"),
                "message": event.get("message", "").rstrip(),
                "log_stream": stream_name,
            })
    events.sort(key=lambda x: x.get("timestamp") or 0)
    return events[-max_events:]


def diagnose_endpoint_failure(sm, endpoint_name, max_log_events=100):
    state = get_endpoint_summary(sm, endpoint_name)
    return {
        **state,
        "cloudwatch_log_group": f"/aws/sagemaker/Endpoints/{endpoint_name}",
        "cloudwatch_events": get_endpoint_cloudwatch_log_tail(
            sm.meta.region_name,
            endpoint_name,
            max_events=max_log_events,
        ),
    }

def deploy_serverless_model_package(
    sm,
    model_package_arn,
    role_arn,
    endpoint_name,
    tags,
    memory_size_mb=2048,
    max_concurrency=5,
):
    package = sm.describe_model_package(ModelPackageName=model_package_arn)
    if package.get("ModelApprovalStatus") != "Approved":
        raise RuntimeError("Deployment blocked: Model Package must be Approved.")

    current = get_endpoint_summary(sm, endpoint_name)

    if current["exists"] and current["endpoint_status"] == "Failed":
        result = diagnose_endpoint_failure(sm, endpoint_name)
        result.update({
            "deployment_succeeded": False,
            "deployment_action": "blocked_by_existing_failed_endpoint",
            "model_package_arn": model_package_arn,
            "requires_explicit_delete": True,
        })
        return result

    timestamp = int(time.time())
    model_name = f"{endpoint_name}-model-{timestamp}"
    config_name = f"{endpoint_name}-config-{timestamp}"

    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={"ModelPackageName": model_package_arn},
        ExecutionRoleArn=role_arn,
        Tags=tags,
    )
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            "VariantName": "AllTraffic",
            "ModelName": model_name,
            "ServerlessConfig": {
                "MemorySizeInMB": int(memory_size_mb),
                "MaxConcurrency": int(max_concurrency),
            },
        }],
        Tags=tags,
    )

    if current["exists"]:
        sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
        action = "updated"
    else:
        sm.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
            Tags=tags,
        )
        action = "created"

    try:
        sm.get_waiter("endpoint_in_service").wait(EndpointName=endpoint_name)
    except WaiterError:
        result = diagnose_endpoint_failure(sm, endpoint_name)
        result.update({
            "deployment_succeeded": False,
            "deployment_action": action,
            "model_name": model_name,
            "endpoint_config_name": config_name,
            "model_package_arn": model_package_arn,
            "serverless_memory_mb": int(memory_size_mb),
            "serverless_max_concurrency": int(max_concurrency),
            "requires_explicit_delete": result.get("endpoint_status") == "Failed",
        })
        return result

    result = get_endpoint_summary(sm, endpoint_name)
    result.update({
        "deployment_succeeded": True,
        "deployment_action": action,
        "model_name": model_name,
        "endpoint_config_name": config_name,
        "model_package_arn": model_package_arn,
        "serverless_memory_mb": int(memory_size_mb),
        "serverless_max_concurrency": int(max_concurrency),
        "requires_explicit_delete": False,
        "cloudwatch_events": [],
    })
    return result

def invoke_json_endpoint(runtime, endpoint_name, payload):
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload).encode("utf-8"),
    )
    return json.loads(response["Body"].read().decode("utf-8"))


def delete_endpoint_if_exists(
    sm,
    endpoint_name,
    wait=True,
    poll_seconds=10,
    timeout_seconds=600,
):
    current = get_endpoint_summary(sm, endpoint_name)
    if not current["exists"]:
        return {
            "endpoint_name": endpoint_name,
            "deleted": False,
            "reason": "Endpoint does not exist.",
        }

    sm.delete_endpoint(EndpointName=endpoint_name)

    if wait:
        started = time.time()
        while time.time() - started < timeout_seconds:
            if not get_endpoint_summary(sm, endpoint_name)["exists"]:
                return {
                    "endpoint_name": endpoint_name,
                    "deleted": True,
                    "reason": "Endpoint deleted and absence confirmed.",
                }
            time.sleep(poll_seconds)

    return {
        "endpoint_name": endpoint_name,
        "deleted": True,
        "reason": "DeleteEndpoint API submitted.",
    }


def _find_log_streams(logs, log_group, job_name, limit=20):
    try:
        response = logs.describe_log_streams(
            logGroupName=log_group,
            logStreamNamePrefix=job_name,
            orderBy="LogStreamName",
            descending=False,
            limit=limit,
        )
    except ClientError as exc:
        if _is_not_found(exc):
            return []
        raise
    return [x["logStreamName"] for x in response.get("logStreams", [])]


def get_cloudwatch_job_log_tail(
    region_name,
    log_group,
    job_name,
    max_events=80,
):
    logs = boto3.client("logs", region_name=region_name)
    streams = _find_log_streams(logs, log_group, job_name)

    events = []
    for stream in streams:
        response = logs.get_log_events(
            logGroupName=log_group,
            logStreamName=stream,
            startFromHead=False,
            limit=max_events,
        )
        for event in response.get("events", []):
            events.append({
                "timestamp": event.get("timestamp"),
                "message": event.get("message", "").rstrip(),
                "log_stream": stream,
            })

    events.sort(key=lambda x: x.get("timestamp") or 0)
    return events[-max_events:]


def get_execution_cloudwatch_report(
    sm,
    execution_arn,
    region_name,
    max_events_per_job=60,
):
    summary = get_pipeline_execution_summary(sm, execution_arn)
    report = []

    for step in summary["steps"]:
        if step.get("processing_job_arn"):
            job_name = step["processing_job_arn"].rsplit("/", 1)[-1]
            report.append({
                "step_name": step["step_name"],
                "job_type": "ProcessingJob",
                "job_name": job_name,
                "log_group": "/aws/sagemaker/ProcessingJobs",
                "events": get_cloudwatch_job_log_tail(
                    region_name,
                    "/aws/sagemaker/ProcessingJobs",
                    job_name,
                    max_events=max_events_per_job,
                ),
            })

        if step.get("training_job_arn"):
            job_name = step["training_job_arn"].rsplit("/", 1)[-1]
            report.append({
                "step_name": step["step_name"],
                "job_type": "TrainingJob",
                "job_name": job_name,
                "log_group": "/aws/sagemaker/TrainingJobs",
                "events": get_cloudwatch_job_log_tail(
                    region_name,
                    "/aws/sagemaker/TrainingJobs",
                    job_name,
                    max_events=max_events_per_job,
                ),
            })

    return report
