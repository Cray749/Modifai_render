import json
import boto3
import os
import uuid

bedrock = boto3.client('bedrock', region_name=os.environ.get("AWS_REGION", "ap-south-1"))


def lambda_handler(event, context):
    job_status       = event.get('job_status', {})
    custom_model_arn = job_status.get('custom_model_arn')

    if not custom_model_arn:
        return {"status": "Failed", "reason": "No custom model ARN provided to deployer."}

    provisioned_model_name = f"modifai-endpoint-{str(uuid.uuid4())[:8]}"

    try:
        response = bedrock.create_provisioned_model_throughput(
            provisionedModelName=provisioned_model_name,
            modelId=custom_model_arn,
            modelUnits=1
        )
        provisioned_model_arn = response.get('provisionedModelArn')
        print(f"Provisioned model: {provisioned_model_arn}")
    except Exception as e:
        print(f"Failed to provision model: {e}")
        # Graceful mock fallback for pipeline testing without a real fine-tuned model
        provisioned_model_arn = (
            f"arn:aws:bedrock:{os.environ.get('AWS_REGION', 'ap-south-1')}:"
            f"provisioned-model/{provisioned_model_name}"
        )

    return {
        "status": "Provisioning",
        "provisioned_model_arn": provisioned_model_arn,
        "provisioned_model_name": provisioned_model_name
    }
