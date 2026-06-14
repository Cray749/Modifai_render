import sys
import os
import boto3
import json
import argparse
from pathlib import Path

# Extracted from your AWS stack
BUCKET_NAME = "modifai-modifaidatabucket-i9qdgwtbeceg"
STATE_MACHINE_ARN = "arn:aws:states:ap-south-1:527371380408:stateMachine:ModifaiStateMachine-A4wCn417NgmJ"

def upload_to_s3(file_path: Path, bucket: str) -> str:
    s3 = boto3.client('s3')
    key = f"inputs/{file_path.name}"
    print(f"Uploading {file_path.name} to s3://{bucket}/{key} ...")
    s3.upload_file(str(file_path), bucket, key)
    return f"s3://{bucket}/{key}"

def start_execution(s3_uris: list, mode: str):
    sf = boto3.client('stepfunctions')
    
    payload = {
        "pipeline_mode": mode,
        "document_s3_uris": s3_uris
    }
    
    print("\nStarting Modifai Agentic Pipeline...")
    response = sf.start_execution(
        stateMachineArn=STATE_MACHINE_ARN,
        input=json.dumps(payload)
    )
    
    execution_arn = response['executionArn']
    print("\n✅ Execution Started Successfully!")
    print(f"Execution ARN: {execution_arn}")
    print("\nYou can monitor the progress visually in the AWS Step Functions Console.")

def main():
    parser = argparse.ArgumentParser(description="Trigger the Modifai Agentic Pipeline")
    parser.add_argument("files", nargs="+", help="Paths to the local PDFs/Documents to process")
    parser.add_argument("--mode", default="DATASET_ONLY", choices=[
        "DATASET_ONLY", 
        "DATASET_AND_FINETUNE", 
        "DATASET_AND_FINETUNE_AND_DEPLOY"
    ], help="Which pipeline mode to run")
    
    args = parser.parse_args()
    
    s3_uris = []
    for fp in args.files:
        path = Path(fp)
        if not path.exists():
            print(f"Error: File not found -> {fp}")
            sys.exit(1)
        uri = upload_to_s3(path, BUCKET_NAME)
        s3_uris.append(uri)
        
    start_execution(s3_uris, args.mode)

if __name__ == "__main__":
    main()
