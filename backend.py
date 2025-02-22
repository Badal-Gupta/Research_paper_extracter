from fastapi import FastAPI, UploadFile, File
import boto3
import json
import uuid

app = FastAPI()
AWS_REGION = "ap-south-1"  # Change this to your preferred AWS region

s3_client = boto3.client("s3", region_name=AWS_REGION)
textract_client = boto3.client("textract", region_name=AWS_REGION)
bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)

# AWS Config
S3_BUCKET = "researchpaperbucket"
DYNAMODB_TABLE = "research_summaries"

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    s3_key = f"uploads/{file_id}/{file.filename}"
    
    # Upload to S3
    s3_client.upload_fileobj(file.file, S3_BUCKET, s3_key)
    
    # Extract text using Textract
    response = textract_client.start_document_text_detection(DocumentLocation={
        "S3Object": {"Bucket": S3_BUCKET, "Name": s3_key}
    })
    
    return {"file_id": file_id, "message": "File uploaded successfully! Processing started."}

@app.get("/summarize/{file_id}")
def summarize_text(file_id: str):
    # Retrieve extracted text from Textract (mocked for now)
    extracted_text = "This is a sample extracted text from the research paper."
    
    # Call Amazon Bedrock for summarization
    response = bedrock_client.invoke_model(
        body=json.dumps({"prompt": f"Summarize the following: {extracted_text}"}),
        modelId="your-bedrock-model-id"
    )
    summary = json.loads(response["body"].read())
    
    # Store in DynamoDB
    table = dynamodb.Table(DYNAMODB_TABLE)
    table.put_item(Item={"file_id": file_id, "summary": summary})
    
    return {"file_id": file_id, "summary": summary}
