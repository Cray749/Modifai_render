import json
import boto3
from google import genai
import os

def test_gemini_secret():
    secret_name = "modifai/gemini"
    region_name = os.environ.get("AWS_REGION", "ap-south-1")
    
    print(f"Fetching secret '{secret_name}' from Secrets Manager in {region_name}...")
    
    try:
        session = boto3.session.Session()
        client = session.client(
            service_name='secretsmanager',
            region_name=region_name
        )
        
        response = client.get_secret_value(SecretId=secret_name)
        api_key = response['SecretString']
        
        try:
            secret_dict = json.loads(api_key)
            if "api_key" in secret_dict:
                api_key = secret_dict["api_key"]
            elif "GEMINI_API_KEY" in secret_dict:
                api_key = secret_dict["GEMINI_API_KEY"]
        except json.JSONDecodeError:
            pass # Plain string
            
        print("✅ Secret retrieved successfully.")
        
        print("Testing Gemini 2.5 Flash...")
        gemini_client = genai.Client(api_key=api_key)
        
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents="Say 'Hello, Gemini is working!'",
            config=genai.types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=50
            )
        )
        
        print("✅ Gemini API call successful!")
        print(f"Response: {response.text}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_gemini_secret()
