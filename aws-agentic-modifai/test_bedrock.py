import boto3

bedrock = boto3.client(
    "bedrock-runtime",
    region_name="ap-south-1"
)

models_to_test = [
    # APAC inference profiles - correct IDs for ap-south-1
    "apac.amazon.nova-micro-v1:0",   # cheapest, fastest
    "apac.amazon.nova-lite-v1:0",    # balanced
    "apac.amazon.nova-pro-v1:0",     # most capable Nova
    # Llama3 / Mistral (may work after daily quota reset)
    "meta.llama3-8b-instruct-v1:0",
    "mistral.mistral-7b-instruct-v0:2",
]

test_prompt = "Say hello in one sentence."

working_model = None
for model_id in models_to_test:
    print(f"\n{'='*60}")
    print(f"Testing model: {model_id}")
    try:
        response = bedrock.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": test_prompt}]}],
            inferenceConfig={"temperature": 0.7, "maxTokens": 50}
        )
        text = response['output']['message']['content'][0]['text']
        print(f"✅ SUCCESS! Model works!")
        print(f"Response: {text[:200]}")
        working_model = model_id
        break
    except Exception as e:
        print(f"❌ FAILED: {e}")

print(f"\n{'='*60}")
if working_model:
    print(f"✅ WORKING MODEL: {working_model}")
    print(f"Update all your Lambdas to use: {working_model}")
else:
    print("❌ No models worked. Daily quota likely exhausted. Try again tomorrow or request quota increase.")