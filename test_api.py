#!/usr/bin/env python3
"""
Quick OpenAI API Connection Test
Run this to verify your API key and connection BEFORE running the main app
"""

import os
from dotenv import load_dotenv

print("="*60)
print("🧪 OpenAI API Connection Test")
print("="*60)

# Step 1: Load environment
print("\n1️⃣ Loading environment variables...")
load_dotenv(dotenv_path=".secretcontainer/.env")

api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not api_key:
    print("❌ FAILED: OPENAI_API_KEY not found in .env file!")
    print("\nPlease create .secretcontainer/.env with:")
    print("OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx")
    exit(1)

print(f"✅ API Key found: {api_key[:8]}...{api_key[-4:]}")
print(f"✅ Model: {model}")

# Step 2: Test import
print("\n2️⃣ Testing OpenAI library...")
try:
    from openai import OpenAI
    print("✅ OpenAI library imported successfully")
except ImportError as e:
    print(f"❌ FAILED: {e}")
    print("\nPlease install: pip install openai")
    exit(1)

# Step 3: Create client
print("\n3️⃣ Creating OpenAI client...")
try:
    client = OpenAI(api_key=api_key)
    print("✅ Client created successfully")
except Exception as e:
    print(f"❌ FAILED: {e}")
    exit(1)

# Step 4: Test API call
print("\n4️⃣ Testing API call...")
try:
    print(f"   Calling model: {model}")
    print("   Sending test message...")
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Say hello in JSON format: {\"message\": \"...\"}"}
        ],
        max_tokens=50
    )
    
    # Extract response
    content = response.choices[0].message.content
    usage = response.usage
    
    print("\n✅ API CALL SUCCESSFUL!")
    print(f"\n📊 Response Details:")
    print(f"   - Model: {response.model}")
    print(f"   - Content: {content}")
    print(f"   - Input tokens: {usage.prompt_tokens}")
    print(f"   - Output tokens: {usage.completion_tokens}")
    print(f"   - Total tokens: {usage.total_tokens}")
    
    # Calculate cost
    if model == "gpt-4o-mini":
        cost = (usage.prompt_tokens / 1_000_000) * 0.150 + (usage.completion_tokens / 1_000_000) * 0.600
        print(f"   - Estimated cost: ${cost:.6f}")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nYour OpenAI API is working correctly.")
    print("You can now run: python app_debug.py")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ API CALL FAILED!")
    print(f"\nError Type: {type(e).__name__}")
    print(f"Error Message: {str(e)}")
    
    print("\n" + "="*60)
    print("🔍 Troubleshooting Tips:")
    print("="*60)
    
    error_str = str(e).lower()
    
    if "authentication" in error_str or "api key" in error_str:
        print("\n1. Invalid API Key")
        print("   - Check your API key in .secretcontainer/.env")
        print("   - Verify key starts with 'sk-proj-' or 'sk-'")
        print("   - Get new key from: https://platform.openai.com/api-keys")
    
    elif "quota" in error_str or "exceeded" in error_str:
        print("\n2. Quota Exceeded")
        print("   - You've run out of API credits")
        print("   - Add credits at: https://platform.openai.com/account/billing")
    
    elif "model" in error_str:
        print("\n3. Model Not Available")
        print(f"   - Model '{model}' may not be available to your account")
        print("   - Try changing OPENAI_MODEL in .env to:")
        print("     * gpt-4o")
        print("     * gpt-3.5-turbo")
    
    elif "connection" in error_str or "timeout" in error_str:
        print("\n4. Network/Connection Issue")
        print("   - Check your internet connection")
        print("   - Check firewall settings")
        print("   - Try again in a few minutes")
    
    elif "rate limit" in error_str:
        print("\n5. Rate Limit Hit")
        print("   - Too many requests in short time")
        print("   - Wait 1 minute and try again")
    
    else:
        print("\n❓ Unknown Error")
        print("   - Check OpenAI status: https://status.openai.com")
        print("   - Review error message above")
    
    print("\n" + "="*60)
    exit(1)