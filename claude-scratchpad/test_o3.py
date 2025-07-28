#!/usr/bin/env python3
"""
Simple test script to verify o3 API works
"""

import os

import openai
from dotenv import load_dotenv

load_dotenv()

# Check for OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY not found")
    exit(1)

client = openai.OpenAI(api_key=api_key)

try:
    response = client.chat.completions.create(
        model="o3-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello and confirm you are working."},
        ],
        max_completion_tokens=100,
    )

    print("✅ o3-mini API test successful!")
    print("Response:", response.choices[0].message.content)

except Exception as e:
    print("❌ o3-mini API test failed:")
    print(f"Error: {e}")
