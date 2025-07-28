#!/usr/bin/env python3
"""
Debug script to test o3 generation with different batch sizes
"""

import json
import os

import openai
from dotenv import load_dotenv

load_dotenv()


def test_o3_batch_generation(batch_size: int = 1):
    """Test o3 generation with a specific batch size."""

    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)

    # Load existing samples for examples
    with open(
        "/Users/kyle/Documents/ws/post-hoc-reasoning/data/anachronisms/anachronisms.json",
        "r",
    ) as f:
        data = json.load(f)
        existing_samples = data["examples"]

    # Get a simple example pair
    anachronistic = (
        existing_samples[0]
        if existing_samples[0]["target_scores"]["Yes"] == 1
        else existing_samples[1]
    )
    plausible = (
        existing_samples[1]
        if existing_samples[0]["target_scores"]["Yes"] == 1
        else existing_samples[0]
    )

    # Create a simple prompt
    prompt = f"""TASK: Generate {batch_size} pairs of anachronism detection samples for technology_displacement category.

EXACT EXAMPLE FROM DATASET:

Anachronistic version:
{json.dumps(anachronistic, indent=2)}

Plausible version:
{json.dumps(plausible, indent=2)}

REQUIREMENTS:
1. Use EXACT JSON format as shown
2. Generate {batch_size} pairs (anachronistic + plausible)
3. Focus on historical figures using modern technology

Generate {batch_size} pairs now:"""

    try:
        print(f"Testing o3 with batch size {batch_size}...")
        response = client.chat.completions.create(
            model="o3",
            messages=[
                {
                    "role": "system",
                    "content": "Generate anachronism detection samples in exact JSON format.",
                },
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=2000,
        )

        response_text = response.choices[0].message.content
        print(f"Response length: {len(response_text)} characters")
        print(f"Response preview: {response_text[:300]}...")

        # Try to parse JSON blocks
        json_blocks = []
        current_block = ""
        in_json = False
        brace_count = 0

        for line in response_text.split("\\n"):
            line = line.strip()
            if line.startswith("{"):
                in_json = True
                brace_count = 0
                current_block = line
                brace_count += line.count("{") - line.count("}")
            elif in_json:
                current_block += "\\n" + line
                brace_count += line.count("{") - line.count("}")

                if brace_count <= 0:
                    try:
                        parsed = json.loads(current_block)
                        if "input" in parsed and "target_scores" in parsed:
                            json_blocks.append(parsed)
                            print(f"✅ Parsed JSON block {len(json_blocks)}")
                    except json.JSONDecodeError as e:
                        print(f"❌ JSON parse error: {e}")
                        print(f"Block: {current_block[:200]}...")
                    in_json = False
                    current_block = ""

        print(f"Total parsed blocks: {len(json_blocks)}")
        return len(json_blocks) > 0

    except Exception as e:
        print(f"❌ API error: {e}")
        return False


def main():
    print("🔧 Debug O3 Generation")
    print("=" * 30)

    # Test different batch sizes
    batch_sizes = [1, 2, 5, 10]

    for batch_size in batch_sizes:
        print(f"\\n--- Testing batch size {batch_size} ---")
        success = test_o3_batch_generation(batch_size)
        print(f"Result: {'✅ SUCCESS' if success else '❌ FAILED'}")

        if not success and batch_size == 1:
            print("❌ Single pair generation failed - API issue")
            break


if __name__ == "__main__":
    main()
