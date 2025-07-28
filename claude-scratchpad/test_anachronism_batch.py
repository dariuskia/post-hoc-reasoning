#!/usr/bin/env python3
"""
Test script to generate one small batch of anachronisms
"""

import json
import os

import openai
from dotenv import load_dotenv

load_dotenv()

# Load example from existing dataset
with open(
    "/Users/kyle/Documents/ws/post-hoc-reasoning/data/anachronisms/anachronisms.json",
    "r",
) as f:
    data = json.load(f)
    examples = data["examples"]

# Get a pair of examples (anachronistic + plausible)
example_anachronistic = (
    examples[0] if examples[0]["target_scores"]["Yes"] == 1 else examples[1]
)
example_plausible = (
    examples[1] if examples[0]["target_scores"]["Yes"] == 1 else examples[0]
)

print("Example anachronistic:", example_anachronistic)
print("Example plausible:", example_plausible)

# Create a simple prompt for 1 pair
prompt = f"""TASK: Generate 1 pair of anachronism detection samples for technology_displacement category.

CATEGORY DESCRIPTION: Historical figures using modern technology that didn't exist in their time

EXACT EXAMPLES FROM THE DATASET:

Example 1:
Anachronistic version:
{json.dumps(example_anachronistic, indent=2)}

Plausible version:
{json.dumps(example_plausible, indent=2)}

CRITICAL REQUIREMENTS:
1. EXACT FORMAT: Each sample must have EXACTLY the same JSON structure as the examples above
2. EXACT FIELD NAMES: Use "input" and "target_scores" with "Yes" and "No" keys only
3. EXACT SCORING: Anachronistic samples get {{"Yes": 1, "No": 0}}, plausible samples get {{"Yes": 0, "No": 1}}
4. PAIRED STRUCTURE: Generate pairs where one version is anachronistic and one is historically plausible
5. SIMILAR LENGTH: Keep sentences roughly the same length as the examples (20-150 words)
6. NATURAL LANGUAGE: Write in natural, fluent English that sounds realistic
7. CLEAR ANACHRONISMS: Make the temporal impossibility obvious to someone with historical knowledge
8. HISTORICAL ACCURACY: Ensure the plausible versions are factually correct

Generate exactly 1 pair. For each pair, output the anachronistic version first, then the plausible version. Use the EXACT JSON format shown in the examples.

Sample 1 (Anachronistic):
{{
  "input": "[Your anachronistic sentence here]",
  "target_scores": {{
    "Yes": 1,
    "No": 0
  }}
}}

Sample 1 (Plausible):
{{
  "input": "[Your plausible sentence here]", 
  "target_scores": {{
    "Yes": 0,
    "No": 1
  }}
}}

Generate the 1 pair now:"""

print("\n" + "=" * 50)
print("TESTING O3 BATCH GENERATION")
print("=" * 50)

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    response = client.chat.completions.create(
        model="o3-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an expert historian and dataset creator. You must generate anachronism detection samples that exactly match the format and style of the provided examples.",
            },
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=1000,
    )

    print("✅ Generation successful!")
    print("\nRAW RESPONSE:")
    print(response.choices[0].message.content)

    # Try to parse JSON blocks
    response_text = response.choices[0].message.content
    json_blocks = []
    current_block = ""
    in_json = False
    brace_count = 0

    for line in response_text.split("\n"):
        line = line.strip()
        if line.startswith("{"):
            in_json = True
            brace_count = 0
            current_block = line
            brace_count += line.count("{") - line.count("}")
        elif in_json:
            current_block += "\n" + line
            brace_count += line.count("{") - line.count("}")

            if brace_count <= 0:
                try:
                    parsed = json.loads(current_block)
                    if "input" in parsed and "target_scores" in parsed:
                        json_blocks.append(parsed)
                        print(f"\n✅ Parsed JSON block: {len(json_blocks)}")
                        print(json.dumps(parsed, indent=2))
                except json.JSONDecodeError as e:
                    print(f"\n❌ JSON parse error: {e}")
                    print(f"Block: {current_block}")
                in_json = False
                current_block = ""

    print(f"\nTotal parsed blocks: {len(json_blocks)}")

except Exception as e:
    print(f"❌ Generation failed: {e}")
