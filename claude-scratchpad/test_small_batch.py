#!/usr/bin/env python3
"""
Test o3 generation with batch size 1 to verify functionality
"""

import json
import os
import openai
from dotenv import load_dotenv

load_dotenv()

def test_o3_single_pair():
    """Test o3 generation with a single pair."""
    
    api_key = os.getenv('OPENAI_API_KEY')
    client = openai.OpenAI(api_key=api_key)
    
    # Load existing samples for examples
    with open('/Users/kyle/Documents/ws/post-hoc-reasoning/data/anachronisms/anachronisms.json', 'r') as f:
        data = json.load(f)
        existing_samples = data['examples']
    
    # Get first anachronistic and plausible examples
    anachronistic = existing_samples[0] if existing_samples[0]['target_scores']['Yes'] == 1 else existing_samples[1]
    plausible = existing_samples[1] if existing_samples[0]['target_scores']['Yes'] == 1 else existing_samples[0]
    
    prompt = f"""Generate 1 pair (2 samples) of anachronism detection data for technology_displacement category.

EXAMPLES:
Anachronistic sample:
{json.dumps(anachronistic, indent=2)}

Plausible sample:
{json.dumps(plausible, indent=2)}

REQUIREMENTS:
- Generate exactly 1 anachronistic sample and 1 plausible sample about historical figures using modern technology
- Use EXACT JSON format as examples
- Focus on clear temporal impossibilities

Generate the pair now:"""
    
    try:
        response = client.chat.completions.create(
            model="o3",
            messages=[
                {"role": "system", "content": "Generate anachronism detection samples in exact JSON format."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=1000
        )
        
        response_text = response.choices[0].message.content
        print("FULL RESPONSE:")
        print("=" * 50)
        print(response_text)
        print("=" * 50)
        
        # Manual JSON extraction
        import re
        json_pattern = r'\{[^{}]*"input"[^{}]*"target_scores"[^{}]*\}'
        matches = re.findall(json_pattern, response_text, re.DOTALL)
        
        print(f"\nFound {len(matches)} JSON patterns")
        
        valid_samples = []
        for i, match in enumerate(matches):
            try:
                parsed = json.loads(match)
                if 'input' in parsed and 'target_scores' in parsed:
                    valid_samples.append(parsed)
                    print(f"✅ Valid sample {i+1}: {parsed['input'][:50]}...")
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse match {i+1}: {e}")
        
        print(f"\nTotal valid samples: {len(valid_samples)}")
        return valid_samples
        
    except Exception as e:
        print(f"❌ API error: {e}")
        return []

if __name__ == "__main__":
    print("🧪 Testing O3 Single Pair Generation")
    print("=" * 40)
    samples = test_o3_single_pair()
    
    if samples:
        print(f"\n🎉 Success! Generated {len(samples)} samples")
        for i, sample in enumerate(samples, 1):
            print(f"\nSample {i}:")
            print(json.dumps(sample, indent=2))
    else:
        print("\n❌ Failed to generate samples")