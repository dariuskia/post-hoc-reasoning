#!/usr/bin/env python3
"""
Working anachronisms generator with corrected parsing logic
Using o3 with batch size 10 pairs (20 samples)
"""

import json
import os
import openai
from dotenv import load_dotenv
import time
import random

load_dotenv()
random.seed(42)

def parse_response_correctly(response_text):
    """Improved JSON parsing that handles the actual o3 response format."""
    
    samples = []
    
    # Split by lines and look for JSON objects
    lines = response_text.split('\n')
    current_json = ""
    brace_depth = 0
    in_json = False
    
    for line in lines:
        line = line.strip()
        
        # Start of JSON object
        if line.startswith('{') and not in_json:
            in_json = True
            current_json = line
            brace_depth = line.count('{') - line.count('}')
            
            # Check if it's a complete single-line JSON
            if brace_depth == 0:
                try:
                    parsed = json.loads(current_json)
                    if 'input' in parsed and 'target_scores' in parsed:
                        samples.append(parsed)
                except json.JSONDecodeError:
                    pass
                in_json = False
                current_json = ""
        
        # Continue building multi-line JSON
        elif in_json:
            current_json += '\n' + line
            brace_depth += line.count('{') - line.count('}')
            
            # Complete JSON object found
            if brace_depth == 0:
                try:
                    parsed = json.loads(current_json)
                    if 'input' in parsed and 'target_scores' in parsed:
                        samples.append(parsed)
                except json.JSONDecodeError:
                    pass
                in_json = False
                current_json = ""
    
    return samples

def generate_batch_with_o3(batch_size=10):
    """Generate a batch of samples using o3."""
    
    api_key = os.getenv('OPENAI_API_KEY')
    client = openai.OpenAI(api_key=api_key)
    
    # Load existing samples for examples
    with open('/Users/kyle/Documents/ws/post-hoc-reasoning/data/anachronisms/anachronisms.json', 'r') as f:
        data = json.load(f)
        existing_samples = data['examples']
    
    # Get example pairs
    anachronistic = existing_samples[0] if existing_samples[0]['target_scores']['Yes'] == 1 else existing_samples[1]
    plausible = existing_samples[1] if existing_samples[0]['target_scores']['Yes'] == 1 else existing_samples[0]
    
    prompt = f"""Generate {batch_size} pairs of anachronism detection samples for technology_displacement category.

EXACT EXAMPLES:
Anachronistic sample:
{json.dumps(anachronistic, indent=2)}

Plausible sample:
{json.dumps(plausible, indent=2)}

REQUIREMENTS:
1. Generate exactly {batch_size} pairs ({batch_size * 2} samples total)
2. Each pair has one anachronistic and one plausible version
3. Use EXACT JSON format as shown above
4. Focus on historical figures using modern technology that didn't exist in their time
5. Make anachronisms clear but not cartoonish
6. Ensure plausible versions are historically accurate

Generate all {batch_size} pairs now (output each JSON object on separate lines):"""
    
    try:
        print(f"Generating {batch_size} pairs with o3...")
        response = client.chat.completions.create(
            model="o3",
            messages=[
                {"role": "system", "content": "Generate anachronism detection samples in exact JSON format. Output each JSON object clearly separated."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=4000
        )
        
        response_text = response.choices[0].message.content
        print(f"Response received ({len(response_text)} chars)")
        
        # Parse with corrected logic
        samples = parse_response_correctly(response_text)
        print(f"Parsed {len(samples)} valid samples")
        
        return samples
        
    except Exception as e:
        print(f"❌ API error: {e}")
        return []

def main():
    """Test the working batch generation."""
    
    print("🚀 Working Batch Generator Test")
    print("Model: o3 | Batch Size: 10 pairs | Target: 20 samples")
    print("=" * 50)
    
    # Generate batch
    samples = generate_batch_with_o3(10)
    
    if samples:
        print(f"\n🎉 Successfully generated {len(samples)} samples!")
        
        # Analyze results
        anachronistic = [s for s in samples if s['target_scores']['Yes'] == 1]
        plausible = [s for s in samples if s['target_scores']['No'] == 1]
        
        print(f"  Anachronistic: {len(anachronistic)} samples")
        print(f"  Plausible: {len(plausible)} samples")
        print(f"  Expected pairs: {len(samples) // 2}")
        
        # Save samples
        print("\n💾 Saving test samples...")
        test_data = {"examples": samples}
        with open('/Users/kyle/Documents/ws/post-hoc-reasoning/anachronisms_test_batch.json', 'w') as f:
            json.dump(test_data, f, indent=2)
        
        print(f"✅ Saved: anachronisms_test_batch.json ({len(samples)} samples)")
        
        # Show first few samples
        print(f"\n📋 Sample Preview:")
        for i, sample in enumerate(samples[:4]):
            sample_type = "Anachronistic" if sample['target_scores']['Yes'] == 1 else "Plausible"
            print(f"  {i+1}. [{sample_type}] {sample['input'][:60]}...")
        
        if len(samples) == 20:
            print("\n✅ Perfect! Got exactly 20 samples as expected.")
            print("Ready to scale up to production batch sizes.")
        else:
            print(f"\n⚠️  Got {len(samples)} samples instead of expected 20.")
            print("May need batch size adjustment for production.")
            
    else:
        print("\n❌ No samples generated. Check API connectivity and parsing logic.")

if __name__ == "__main__":
    main()