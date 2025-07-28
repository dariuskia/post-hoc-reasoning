#!/usr/bin/env python3
"""
Small-scale production test - generate 40 samples to verify system works
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

def generate_category_batch(category, batch_size=10):
    """Generate a batch for a specific category."""
    
    api_key = os.getenv('OPENAI_API_KEY')
    client = openai.OpenAI(api_key=api_key)
    
    # Load existing samples for examples
    with open('/Users/kyle/Documents/ws/post-hoc-reasoning/data/anachronisms/anachronisms.json', 'r') as f:
        data = json.load(f)
        existing_samples = data['examples']
    
    # Get example pairs
    anachronistic = existing_samples[0] if existing_samples[0]['target_scores']['Yes'] == 1 else existing_samples[1]
    plausible = existing_samples[1] if existing_samples[0]['target_scores']['Yes'] == 1 else existing_samples[0]
    
    category_descriptions = {
        'technology_displacement': "Historical figures using modern technology that didn't exist in their time",
        'temporal_displacement': "People from different historical eras interacting impossibly", 
        'cultural_anachronisms': "Historical figures engaging with modern culture, entertainment, or products",
        'scientific_anachronisms': "Historical figures using modern scientific knowledge or medical advances"
    }
    
    prompt = f"""Generate {batch_size} pairs of anachronism detection samples for {category}.

CATEGORY: {category_descriptions.get(category, category)}

EXACT EXAMPLES:
Anachronistic sample:
{json.dumps(anachronistic, indent=2)}

Plausible sample:
{json.dumps(plausible, indent=2)}

REQUIREMENTS:
1. Generate exactly {batch_size} pairs ({batch_size * 2} samples total)
2. Each pair has one anachronistic and one plausible version
3. Use EXACT JSON format as shown above
4. Focus on {category_descriptions.get(category, category)}
5. Make anachronisms clear but historically informed
6. Ensure plausible versions are completely accurate

Generate all {batch_size} pairs now:"""
    
    try:
        print(f"  Generating {batch_size} pairs for {category}...")
        response = client.chat.completions.create(
            model="o3",
            messages=[
                {"role": "system", "content": "Generate anachronism detection samples in exact JSON format. Focus on historical accuracy and clear temporal impossibilities."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=4000
        )
        
        response_text = response.choices[0].message.content
        samples = parse_response_correctly(response_text)
        print(f"  ✅ Generated {len(samples)} samples for {category}")
        return samples
        
    except Exception as e:
        print(f"  ❌ Failed to generate {category}: {e}")
        return []

def main():
    """Test production generation with 40 samples (4 categories × 10 pairs each)."""
    
    print("🧪 Small Production Test")
    print("Model: o3 | Target: 40 samples (20 pairs)")
    print("=" * 40)
    
    categories = {
        'technology_displacement': 10,  # 10 pairs = 20 samples
        'temporal_displacement': 5,     # 5 pairs = 10 samples  
        'cultural_anachronisms': 3,     # 3 pairs = 6 samples
        'scientific_anachronisms': 2    # 2 pairs = 4 samples
    }
    
    all_samples = []
    
    for category, batch_size in categories.items():
        print(f"\n--- {category.upper()} ---")
        category_samples = generate_category_batch(category, batch_size)
        all_samples.extend(category_samples)
        
        # Add delay between categories
        if category != list(categories.keys())[-1]:
            print(f"  ⏱️  Waiting 5 seconds...")
            time.sleep(5)
    
    print(f"\n🎉 Test Complete!")
    print(f"Generated: {len(all_samples)} samples")
    
    # Analyze results
    anachronistic = [s for s in all_samples if s['target_scores']['Yes'] == 1]
    plausible = [s for s in all_samples if s['target_scores']['No'] == 1]
    
    print(f"  Anachronistic: {len(anachronistic)} samples")
    print(f"  Plausible: {len(plausible)} samples")
    print(f"  Expected: 40 samples")
    
    if all_samples:
        # Save test results
        print(f"\n💾 Saving test samples...")
        test_data = {"examples": all_samples}
        with open('/Users/kyle/Documents/ws/post-hoc-reasoning/anachronisms_small_test.json', 'w') as f:
            json.dump(test_data, f, indent=2)
        
        print(f"✅ Saved: anachronisms_small_test.json ({len(all_samples)} samples)")
        
        # Show preview
        print(f"\n📋 Sample Preview:")
        for i, sample in enumerate(all_samples[:6]):
            sample_type = "Anachronistic" if sample['target_scores']['Yes'] == 1 else "Plausible"
            print(f"  {i+1}. [{sample_type}] {sample['input'][:60]}...")
        
        if len(all_samples) >= 35:  # Allow some variation
            print(f"\n✅ Success! System working correctly.")
            print(f"Ready for full production run.")
        else:
            print(f"\n⚠️  Generated {len(all_samples)} samples, expected ~40.")
            print(f"May need adjustments for full production.")
    else:
        print(f"\n❌ No samples generated. Check system configuration.")

if __name__ == "__main__":
    main()