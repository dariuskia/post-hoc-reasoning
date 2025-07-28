#!/usr/bin/env python3
"""
Full Production Anachronisms Generator - Corrected Version
Generates 974 new samples in manageable batches with checkpoint saving
"""

import json
import os
import openai
from dotenv import load_dotenv
import time
import random

load_dotenv()
random.seed(42)

class FullProductionGenerator:
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)
        self.batch_size = 10  # 10 pairs (20 samples) per batch
        self.checkpoint_file = '/Users/kyle/Documents/ws/post-hoc-reasoning/generation_checkpoint.json'
        
    def parse_response_correctly(self, response_text):
        """Parse o3 response into JSON samples."""
        samples = []
        lines = response_text.split('\n')
        current_json = ""
        brace_depth = 0
        in_json = False
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('{') and not in_json:
                in_json = True
                current_json = line
                brace_depth = line.count('{') - line.count('}')
                
                if brace_depth == 0:
                    try:
                        parsed = json.loads(current_json)
                        if 'input' in parsed and 'target_scores' in parsed:
                            samples.append(parsed)
                    except json.JSONDecodeError:
                        pass
                    in_json = False
                    current_json = ""
            
            elif in_json:
                current_json += '\n' + line
                brace_depth += line.count('{') - line.count('}')
                
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
    
    def load_checkpoint(self):
        """Load generation progress from checkpoint."""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {"completed_categories": [], "all_samples": []}
    
    def save_checkpoint(self, checkpoint_data):
        """Save generation progress to checkpoint."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    
    def generate_category_batch(self, category, batch_size=10):
        """Generate a batch for a specific category."""
        
        # Load existing samples for examples
        with open('/Users/kyle/Documents/ws/post-hoc-reasoning/data/anachronisms/anachronisms.json', 'r') as f:
            data = json.load(f)
            existing_samples = data['examples']
        
        # Get example pairs
        anachronistic = existing_samples[0] if existing_samples[0]['target_scores']['Yes'] == 1 else existing_samples[1]
        plausible = existing_samples[1] if existing_samples[0]['target_scores']['Yes'] == 1 else existing_samples[0]
        
        category_descriptions = {
            'technology_displacement': "Historical figures using modern technology that didn't exist in their time (computers, smartphones, internet, GPS, etc.)",
            'temporal_displacement': "People from different historical eras interacting impossibly (centuries apart)", 
            'cultural_anachronisms': "Historical figures engaging with modern culture, entertainment, lifestyle, or products",
            'scientific_anachronisms': "Historical figures using modern scientific knowledge, medical advances, or research methods",
            'institutional_anachronisms': "Historical figures interacting with modern institutions, laws, or organizational concepts"
        }
        
        prompt = f"""Generate {batch_size} pairs of anachronism detection samples for {category}.

CATEGORY: {category_descriptions.get(category, category)}

EXACT EXAMPLES FROM DATASET:
Anachronistic sample:
{json.dumps(anachronistic, indent=2)}

Plausible sample:
{json.dumps(plausible, indent=2)}

CRITICAL REQUIREMENTS:
1. Generate exactly {batch_size} pairs ({batch_size * 2} samples total)
2. Each pair has one anachronistic and one plausible version
3. Use EXACT JSON format as shown above
4. Focus specifically on {category_descriptions.get(category, category)}
5. Make anachronisms clear but historically informed
6. Ensure plausible versions are completely historically accurate
7. Use diverse historical figures and time periods
8. Vary the anachronistic elements within the category

Generate all {batch_size} pairs now:"""
        
        try:
            response = self.client.chat.completions.create(
                model="o3",
                messages=[
                    {"role": "system", "content": "You are an expert historian and dataset creator. Generate anachronism detection samples in exact JSON format. Focus on historical accuracy and clear temporal impossibilities."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=4000
            )
            
            response_text = response.choices[0].message.content
            samples = self.parse_response_correctly(response_text)
            return samples
            
        except Exception as e:
            print(f"    ❌ API error: {e}")
            return []
    
    def generate_category_samples(self, category, target_count):
        """Generate all samples for a category using batches."""
        
        print(f"\n=== GENERATING {category.upper()} ===")
        print(f"Target: {target_count} samples ({target_count//2} pairs)")
        
        samples = []
        pairs_needed = target_count // 2
        batches_needed = (pairs_needed + self.batch_size - 1) // self.batch_size
        
        print(f"Batches needed: {batches_needed} (batch size: {self.batch_size} pairs)")
        
        for batch_num in range(batches_needed):
            print(f"\n  🔄 Processing batch {batch_num + 1}/{batches_needed}...")
            
            # Calculate pairs needed for this batch
            remaining_pairs = pairs_needed - (len(samples) // 2)
            current_batch_pairs = min(self.batch_size, remaining_pairs)
            
            if current_batch_pairs <= 0:
                break
            
            # Generate batch
            batch_samples = self.generate_category_batch(category, current_batch_pairs)
            
            if batch_samples:
                samples.extend(batch_samples)
                print(f"    ✅ Generated {len(batch_samples)} samples ({len(batch_samples)//2} pairs)")
                print(f"    📊 Category progress: {len(samples)}/{target_count} samples")
            else:
                print(f"    ❌ Failed to generate batch {batch_num + 1}")
            
            # Add delay between batches
            if batch_num < batches_needed - 1:
                print(f"    ⏱️  Waiting 10 seconds...")
                time.sleep(10)
            
            if len(samples) >= target_count:
                break
        
        final_samples = samples[:target_count]
        print(f"\n  ✅ {category}: Generated {len(final_samples)}/{target_count} samples")
        return final_samples
    
    def run_full_production(self):
        """Run full production generation with checkpointing."""
        
        print("🚀 Full Production Anachronisms Generator")
        print("Model: o3 | Batch Size: 10 pairs | Target: 974 samples")
        print("=" * 60)
        
        # Load checkpoint
        checkpoint = self.load_checkpoint()
        completed_categories = set(checkpoint.get("completed_categories", []))
        all_samples = checkpoint.get("all_samples", [])
        
        print(f"Checkpoint loaded: {len(completed_categories)} categories completed, {len(all_samples)} samples")
        
        # Target distribution
        target_distribution = {
            'technology_displacement': 341,    # 35% - 171 pairs
            'temporal_displacement': 244,      # 25% - 122 pairs
            'cultural_anachronisms': 195,      # 20% - 98 pairs
            'scientific_anachronisms': 146,    # 15% - 73 pairs
            'institutional_anachronisms': 48   # 5% - 24 pairs
        }
        
        print(f"\n📊 Target Distribution: {sum(target_distribution.values())} samples")
        for category, count in target_distribution.items():
            status = "✅ COMPLETED" if category in completed_categories else "⏳ PENDING"
            print(f"  {category}: {count} samples - {status}")
        
        # Generate samples for remaining categories
        start_time = time.time()
        
        for category, target_count in target_distribution.items():
            if category in completed_categories:
                print(f"\n⏭️  Skipping {category} (already completed)")
                continue
            
            # Generate category samples
            category_samples = self.generate_category_samples(category, target_count)
            all_samples.extend(category_samples)
            
            # Update checkpoint
            completed_categories.add(category)
            checkpoint_data = {
                "completed_categories": list(completed_categories),
                "all_samples": all_samples
            }
            self.save_checkpoint(checkpoint_data)
            
            elapsed = time.time() - start_time
            remaining_categories = len(target_distribution) - len(completed_categories)
            
            print(f"\n📈 Progress: {len(completed_categories)}/{len(target_distribution)} categories completed")
            print(f"    Total samples: {len(all_samples)}/{sum(target_distribution.values())}")
            print(f"    Elapsed time: {elapsed/60:.1f} minutes")
            if remaining_categories > 0:
                print(f"    Estimated remaining: {(elapsed/len(completed_categories))*remaining_categories/60:.1f} minutes")
        
        # Shuffle and save final results
        random.shuffle(all_samples)
        
        print(f"\n🎉 Production generation complete!")
        print(f"Generated: {len(all_samples)} samples")
        
        # Save final files
        print(f"\n💾 Saving production files...")
        
        # Save new samples only
        new_samples_data = {"examples": all_samples}
        with open('/Users/kyle/Documents/ws/post-hoc-reasoning/anachronisms_new_samples.json', 'w') as f:
            json.dump(new_samples_data, f, indent=2)
        print(f"✅ Saved: anachronisms_new_samples.json ({len(all_samples)} samples)")
        
        # Create combined dataset
        with open('/Users/kyle/Documents/ws/post-hoc-reasoning/data/anachronisms/anachronisms.json', 'r') as f:
            original_data = json.load(f)
            original_samples = original_data['examples']
        
        combined_samples = original_samples + all_samples
        random.shuffle(combined_samples)
        
        combined_data = {"examples": combined_samples}
        with open('/Users/kyle/Documents/ws/post-hoc-reasoning/anachronisms_new_full.json', 'w') as f:
            json.dump(combined_data, f, indent=2)
        print(f"✅ Saved: anachronisms_new_full.json ({len(combined_samples)} samples)")
        
        # Clean up checkpoint
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
            print(f"🧹 Cleaned up checkpoint file")
        
        # Final summary
        print(f"\n🎯 PRODUCTION SUMMARY")
        print(f"Original dataset: {len(original_samples)} samples")
        print(f"New samples: {len(all_samples)} samples")
        print(f"Final dataset: {len(combined_samples)} samples")
        print(f"Expansion: {len(original_samples)} → {len(combined_samples)} samples")
        
        return all_samples

def main():
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found")
        return
    
    # Run full production
    generator = FullProductionGenerator(api_key)
    generator.run_full_production()

if __name__ == "__main__":
    main()