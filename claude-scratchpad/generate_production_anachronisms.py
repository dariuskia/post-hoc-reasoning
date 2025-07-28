#!/usr/bin/env python3
"""
Production Anachronisms Dataset Generator

Generates 974 new anachronisms samples using o3 model
with production-scale batch processing (50 pairs per batch).

Target: 974 new samples (expand from 226 to 1200 total)
"""

import json
import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set seed for reproducibility
random.seed(42)


class ProductionAnachronismGenerator:
    """Production-scale generator using o3 with large batch processing."""

    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.generation_batch_size = 10  # PRODUCTION: 10 pairs (20 samples) per batch
        self.validation_cache = {}

        # Load component databases
        with open(
            "/Users/kyle/Documents/ws/post-hoc-reasoning/claude-scratchpad/anachronisms_components.json",
            "r",
        ) as f:
            self.components = json.load(f)

    def get_example_pairs(
        self, existing_samples: List[Dict], category: str, count: int = 3
    ) -> List[Tuple[Dict, Dict]]:
        """Get example pairs from existing dataset that match the category."""

        pairs = []
        for i in range(0, len(existing_samples) - 1, 2):
            if i + 1 >= len(existing_samples):
                break

            anachronistic = (
                existing_samples[i]
                if existing_samples[i]["target_scores"]["Yes"] == 1
                else existing_samples[i + 1]
            )
            plausible = (
                existing_samples[i + 1]
                if existing_samples[i]["target_scores"]["Yes"] == 1
                else existing_samples[i]
            )

            # Simple category matching based on content
            if self._sample_matches_category(anachronistic["input"], category):
                pairs.append((anachronistic, plausible))
                if len(pairs) >= count:
                    break

        return pairs

    def _sample_matches_category(self, text: str, category: str) -> bool:
        """Check if a sample matches the given category."""
        text_lower = text.lower()

        if category == "technology_displacement":
            tech_words = [
                "computer",
                "laptop",
                "phone",
                "internet",
                "gps",
                "digital",
                "electronic",
                "radio",
                "television",
                "streaming",
                "app",
            ]
            return any(word in text_lower for word in tech_words)
        elif category == "temporal_displacement":
            return any(
                name in text
                for name in [
                    "Einstein",
                    "Newton",
                    "Darwin",
                    "Franklin",
                    "Washington",
                    "Shakespeare",
                    "Leonardo",
                ]
            )
        elif category == "cultural_anachronisms":
            cultural_words = [
                "fan",
                "music",
                "movie",
                "game",
                "sport",
                "food",
                "entertainment",
                "favorite",
            ]
            return any(word in text_lower for word in cultural_words)
        elif category == "scientific_anachronisms":
            science_words = [
                "dna",
                "genetic",
                "nuclear",
                "atomic",
                "vaccine",
                "antibiotic",
                "medical",
                "mri",
                "scan",
            ]
            return any(word in text_lower for word in science_words)
        else:  # institutional_anachronisms
            inst_words = [
                "constitution",
                "democracy",
                "organization",
                "institution",
                "law",
                "legal",
                "vote",
                "election",
            ]
            return any(word in text_lower for word in inst_words)

    def create_production_batch_prompt(
        self, category: str, batch_size: int, example_pairs: List[Tuple[Dict, Dict]]
    ) -> str:
        """Create a comprehensive prompt for large-scale batch generation using o3."""

        # Format example pairs to show exact structure
        examples_text = "EXACT EXAMPLES FROM THE DATASET:\\n\\n"
        for i, (anachronistic, plausible) in enumerate(example_pairs, 1):
            examples_text += f"Example {i}:\\n"
            examples_text += (
                f"Anachronistic version:\\n{json.dumps(anachronistic, indent=2)}\\n\\n"
            )
            examples_text += (
                f"Plausible version:\\n{json.dumps(plausible, indent=2)}\\n\\n"
            )

        category_descriptions = {
            "technology_displacement": "Historical figures using modern technology that didn't exist in their time (computers, smartphones, internet, etc.)",
            "temporal_displacement": "People from different historical eras interacting impossibly (centuries apart)",
            "cultural_anachronisms": "Historical figures engaging with modern culture, entertainment, lifestyle, or products",
            "scientific_anachronisms": "Historical figures using modern scientific knowledge, medical advances, or research methods",
            "institutional_anachronisms": "Historical figures interacting with modern institutions, laws, or organizational concepts",
        }

        prompt = f"""TASK: Generate {batch_size} pairs of anachronism detection samples for the category: {category}

CATEGORY DESCRIPTION: {category_descriptions.get(category, category)}

{examples_text}

CRITICAL REQUIREMENTS FOR ALL {batch_size} PAIRS:
1. EXACT FORMAT: Each sample must have EXACTLY the same JSON structure as the examples above
2. EXACT FIELD NAMES: Use "input" and "target_scores" with "Yes" and "No" keys only
3. EXACT SCORING: Anachronistic samples get {{"Yes": 1, "No": 0}}, plausible samples get {{"Yes": 0, "No": 1}}
4. PAIRED STRUCTURE: Generate pairs where one version is anachronistic and one is historically plausible
5. SIMILAR LENGTH: Keep sentences roughly the same length as the examples (30-200 words)
6. NATURAL LANGUAGE: Write in natural, fluent English that sounds realistic and professional
7. CLEAR ANACHRONISMS: Make the temporal impossibility obvious to someone with historical knowledge
8. HISTORICAL ACCURACY: Ensure the plausible versions are factually correct and chronologically possible
9. VARIETY: Use different historical figures, time periods, and scenarios for each pair
10. QUALITY: Each pair should be publication-ready for the dataset

SPECIFIC INSTRUCTIONS FOR {category.upper()}:
- Focus on {category_descriptions.get(category, category)}
- Use diverse historical figures from different time periods (Ancient, Medieval, Renaissance, Modern)
- Vary the anachronistic elements to cover different aspects of {category.replace('_', ' ')}
- Ensure both versions of each pair cover the same basic scenario with only the anachronistic element changed
- Make anachronisms clear but not cartoonishly obvious
- Ensure plausible versions are historically accurate and well-researched

QUALITY STANDARDS:
- Historical figures must be real and correctly referenced
- Time periods must be accurate
- Anachronistic elements must be clearly from wrong time periods
- Language must be natural and professional
- All samples must follow the exact JSON format
- Each pair must effectively test anachronism detection skills

OUTPUT FORMAT:
Generate exactly {batch_size} pairs. For each pair, output the anachronistic version first, then the plausible version. Use the EXACT JSON format shown in the examples.

Begin generation of all {batch_size} pairs now:"""

        return prompt

    def generate_production_batch(
        self, category: str, batch_size: int = 50
    ) -> List[Dict]:
        """Generate a production batch using o3 model."""

        # Load existing samples to show as examples
        existing_samples = load_existing_dataset()
        example_pairs = self.get_example_pairs(existing_samples, category, 3)

        prompt = self.create_production_batch_prompt(
            category, batch_size, example_pairs
        )

        try:
            print(f"    Sending request to o3 for {batch_size} pairs...")
            response = self.client.chat.completions.create(
                model="o3",  # PRODUCTION: Using full o3 model
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert historian and dataset creator. You must generate anachronism detection samples that exactly match the format and style of the provided examples. Focus on historical accuracy and clear temporal impossibilities.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=8000,  # PRODUCTION: Increased for large batches
            )

            print(f"    Received response, parsing...")
            parsed_samples = self._parse_production_response(
                response.choices[0].message.content, category
            )
            print(f"    Successfully parsed {len(parsed_samples)} samples from batch")
            return parsed_samples

        except Exception as e:
            print(f"    ERROR: Failed to generate batch: {e}")
            return []

    def _parse_production_response(
        self, response_text: str, category: str
    ) -> List[Dict]:
        """Parse the production response from o3 into individual samples."""

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

    def generate_category_samples(self, category: str, target_count: int) -> List[Dict]:
        """Generate samples for a specific category using production batching."""

        print(f"\\n=== GENERATING {category.upper()} ===")
        print(f"Target: {target_count} samples ({target_count//2} pairs)")

        samples = []
        pairs_needed = target_count // 2
        batches_needed = (
            pairs_needed + self.generation_batch_size - 1
        ) // self.generation_batch_size

        print(
            f"Batches needed: {batches_needed} (batch size: {self.generation_batch_size} pairs)"
        )

        for batch_num in range(batches_needed):
            print(f"\\n  🔄 Processing batch {batch_num + 1}/{batches_needed}...")

            # Calculate pairs needed for this batch
            remaining_pairs = pairs_needed - (len(samples) // 2)
            current_batch_pairs = min(self.generation_batch_size, remaining_pairs)

            if current_batch_pairs <= 0:
                break

            # Generate batch using o3
            batch_samples = self.generate_production_batch(
                category, current_batch_pairs
            )

            if batch_samples:
                samples.extend(batch_samples)
                print(
                    f"    ✅ Generated {len(batch_samples)} samples ({len(batch_samples)//2} pairs)"
                )
            else:
                print(f"    ❌ Failed to generate batch {batch_num + 1}")

            # Add delay between batches to avoid rate limiting
            if batch_num < batches_needed - 1:
                print(f"    ⏱️  Waiting 10 seconds before next batch...")
                time.sleep(10)

            if len(samples) >= target_count:
                break

        final_samples = samples[:target_count]
        print(
            f"\\n  ✅ {category}: Generated {len(final_samples)}/{target_count} samples"
        )
        return final_samples


def load_existing_dataset() -> List[Dict]:
    """Load existing anachronisms dataset."""
    with open(
        "/Users/kyle/Documents/ws/post-hoc-reasoning/data/anachronisms/anachronisms.json",
        "r",
    ) as f:
        data = json.load(f)
        return data["examples"]


def main():
    """Generate 974 new anachronisms samples using production-scale o3 processing."""

    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in .env file")
        return

    print("🚀 Production Anachronisms Generator")
    print("Model: o3 | Batch Size: 10 pairs | Target: 974 samples")
    print("=" * 60)

    # Initialize production generator
    print("Initializing production generator...")
    generator = ProductionAnachronismGenerator(api_key)

    print("Loading existing dataset...")
    existing_samples = load_existing_dataset()
    print(f"Found {len(existing_samples)} existing samples")

    # Production target distribution for 974 new samples
    target_distribution = {
        "technology_displacement": 341,  # 35% - 171 pairs
        "temporal_displacement": 244,  # 25% - 122 pairs
        "cultural_anachronisms": 195,  # 20% - 98 pairs
        "scientific_anachronisms": 146,  # 15% - 73 pairs
        "institutional_anachronisms": 48,  # 5% - 24 pairs
    }

    print(f"\\n📊 Production Target: {sum(target_distribution.values())} samples")
    print("Distribution:")
    for category, count in target_distribution.items():
        pairs = count // 2
        batches = (pairs + 9) // 10  # Round up for batch calculation  
        print(f"  {category}: {count} samples ({pairs} pairs, {batches} batches)")

    # Generate samples for each category
    all_new_samples = []
    total_batches = sum(
        (count // 2 + 9) // 10 for count in target_distribution.values()
    )
    current_batch = 0

    print(f"\\n🔥 Starting production generation ({total_batches} total batches)...")

    for category, target_count in target_distribution.items():
        category_samples = generator.generate_category_samples(category, target_count)
        all_new_samples.extend(category_samples)

        current_batch += (target_count // 2 + 9) // 10
        progress = (current_batch / total_batches) * 100
        print(
            f"\\n📈 Progress: {progress:.1f}% ({len(all_new_samples)}/{sum(target_distribution.values())} samples)"
        )

    # Shuffle the combined samples
    random.shuffle(all_new_samples)

    print(f"\\n🎉 Production generation complete!")
    print(f"Generated: {len(all_new_samples)} samples")

    # Save new samples
    print("\\n💾 Saving production files...")

    # Save new samples only
    new_samples_data = {"examples": all_new_samples}
    with open(
        "/Users/kyle/Documents/ws/post-hoc-reasoning/anachronisms_new_samples.json", "w"
    ) as f:
        json.dump(new_samples_data, f, indent=2)
    print(f"✅ Saved: anachronisms_new_samples.json ({len(all_new_samples)} samples)")

    # Create combined dataset
    combined_samples = existing_samples + all_new_samples
    random.shuffle(combined_samples)

    combined_data = {"examples": combined_samples}
    with open(
        "/Users/kyle/Documents/ws/post-hoc-reasoning/anachronisms_new_full.json", "w"
    ) as f:
        json.dump(combined_data, f, indent=2)
    print(f"✅ Saved: anachronisms_new_full.json ({len(combined_samples)} samples)")

    # Final summary
    print(f"\\n🎯 PRODUCTION SUMMARY")
    print(f"Original dataset: {len(existing_samples)} samples")
    print(f"New samples: {len(all_new_samples)} samples")
    print(f"Final dataset: {len(combined_samples)} samples")
    print(f"Expansion: {len(existing_samples)} → {len(combined_samples)} samples")

    # Category breakdown
    print(f"\\n📊 Final Distribution:")
    for category, target in target_distribution.items():
        actual = len(
            [
                s
                for s in all_new_samples
                if generator._sample_matches_category(s["input"], category)
            ]
        )
        print(f"  {category}: {actual}/{target} samples ({actual/target*100:.1f}%)")


if __name__ == "__main__":
    main()
