#!/usr/bin/env python3
"""
Demo Anachronisms Generator - Creates smaller sample for demonstration

Generates 100 new anachronisms samples as a proof of concept.
"""

import json
import os
import random
import time
from typing import Dict, List

import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set seed for reproducibility
random.seed(42)

# Import the main generator class
import sys

sys.path.append("/Users/kyle/Documents/ws/post-hoc-reasoning/claude-scratchpad")
from generate_anachronisms_samples import AnachronismGenerator, load_existing_dataset


def main():
    """Generate 100 new anachronisms samples for demonstration."""

    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        return

    print("Initializing Demo Anachronisms Generator...")
    generator = AnachronismGenerator(api_key)

    print("Loading existing dataset...")
    existing_samples = load_existing_dataset()
    print(f"Found {len(existing_samples)} existing samples")

    # Smaller target distribution for demo (20 total samples)
    target_distribution = {
        "technology_displacement": 8,  # 40%
        "temporal_displacement": 6,  # 30%
        "cultural_anachronisms": 4,  # 20%
        "scientific_anachronisms": 2,  # 10%
    }

    print(f"Demo target generation: {sum(target_distribution.values())} samples")
    print("Distribution:", target_distribution)

    # Generate samples for each category using small batches
    all_new_samples = []
    batch_size = 2  # Small batches for demo

    for category, target_count in target_distribution.items():
        print(f"\n=== GENERATING {category.upper()} ===")
        category_samples = generator.generate_samples_by_category(
            category, target_count, batch_size
        )
        all_new_samples.extend(category_samples)
        print(
            f"✅ Generated {len(category_samples)}/{target_count} samples for {category}"
        )

    # Shuffle the combined samples
    random.shuffle(all_new_samples)

    print(f"\n🎉 Generated {len(all_new_samples)} total new samples")

    # Save demo samples
    print("Saving anachronisms_demo_samples.json...")
    demo_samples_data = {"examples": all_new_samples}
    with open(
        "/Users/kyle/Documents/ws/post-hoc-reasoning/anachronisms_demo_samples.json",
        "w",
    ) as f:
        json.dump(demo_samples_data, f, indent=2)

    # Create demo combined dataset
    print("Creating demo combined dataset...")
    combined_samples = existing_samples + all_new_samples
    random.shuffle(combined_samples)

    print(f"Demo combined dataset has {len(combined_samples)} samples")

    # Save demo combined dataset
    print("Saving anachronisms_demo_full.json...")
    combined_data = {"examples": combined_samples}
    with open(
        "/Users/kyle/Documents/ws/post-hoc-reasoning/anachronisms_demo_full.json", "w"
    ) as f:
        json.dump(combined_data, f, indent=2)

    print("\n🎯 Demo generation complete!")
    print(f"Files created:")
    print(f"- anachronisms_demo_samples.json ({len(all_new_samples)} new samples)")
    print(f"- anachronisms_demo_full.json ({len(combined_samples)} total samples)")

    # Show some example samples
    print(f"\n📋 Sample Examples:")
    for i, sample in enumerate(all_new_samples[:4]):
        print(f"\nExample {i+1}:")
        print(f"Input: {sample['input']}")
        print(
            f"Anachronistic: {'Yes' if sample['target_scores']['Yes'] == 1 else 'No'}"
        )

    # Summary statistics
    print(f"\n📊 Generation Summary:")
    for category, target in target_distribution.items():
        actual = len(
            [
                s
                for s in all_new_samples
                if "metadata" in s and s.get("metadata", {}).get("category") == category
            ]
        )
        print(
            f"{category}: {len([s for s in all_new_samples if i % 2 == (1 if s['target_scores']['Yes'] == 1 else 0) for i, s in enumerate(all_new_samples)])} samples"
        )

    print(
        f"\n✨ Demo successfully expanded dataset from {len(existing_samples)} to {len(combined_samples)} samples!"
    )


if __name__ == "__main__":
    main()
