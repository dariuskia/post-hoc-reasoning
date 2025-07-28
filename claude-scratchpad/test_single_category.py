#!/usr/bin/env python3
"""
Test script to generate samples for just one category
"""

import os
import sys

sys.path.append("/Users/kyle/Documents/ws/post-hoc-reasoning/claude-scratchpad")

from dotenv import load_dotenv
from generate_anachronisms_samples import AnachronismGenerator, load_existing_dataset

load_dotenv()


def main():
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found")
        return

    print("Initializing generator...")
    generator = AnachronismGenerator(api_key)

    print("Testing with technology_displacement category...")

    # Generate just 10 samples (5 pairs) for testing
    category = "technology_displacement"
    target_count = 10
    batch_size = 2  # Very small batch for testing

    samples = generator.generate_samples_by_category(category, target_count, batch_size)

    print(f"\nGenerated {len(samples)} samples:")
    for i, sample in enumerate(samples):
        print(f"\nSample {i+1}:")
        print(f"Input: {sample['input']}")
        print(f"Target: {sample['target_scores']}")


if __name__ == "__main__":
    main()
