#!/usr/bin/env python3
"""
Simple script to clean anachronism samples based on O3 review
"""

import json


def main():
    print("🧹 Cleaning Anachronism Samples")
    print("=" * 40)

    # Load the comprehensive review
    with open(
        "/Users/kyle/Documents/ws/post-hoc-reasoning/anachronisms_comprehensive_review.json",
        "r",
    ) as f:
        review_data = json.load(f)

    # Load original samples
    with open(
        "/Users/kyle/Documents/ws/post-hoc-reasoning/anachronisms_demo_samples.json",
        "r",
    ) as f:
        original_data = json.load(f)
        original_samples = original_data["examples"]

    print(f"Original samples: {len(original_samples)}")

    # Get the flagged samples details
    flagged_samples = review_data["detailed_results"]["flagged_samples"]

    print(f"Flagged samples: {len(flagged_samples)}")
    print("\\nFlagged samples to review:")

    # Manually identify the problematic samples based on the review
    samples_to_remove = []

    for i, item in enumerate(flagged_samples):
        sample_text = item["sample"]["input"]
        print(f"\\n{i+1}. {sample_text}")

        # Check for known problematic patterns
        if "Suleiman the Magnificent and Bayezid II" in sample_text:
            print(
                "   ❌ ISSUE: Historical inaccuracy - these rulers were not contemporaries"
            )
            samples_to_remove.append(sample_text)
        elif (
            "Leonardo da Vinci unveiled his groundbreaking anatomical study"
            in sample_text
        ):
            print(
                "   ⚠️  MINOR: Phrasing suggests public unveiling (was actually private)"
            )
            # Keep this one - minor issue only
        elif (
            "Leonardo da Vinci exchanged innovative ideas on engineering with his contemporary, Michelangelo"
            in sample_text
        ):
            print(
                "   ⚠️  MINOR: They were contemporaries but collaboration is speculative"
            )
            # Keep this one - historically plausible
        else:
            print("   ❓ Unknown issue - needs manual review")

    # Create cleaned dataset
    cleaned_samples = []
    removed_count = 0

    for sample in original_samples:
        if sample["input"] not in samples_to_remove:
            cleaned_samples.append(sample)
        else:
            removed_count += 1
            print(f"\\n❌ REMOVING: {sample['input'][:80]}...")

    print(f"\\n✅ Cleaning complete!")
    print(f"Original: {len(original_samples)} samples")
    print(f"Removed: {removed_count} samples")
    print(f"Final: {len(cleaned_samples)} samples")

    # Save cleaned dataset
    cleaned_data = {"examples": cleaned_samples}
    with open(
        "/Users/kyle/Documents/ws/post-hoc-reasoning/anachronisms_final_clean.json", "w"
    ) as f:
        json.dump(cleaned_data, f, indent=2)

    print(f"\\n💾 Saved cleaned dataset: anachronisms_final_clean.json")

    # Show some examples from cleaned dataset
    print(f"\\n📋 Sample of cleaned data:")
    for i, sample in enumerate(cleaned_samples[:4]):
        sample_type = (
            "Anachronistic" if sample["target_scores"]["Yes"] == 1 else "Plausible"
        )
        print(f"{i+1}. [{sample_type}] {sample['input'][:80]}...")


if __name__ == "__main__":
    main()
