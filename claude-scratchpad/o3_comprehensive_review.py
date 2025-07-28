#!/usr/bin/env python3
"""
Comprehensive O3 Review System

Reviews ALL anachronism samples and flags problematic ones for deletion.
"""

import json
import os
import random
import time
from typing import Dict, List, Tuple

import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ComprehensiveO3Reviewer:
    """Comprehensive review system using o3 to flag problematic samples."""

    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)

    def review_single_sample(self, sample: Dict, sample_id: str) -> Dict:
        """Review a single sample for quality issues."""

        sample_type = (
            "ANACHRONISTIC" if sample["target_scores"]["Yes"] == 1 else "PLAUSIBLE"
        )

        review_prompt = f"""You are an expert historian and dataset quality reviewer. Please carefully evaluate this single anachronism detection sample.

SAMPLE TYPE: {sample_type}
SAMPLE:
{json.dumps(sample, indent=2)}

For this {sample_type.lower()} sample, please check for the following issues:

{"FOR ANACHRONISTIC SAMPLES:" if sample_type == "ANACHRONISTIC" else "FOR PLAUSIBLE SAMPLES:"}
{'''1. **Clear Anachronism**: Is the temporal impossibility obvious and unambiguous?
2. **Historical Context**: Is the historical figure/period referenced correctly?
3. **Modern Element**: Is the anachronistic element clearly from a later time period?''' if sample_type == "ANACHRONISTIC" else '''1. **Historical Accuracy**: Are all historical facts, dates, people, and contexts accurate?
2. **Timeline Consistency**: Do all referenced people/events actually overlap in time?
3. **Factual Correctness**: Are there any subtle historical inaccuracies?'''}

UNIVERSAL CHECKS (for both types):
4. **Linguistic Quality**: Is the sentence well-formed, natural, and fluent?
5. **Format Compliance**: Does it follow the exact JSON format requirements?
6. **Appropriateness**: Is this suitable for an anachronism detection task?

Please identify ANY problems and provide a clear recommendation:

**SEVERITY LEVELS:**
- **CRITICAL**: Major historical inaccuracy, format error, or completely inappropriate
- **MODERATE**: Minor historical issues, awkward phrasing, or clarity problems  
- **MINOR**: Very small issues that don't affect functionality
- **NONE**: No issues found

Format your response as JSON:
{{
  "severity": "CRITICAL/MODERATE/MINOR/NONE",
  "recommendation": "DELETE/FLAG_FOR_REVIEW/APPROVE",
  "issues_found": [
    {{
      "type": "historical_accuracy/anachronism_validity/linguistic_quality/format_compliance/appropriateness",
      "description": "Specific description of the issue",
      "severity": "CRITICAL/MODERATE/MINOR"
    }}
  ],
  "historical_accuracy_score": X,
  "overall_quality_score": X,
  "detailed_feedback": "Specific explanation...",
  "suggested_fix": "If fixable, suggest how to correct it"
}}"""

        try:
            response = self.client.chat.completions.create(
                model="o3-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert historian and strict quality reviewer. Flag ANY historical inaccuracies or quality issues, no matter how minor.",
                    },
                    {"role": "user", "content": review_prompt},
                ],
                max_completion_tokens=1000,
            )

            # Parse the JSON response
            review_text = response.choices[0].message.content.strip()

            try:
                # Find JSON block in response
                start_idx = review_text.find("{")
                end_idx = review_text.rfind("}") + 1
                json_str = review_text[start_idx:end_idx]
                review_data = json.loads(json_str)

                # Add metadata
                review_data["sample_id"] = sample_id
                review_data["sample_type"] = sample_type
                review_data["original_sample"] = sample

                return review_data

            except json.JSONDecodeError:
                return {
                    "sample_id": sample_id,
                    "sample_type": sample_type,
                    "error": "Failed to parse review response",
                    "raw_response": review_text,
                    "severity": "CRITICAL",
                    "recommendation": "FLAG_FOR_REVIEW",
                }

        except Exception as e:
            return {
                "sample_id": sample_id,
                "sample_type": sample_type,
                "error": f"API call failed: {e}",
                "severity": "CRITICAL",
                "recommendation": "FLAG_FOR_REVIEW",
            }

    def review_all_samples(self, samples: List[Dict]) -> List[Dict]:
        """Review all samples and flag problematic ones."""

        reviews = []

        for i, sample in enumerate(samples):
            sample_id = f"sample_{i+1:03d}"
            print(f"Reviewing {sample_id} ({i+1}/{len(samples)})...")

            review = self.review_single_sample(sample, sample_id)
            reviews.append(review)

            # Add delay between reviews to avoid rate limiting
            if i < len(samples) - 1:
                time.sleep(2)

        return reviews

    def analyze_and_filter_samples(
        self, reviews: List[Dict], samples: List[Dict]
    ) -> Dict:
        """Analyze reviews and create filtered sample list."""

        # Categorize samples by recommendation
        to_delete = []
        to_flag = []
        approved = []

        for review in reviews:
            sample_idx = int(review["sample_id"].split("_")[1]) - 1

            if review["recommendation"] == "DELETE":
                to_delete.append((sample_idx, review))
            elif review["recommendation"] == "FLAG_FOR_REVIEW":
                to_flag.append((sample_idx, review))
            else:  # APPROVE
                approved.append((sample_idx, review))

        # Create filtered samples (removing problematic ones)
        good_samples = []
        deleted_samples = []
        flagged_samples = []

        for i, sample in enumerate(samples):
            review = reviews[i]

            if review["recommendation"] == "DELETE":
                deleted_samples.append(
                    {
                        "sample": sample,
                        "review": review,
                        "reason": "DELETED - "
                        + review.get("detailed_feedback", "Quality issues"),
                    }
                )
            elif review["recommendation"] == "FLAG_FOR_REVIEW":
                flagged_samples.append(
                    {
                        "sample": sample,
                        "review": review,
                        "reason": "FLAGGED - "
                        + review.get("detailed_feedback", "Needs manual review"),
                    }
                )
            else:
                good_samples.append(sample)

        return {
            "original_count": len(samples),
            "approved_count": len(good_samples),
            "deleted_count": len(deleted_samples),
            "flagged_count": len(flagged_samples),
            "good_samples": good_samples,
            "deleted_samples": deleted_samples,
            "flagged_samples": flagged_samples,
            "all_reviews": reviews,
        }


def load_demo_samples() -> List[Dict]:
    """Load the demo samples for comprehensive review."""
    with open(
        "/Users/kyle/Documents/ws/post-hoc-reasoning/anachronisms_demo_samples.json",
        "r",
    ) as f:
        data = json.load(f)
        return data["examples"]


def main():
    """Run comprehensive o3 review of all anachronism samples."""

    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found")
        return

    print("🔍 Comprehensive O3 Review - Flagging Problematic Samples")
    print("=" * 60)

    # Load demo samples
    print("Loading demo samples...")
    try:
        samples = load_demo_samples()
        print(f"Loaded {len(samples)} samples for comprehensive review")
    except FileNotFoundError:
        print(
            "❌ anachronisms_demo_samples.json not found. Please generate demo samples first."
        )
        return

    # Initialize review system
    reviewer = ComprehensiveO3Reviewer(api_key)

    # Review all samples
    print(f"\n🧐 Conducting comprehensive O3 review of all {len(samples)} samples...")
    print("This will take several minutes...")

    all_reviews = reviewer.review_all_samples(samples)

    # Analyze and filter
    print("\n📊 Analyzing results and filtering samples...")
    results = reviewer.analyze_and_filter_samples(all_reviews, samples)

    # Print summary
    print(f"\n✅ Comprehensive Review Complete!")
    print(f"📈 Summary:")
    print(f"  Original samples: {results['original_count']}")
    print(f"  ✅ Approved: {results['approved_count']}")
    print(f"  ⚠️  Flagged for review: {results['flagged_count']}")
    print(f"  ❌ Marked for deletion: {results['deleted_count']}")

    # Show problematic samples
    if results["deleted_count"] > 0:
        print(f"\n❌ SAMPLES MARKED FOR DELETION:")
        for i, item in enumerate(results["deleted_samples"], 1):
            print(f"  {i}. {item['reason']}")
            print(f"     Input: {item['sample']['input'][:100]}...")

    if results["flagged_count"] > 0:
        print(f"\n⚠️  SAMPLES FLAGGED FOR REVIEW:")
        for i, item in enumerate(results["flagged_samples"], 1):
            print(f"  {i}. {item['reason']}")
            print(f"     Input: {item['sample']['input'][:100]}...")

    # Save comprehensive results
    comprehensive_results = {
        "metadata": {
            "review_date": "2025-07-28",
            "reviewer": "o3-mini",
            "review_type": "comprehensive_quality_check",
            "total_samples_reviewed": len(samples),
        },
        "summary": {
            "original_count": results["original_count"],
            "approved_count": results["approved_count"],
            "deleted_count": results["deleted_count"],
            "flagged_count": results["flagged_count"],
            "approval_rate": results["approved_count"] / results["original_count"],
        },
        "detailed_results": results,
    }

    print(f"\n💾 Saving comprehensive review results...")
    with open(
        "/Users/kyle/Documents/ws/post-hoc-reasoning/anachronisms_comprehensive_review.json",
        "w",
    ) as f:
        json.dump(comprehensive_results, f, indent=2)

    # Save cleaned dataset
    if results["good_samples"]:
        cleaned_data = {"examples": results["good_samples"]}
        print(
            f"💾 Saving cleaned dataset with {len(results['good_samples'])} approved samples..."
        )
        with open(
            "/Users/kyle/Documents/ws/post-hoc-reasoning/anachronisms_cleaned_samples.json",
            "w",
        ) as f:
            json.dump(cleaned_data, f, indent=2)

    print(f"\n📄 Files created:")
    print(f"  - anachronisms_comprehensive_review.json (detailed review results)")
    print(
        f"  - anachronisms_cleaned_samples.json ({results['approved_count']} clean samples)"
    )

    print(f"\n🎯 Quality Rate: {results['approval_rate']:.1%}")


if __name__ == "__main__":
    main()
