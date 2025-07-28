#!/usr/bin/env python3
"""
O3 Full Review and Filter System

Reviews ALL 972 anachronism samples with o3 and creates filtered output files:
- *_reviewed_rejected.json: All rejected samples
- *_reviewed_new_samples.json: All approved new samples  
- *_reviewed_full.json: Original samples + approved new samples
"""

import json
import os
import time
from typing import Dict, List

import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class O3FullReviewSystem:
    """Full review system using o3 for comprehensive quality validation."""

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
- **CRITICAL**: Major historical inaccuracy, format error, or completely inappropriate → REJECT
- **MODERATE**: Minor historical issues, awkward phrasing, or clarity problems → REJECT  
- **MINOR**: Very small issues that don't affect functionality → APPROVE
- **NONE**: No issues found → APPROVE

Format your response as JSON:
{{
  "severity": "CRITICAL/MODERATE/MINOR/NONE",
  "recommendation": "REJECT/APPROVE",
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
                        "content": "You are an expert historian and strict quality reviewer. Flag ANY historical inaccuracies or quality issues. Be thorough but not overly harsh - only reject samples with clear problems.",
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
                    "recommendation": "REJECT",
                }

        except Exception as e:
            return {
                "sample_id": sample_id,
                "sample_type": sample_type,
                "error": f"API call failed: {e}",
                "severity": "CRITICAL",
                "recommendation": "REJECT",
            }

    def review_all_samples(self, samples: List[Dict]) -> List[Dict]:
        """Review all samples and flag problematic ones."""

        reviews = []
        total_samples = len(samples)

        for i, sample in enumerate(samples):
            sample_id = f"sample_{i+1:04d}"
            print(f"Reviewing {sample_id} ({i+1}/{total_samples})...")

            review = self.review_single_sample(sample, sample_id)
            reviews.append(review)

            # Add delay between reviews to avoid rate limiting
            if i < len(samples) - 1:
                # Progress update every 50 samples
                if (i + 1) % 50 == 0:
                    approved = len([r for r in reviews if r.get("recommendation") == "APPROVE"])
                    rejected = len([r for r in reviews if r.get("recommendation") == "REJECT"])
                    print(f"  Progress: {i+1}/{total_samples} | Approved: {approved} | Rejected: {rejected}")
                
                time.sleep(1)  # 1 second delay between reviews

        return reviews

    def analyze_and_filter_samples(
        self, reviews: List[Dict], samples: List[Dict]
    ) -> Dict:
        """Analyze reviews and create filtered sample lists."""

        # Categorize samples by recommendation
        approved_samples = []
        rejected_samples = []

        for i, review in enumerate(reviews):
            sample = samples[i]
            
            if review.get("recommendation") == "APPROVE":
                approved_samples.append(sample)
            else:  # REJECT or error
                rejected_samples.append({
                    "sample": sample,
                    "review": review,
                    "reason": review.get("detailed_feedback", "Quality issues detected"),
                    "sample_id": review.get("sample_id", f"sample_{i+1:04d}")
                })

        return {
            "original_count": len(samples),
            "approved_count": len(approved_samples),
            "rejected_count": len(rejected_samples),
            "approved_samples": approved_samples,
            "rejected_samples": rejected_samples,
            "all_reviews": reviews,
        }


def load_new_samples() -> List[Dict]:
    """Load the new samples for comprehensive review."""
    with open(
        "/Users/kyle/Documents/ws/post-hoc-reasoning/anachronisms_new_samples.json",
        "r",
    ) as f:
        data = json.load(f)
        return data["examples"]


def load_original_samples() -> List[Dict]:
    """Load the original anachronism samples."""
    try:
        with open(
            "/Users/kyle/Documents/ws/post-hoc-reasoning/anachronisms_new_full.json",
            "r",
        ) as f:
            data = json.load(f)
            return data["examples"]
    except FileNotFoundError:
        print("Warning: anachronisms_new_full.json not found. Using empty original set.")
        return []


def main():
    """Run comprehensive o3 review of all anachronism samples."""

    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found")
        return

    print("🔍 O3 Full Review and Filter System")
    print("=" * 60)

    # Load new samples
    print("Loading new samples...")
    try:
        new_samples = load_new_samples()
        print(f"Loaded {len(new_samples)} new samples for comprehensive review")
    except FileNotFoundError:
        print("❌ anachronisms_new_samples.json not found.")
        return

    # Load original samples
    print("Loading original samples...")
    original_samples = load_original_samples()
    print(f"Loaded {len(original_samples)} original samples")

    # Initialize review system
    reviewer = O3FullReviewSystem(api_key)

    # Review all new samples
    print(f"\n🧐 Conducting comprehensive O3 review of all {len(new_samples)} new samples...")
    print("This will take considerable time (estimated: ~30-40 minutes)...")
    
    start_time = time.time()
    all_reviews = reviewer.review_all_samples(new_samples)
    end_time = time.time()
    
    print(f"\n⏱️ Review completed in {(end_time - start_time)/60:.1f} minutes")

    # Analyze and filter
    print("\n📊 Analyzing results and filtering samples...")
    results = reviewer.analyze_and_filter_samples(all_reviews, new_samples)

    # Print summary
    print(f"\n✅ Comprehensive Review Complete!")
    print(f"📈 Summary:")
    print(f"  Original new samples: {results['original_count']}")
    print(f"  ✅ Approved: {results['approved_count']}")
    print(f"  ❌ Rejected: {results['rejected_count']}")
    print(f"  Approval rate: {results['approved_count']/results['original_count']:.1%}")

    # Show sample rejected items
    if results["rejected_count"] > 0:
        print(f"\n❌ SAMPLE REJECTED ITEMS (showing first 5):")
        for i, item in enumerate(results["rejected_samples"][:5], 1):
            print(f"  {i}. {item['sample_id']}: {item['reason'][:100]}...")
            print(f"     Input: {item['sample']['input'][:80]}...")

    # Save results
    print(f"\n💾 Creating output files...")
    
    # 1. Save rejected samples
    rejected_data = {
        "metadata": {
            "review_date": "2025-07-28",
            "reviewer": "o3-mini",
            "review_type": "comprehensive_quality_check",
            "total_reviewed": len(new_samples),
            "rejected_count": results['rejected_count']
        },
        "rejected_samples": results["rejected_samples"]
    }
    
    with open(
        "/Users/kyle/Documents/ws/post-hoc-reasoning/anachronisms_reviewed_rejected.json",
        "w",
    ) as f:
        json.dump(rejected_data, f, indent=2)

    # 2. Save approved new samples
    approved_new_data = {"examples": results["approved_samples"]}
    with open(
        "/Users/kyle/Documents/ws/post-hoc-reasoning/anachronisms_reviewed_new_samples.json",
        "w",
    ) as f:
        json.dump(approved_new_data, f, indent=2)

    # 3. Save full combined dataset (original + approved new)
    full_combined_samples = original_samples + results["approved_samples"]
    full_data = {"examples": full_combined_samples}
    with open(
        "/Users/kyle/Documents/ws/post-hoc-reasoning/anachronisms_reviewed_full.json",
        "w",
    ) as f:
        json.dump(full_data, f, indent=2)

    # Save comprehensive review report
    comprehensive_report = {
        "metadata": {
            "review_date": "2025-07-28",
            "reviewer": "o3-mini",
            "review_type": "comprehensive_full_review",
            "total_new_samples_reviewed": len(new_samples),
            "review_duration_minutes": (end_time - start_time) / 60,
        },
        "summary": {
            "original_count": results["original_count"],
            "approved_count": results["approved_count"],
            "rejected_count": results["rejected_count"],
            "approval_rate": results["approved_count"] / results["original_count"],
            "original_samples_count": len(original_samples),
            "final_combined_count": len(full_combined_samples)
        },
        "detailed_results": results,
    }

    with open(
        "/Users/kyle/Documents/ws/post-hoc-reasoning/anachronisms_full_review_report.json",
        "w",
    ) as f:
        json.dump(comprehensive_report, f, indent=2)

    print(f"\n📄 Files created:")
    print(f"  - anachronisms_reviewed_rejected.json ({results['rejected_count']} rejected samples)")
    print(f"  - anachronisms_reviewed_new_samples.json ({results['approved_count']} approved new samples)")
    print(f"  - anachronisms_reviewed_full.json ({len(full_combined_samples)} total samples)")
    print(f"  - anachronisms_full_review_report.json (comprehensive review report)")

    print(f"\n🎯 Final Dataset Statistics:")
    print(f"  Original samples: {len(original_samples)}")
    print(f"  New approved samples: {results['approved_count']}")
    print(f"  Total combined samples: {len(full_combined_samples)}")
    print(f"  Rejection rate: {results['rejected_count']/results['original_count']:.1%}")


if __name__ == "__main__":
    main()