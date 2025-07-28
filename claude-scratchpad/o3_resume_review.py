#!/usr/bin/env python3
"""
O3 Resume Review System

Resumes review from a checkpoint and processes remaining samples.
"""

import json
import os
import time
from typing import Dict, List

import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class O3ResumeReviewSystem:
    """Resume review system for continuing interrupted reviews."""

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
  "historical_accuracy_score": 8,
  "overall_quality_score": 8,
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


def process_remaining_samples(start_index: int = 450):
    """Process samples starting from the given index."""
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found")
        return

    print(f"🔍 O3 Resume Review System - Starting from sample {start_index + 1}")
    print("=" * 60)

    # Load new samples
    with open("/Users/kyle/Documents/ws/post-hoc-reasoning/anachronisms_new_samples.json", "r") as f:
        data = json.load(f)
        all_samples = data["examples"]

    remaining_samples = all_samples[start_index:]
    print(f"Processing {len(remaining_samples)} remaining samples (out of {len(all_samples)} total)")

    # Initialize review system
    reviewer = O3ResumeReviewSystem(api_key)

    # Process remaining samples
    reviews = []
    start_time = time.time()
    
    for i, sample in enumerate(remaining_samples):
        sample_id = f"sample_{start_index + i + 1:04d}"
        actual_index = start_index + i + 1
        print(f"Reviewing {sample_id} ({actual_index}/{len(all_samples)})...")

        review = reviewer.review_single_sample(sample, sample_id)
        reviews.append(review)

        # Progress updates
        if (i + 1) % 50 == 0:
            approved = len([r for r in reviews if r.get("recommendation") == "APPROVE"])
            rejected = len([r for r in reviews if r.get("recommendation") == "REJECT"])
            elapsed = (time.time() - start_time) / 60
            print(f"  Progress: {actual_index}/{len(all_samples)} | Approved: {approved} | Rejected: {rejected} | Time: {elapsed:.1f}min")

        # Save checkpoint every 100 samples
        if (i + 1) % 100 == 0:
            checkpoint_file = f"/Users/kyle/Documents/ws/post-hoc-reasoning/review_checkpoint_{actual_index}.json"
            with open(checkpoint_file, "w") as f:
                json.dump({"reviews": reviews, "last_index": actual_index}, f, indent=2)
            print(f"  Checkpoint saved: {checkpoint_file}")

        time.sleep(0.5)  # Shorter delay for faster processing

    # Save final results
    end_time = time.time()
    
    # Final statistics
    approved = len([r for r in reviews if r.get("recommendation") == "APPROVE"])
    rejected = len([r for r in reviews if r.get("recommendation") == "REJECT"])
    
    print(f"\n✅ Batch Review Complete!")
    print(f"📈 Batch Summary:")
    print(f"  Samples processed: {len(remaining_samples)}")
    print(f"  ✅ Approved: {approved}")
    print(f"  ❌ Rejected: {rejected}")
    print(f"  Approval rate: {approved/(approved+rejected):.1%}")
    print(f"  Time taken: {(end_time - start_time)/60:.1f} minutes")

    # Save final batch results
    batch_results = {
        "metadata": {
            "batch_start_index": start_index,
            "batch_end_index": start_index + len(remaining_samples) - 1,
            "samples_processed": len(remaining_samples),
            "processing_time_minutes": (end_time - start_time) / 60
        },
        "batch_reviews": reviews,
        "batch_summary": {
            "approved": approved,
            "rejected": rejected,
            "approval_rate": approved / (approved + rejected) if (approved + rejected) > 0 else 0
        }
    }
    
    batch_file = f"/Users/kyle/Documents/ws/post-hoc-reasoning/review_batch_{start_index}_to_{start_index + len(remaining_samples) - 1}.json"
    with open(batch_file, "w") as f:
        json.dump(batch_results, f, indent=2)
    
    print(f"\n💾 Batch results saved: {batch_file}")
    return reviews


if __name__ == "__main__":
    # Start from index 450 (where the previous run left off)
    process_remaining_samples(450)