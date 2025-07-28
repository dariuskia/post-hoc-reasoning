#!/usr/bin/env python3
"""
Production O3 Review System for Anachronisms

Uses o3 model to review generated anachronism samples in batches of 10 pairs.
Reviews 10% of generated samples with stratified sampling across categories.
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

# Set seed for reproducible sampling
random.seed(42)


class ProductionO3Reviewer:
    """Production review system using o3 model for quality validation."""

    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.review_batch_size = (
            10  # PRODUCTION: 10 pairs (20 samples) per review batch
        )

    def stratified_sample_pairs(
        self, samples: List[Dict], total_pairs: int = 97
    ) -> List[Tuple[Dict, Dict]]:
        """Select representative sample pairs using stratified sampling across categories."""

        # Group samples into pairs
        anachronistic_samples = [s for s in samples if s["target_scores"]["Yes"] == 1]
        plausible_samples = [s for s in samples if s["target_scores"]["No"] == 1]

        # Create pairs (assuming they were generated as pairs)
        all_pairs = []
        min_pairs = min(len(anachronistic_samples), len(plausible_samples))
        for i in range(min_pairs):
            all_pairs.append((anachronistic_samples[i], plausible_samples[i]))

        # Categorize pairs by content
        categorized_pairs = {
            "technology_displacement": [],
            "temporal_displacement": [],
            "cultural_anachronisms": [],
            "scientific_anachronisms": [],
            "institutional_anachronisms": [],
        }

        for pair in all_pairs:
            anachronistic, plausible = pair
            category = self._categorize_sample(anachronistic["input"])
            categorized_pairs[category].append(pair)

        # Stratified sampling based on target distribution
        target_per_category = {
            "technology_displacement": int(total_pairs * 0.35),  # 34 pairs
            "temporal_displacement": int(total_pairs * 0.25),  # 24 pairs
            "cultural_anachronisms": int(total_pairs * 0.20),  # 19 pairs
            "scientific_anachronisms": int(total_pairs * 0.15),  # 15 pairs
            "institutional_anachronisms": int(total_pairs * 0.05),  # 5 pairs
        }

        selected_pairs = []
        for category, target_count in target_per_category.items():
            available_pairs = categorized_pairs[category]
            if available_pairs:
                sample_count = min(target_count, len(available_pairs))
                selected = random.sample(available_pairs, sample_count)
                selected_pairs.extend(selected)

        # If we need more pairs, randomly sample from remaining
        remaining_needed = total_pairs - len(selected_pairs)
        if remaining_needed > 0:
            all_remaining = [p for p in all_pairs if p not in selected_pairs]
            if all_remaining:
                additional = random.sample(
                    all_remaining, min(remaining_needed, len(all_remaining))
                )
                selected_pairs.extend(additional)

        return selected_pairs[:total_pairs]

    def _categorize_sample(self, text: str) -> str:
        """Categorize a sample based on its content."""
        text_lower = text.lower()

        if any(
            word in text_lower
            for word in [
                "computer",
                "laptop",
                "phone",
                "internet",
                "gps",
                "digital",
                "streaming",
                "app",
            ]
        ):
            return "technology_displacement"
        elif any(
            word in text_lower
            for word in [
                "fan",
                "music",
                "movie",
                "game",
                "sport",
                "food",
                "entertainment",
                "favorite",
            ]
        ):
            return "cultural_anachronisms"
        elif any(
            word in text_lower
            for word in [
                "dna",
                "genetic",
                "nuclear",
                "atomic",
                "vaccine",
                "antibiotic",
                "medical",
                "mri",
            ]
        ):
            return "scientific_anachronisms"
        elif any(
            word in text_lower
            for word in [
                "constitution",
                "democracy",
                "organization",
                "law",
                "vote",
                "election",
            ]
        ):
            return "institutional_anachronisms"
        else:
            return "temporal_displacement"

    def create_batch_review_prompt(
        self, sample_pairs: List[Tuple[Dict, Dict]], batch_id: str
    ) -> str:
        """Create comprehensive review prompt for a batch of sample pairs."""

        pairs_text = ""
        for i, (anachronistic, plausible) in enumerate(sample_pairs, 1):
            pairs_text += f"""
PAIR {i}:
Anachronistic Sample:
{json.dumps(anachronistic, indent=2)}

Plausible Sample:
{json.dumps(plausible, indent=2)}

---
"""

        prompt = f"""You are an expert historian and dataset quality reviewer conducting a comprehensive evaluation of anachronism detection sample pairs.

BATCH ID: {batch_id}
PAIRS TO REVIEW: {len(sample_pairs)}

{pairs_text}

For each pair, evaluate the following criteria on a scale of 1-5 (where 5 is excellent):

EVALUATION CRITERIA:
1. **Historical Accuracy** (Plausible version): Are all historical facts, dates, people, and contexts accurate?
2. **Anachronism Validity** (Anachronistic version): Is the temporal impossibility clear and unambiguous?
3. **Linguistic Quality**: Are both sentences well-formed, natural, and fluent?
4. **Task Appropriateness**: Does this pair effectively test anachronism detection skills?
5. **Difficulty Level**: Is this an appropriate difficulty level (not too obvious, not too obscure)?
6. **Format Compliance**: Do both samples follow the exact JSON format requirements?

ASSESSMENT LEVELS:
- **CRITICAL**: Major issues requiring deletion (historical inaccuracies, format errors)
- **MODERATE**: Minor issues requiring revision (awkward phrasing, unclear anachronisms)
- **MINOR**: Very small issues that don't affect functionality
- **NONE**: No issues found

For each pair, provide:
- Individual scores (1-5) for each criterion
- Overall severity assessment (CRITICAL/MODERATE/MINOR/NONE)
- Recommendation (APPROVE/REVISE/DELETE)
- Specific feedback explaining your assessment
- Suggested improvements if applicable

FORMAT YOUR RESPONSE AS JSON:
{{
  "batch_id": "{batch_id}",
  "total_pairs_reviewed": {len(sample_pairs)},
  "individual_reviews": [
    {{
      "pair_number": 1,
      "scores": {{
        "historical_accuracy": X,
        "anachronism_validity": X,
        "linguistic_quality": X,
        "task_appropriateness": X,
        "difficulty_level": X,
        "format_compliance": X
      }},
      "overall_score": X.X,
      "severity": "CRITICAL/MODERATE/MINOR/NONE",
      "recommendation": "APPROVE/REVISE/DELETE",
      "detailed_feedback": "...",
      "issues_identified": ["...", "..."],
      "suggested_improvements": ["...", "..."]
    }},
    ...
  ],
  "batch_summary": {{
    "approval_rate": X.X,
    "average_score": X.X,
    "common_issues": ["...", "..."],
    "recommendations": ["...", "..."]
  }}
}}

Begin comprehensive review of all {len(sample_pairs)} pairs:"""

        return prompt

    def review_batch_with_o3(
        self, sample_pairs: List[Tuple[Dict, Dict]], batch_id: str
    ) -> Dict:
        """Review a batch of sample pairs using o3 model."""

        prompt = self.create_batch_review_prompt(sample_pairs, batch_id)

        try:
            print(f"    📋 Reviewing batch {batch_id} with o3...")
            response = self.client.chat.completions.create(
                model="o3",  # PRODUCTION: Using full o3 model
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert historian and strict quality reviewer. Provide detailed, objective assessments of anachronism detection samples. Flag ANY historical inaccuracies or quality issues.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=4000,  # Increased for batch reviews
            )

            # Parse the JSON response
            review_text = response.choices[0].message.content.strip()

            try:
                # Find JSON block in response
                start_idx = review_text.find("{")
                end_idx = review_text.rfind("}") + 1
                json_str = review_text[start_idx:end_idx]
                review_data = json.loads(json_str)

                print(f"    ✅ Successfully reviewed {len(sample_pairs)} pairs")
                return review_data

            except json.JSONDecodeError:
                return {
                    "batch_id": batch_id,
                    "error": "Failed to parse review response",
                    "raw_response": (
                        review_text[:500] + "..."
                        if len(review_text) > 500
                        else review_text
                    ),
                }

        except Exception as e:
            return {"batch_id": batch_id, "error": f"API call failed: {e}"}

    def conduct_production_review(self, samples: List[Dict]) -> Dict:
        """Conduct comprehensive review of generated samples."""

        print(f"\\n🔍 PRODUCTION O3 REVIEW")
        print("=" * 50)

        # Calculate review sample size (10% of total)
        total_samples = len(samples)
        total_pairs = total_samples // 2
        review_pairs = max(1, int(total_pairs * 0.1))  # Review 10% (min 1 pair)

        print(f"Total samples: {total_samples}")
        print(f"Total pairs: {total_pairs}")
        print(f"Review pairs: {review_pairs} (10%)")

        # Stratified sampling
        print("\\n📊 Selecting representative sample pairs...")
        selected_pairs = self.stratified_sample_pairs(samples, review_pairs)

        print(f"Selected {len(selected_pairs)} pairs for review")

        # Calculate review batches
        review_batches = (
            len(selected_pairs) + self.review_batch_size - 1
        ) // self.review_batch_size
        print(
            f"Review batches needed: {review_batches} (batch size: {self.review_batch_size} pairs)"
        )

        # Conduct batch reviews
        all_reviews = []

        for batch_num in range(review_batches):
            batch_id = f"review_batch_{batch_num + 1:03d}"
            print(f"\\n  🔄 Processing {batch_id}...")

            # Get pairs for this batch
            start_idx = batch_num * self.review_batch_size
            end_idx = min(start_idx + self.review_batch_size, len(selected_pairs))
            batch_pairs = selected_pairs[start_idx:end_idx]

            # Review batch
            batch_review = self.review_batch_with_o3(batch_pairs, batch_id)
            all_reviews.append(batch_review)

            # Add delay between batches
            if batch_num < review_batches - 1:
                print(f"    ⏱️  Waiting 5 seconds before next review batch...")
                time.sleep(5)

        # Analyze overall results
        return self._analyze_review_results(all_reviews, selected_pairs)

    def _analyze_review_results(
        self, all_reviews: List[Dict], reviewed_pairs: List[Tuple[Dict, Dict]]
    ) -> Dict:
        """Analyze comprehensive review results."""

        valid_reviews = [r for r in all_reviews if "individual_reviews" in r]

        if not valid_reviews:
            return {
                "error": "No valid reviews found",
                "total_reviews": len(all_reviews),
                "failed_reviews": len(all_reviews),
            }

        # Aggregate statistics
        all_individual_reviews = []
        for batch_review in valid_reviews:
            all_individual_reviews.extend(batch_review.get("individual_reviews", []))

        # Calculate metrics
        total_reviewed = len(all_individual_reviews)
        approved = len(
            [r for r in all_individual_reviews if r.get("recommendation") == "APPROVE"]
        )
        flagged = len(
            [r for r in all_individual_reviews if r.get("recommendation") == "REVISE"]
        )
        deleted = len(
            [r for r in all_individual_reviews if r.get("recommendation") == "DELETE"]
        )

        # Calculate average scores
        criteria = [
            "historical_accuracy",
            "anachronism_validity",
            "linguistic_quality",
            "task_appropriateness",
            "difficulty_level",
            "format_compliance",
        ]

        avg_scores = {}
        for criterion in criteria:
            scores = [
                r["scores"][criterion]
                for r in all_individual_reviews
                if "scores" in r and criterion in r["scores"]
            ]
            avg_scores[criterion] = sum(scores) / len(scores) if scores else 0

        overall_scores = [
            r.get("overall_score", 0)
            for r in all_individual_reviews
            if "overall_score" in r
        ]
        avg_overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0

        # Collect common issues
        all_issues = []
        for review in all_individual_reviews:
            all_issues.extend(review.get("issues_identified", []))

        return {
            "metadata": {
                "review_date": "2025-07-28",
                "reviewer": "o3",
                "review_type": "production_quality_check",
                "total_samples_in_dataset": len(reviewed_pairs) * 2,
                "pairs_reviewed": len(reviewed_pairs),
                "review_coverage": len(reviewed_pairs)
                / (len(reviewed_pairs) * 10),  # Approximate
            },
            "summary_statistics": {
                "total_reviewed": total_reviewed,
                "approved": approved,
                "flagged_for_revision": flagged,
                "marked_for_deletion": deleted,
                "approval_rate": approved / total_reviewed if total_reviewed > 0 else 0,
                "quality_score": avg_overall,
            },
            "detailed_scores": {
                "average_overall_score": avg_overall,
                "criterion_scores": avg_scores,
            },
            "quality_assessment": {
                "high_quality_samples": approved,
                "needs_revision": flagged,
                "should_be_removed": deleted,
                "common_issues": list(set(all_issues)),
            },
            "detailed_reviews": all_reviews,
            "reviewed_sample_pairs": reviewed_pairs,
        }


def load_generated_samples() -> List[Dict]:
    """Load the generated anachronism samples for review."""
    try:
        with open(
            "/Users/kyle/Documents/ws/post-hoc-reasoning/anachronisms_new_samples.json",
            "r",
        ) as f:
            data = json.load(f)
            return data["examples"]
    except FileNotFoundError:
        print(
            "❌ anachronisms_new_samples.json not found. Please generate samples first."
        )
        return []


def main():
    """Run production o3 review of generated anachronism samples."""

    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found")
        return

    print("🔍 Production O3 Review System")
    print("Model: o3 | Batch Size: 10 pairs | Coverage: 10%")
    print("=" * 50)

    # Load generated samples
    print("Loading generated samples...")
    samples = load_generated_samples()
    if not samples:
        return

    print(f"Loaded {len(samples)} generated samples")

    # Initialize reviewer
    reviewer = ProductionO3Reviewer(api_key)

    # Conduct comprehensive review
    review_results = reviewer.conduct_production_review(samples)

    # Save results
    print("\\n💾 Saving review results...")
    with open(
        "/Users/kyle/Documents/ws/post-hoc-reasoning/anachronisms_production_review.json",
        "w",
    ) as f:
        json.dump(review_results, f, indent=2)

    # Print summary
    if "error" not in review_results:
        stats = review_results["summary_statistics"]
        scores = review_results["detailed_scores"]

        print("\\n✅ Production Review Complete!")
        print(f"📄 Detailed report: anachronisms_production_review.json")

        print(f"\\n📈 Summary Statistics:")
        print(f"  Total reviewed: {stats['total_reviewed']} pairs")
        print(f"  ✅ Approved: {stats['approved']} ({stats['approval_rate']:.1%})")
        print(f"  ⚠️  Flagged: {stats['flagged_for_revision']}")
        print(f"  ❌ Delete: {stats['marked_for_deletion']}")
        print(f"  🎯 Quality Score: {scores['average_overall_score']:.2f}/5.0")

        print(f"\\n📊 Detailed Scores:")
        for criterion, score in scores["criterion_scores"].items():
            print(f"  {criterion.replace('_', ' ').title()}: {score:.2f}/5.0")

        if review_results["quality_assessment"]["common_issues"]:
            print(f"\\n⚠️  Common Issues:")
            for issue in review_results["quality_assessment"]["common_issues"][:5]:
                print(f"  - {issue}")
    else:
        print(f"❌ Review failed: {review_results['error']}")


if __name__ == "__main__":
    main()
