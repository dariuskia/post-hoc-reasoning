#!/usr/bin/env python3
"""
O3 Manual Review System for Anachronisms

Uses o3 to manually review randomly selected anachronism sample pairs
for quality assurance and validation.
"""

import json
import os
import random
from typing import Dict, List, Tuple

import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class O3ReviewSystem:
    """Manual review system using o3 for quality validation."""

    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)

    def review_sample_pair(self, anachronistic: Dict, plausible: Dict) -> Dict:
        """Review a single sample pair using o3."""

        review_prompt = f"""You are an expert historian and dataset quality reviewer. Please carefully evaluate this anachronism detection sample pair.

ANACHRONISTIC SAMPLE:
{json.dumps(anachronistic, indent=2)}

PLAUSIBLE SAMPLE:  
{json.dumps(plausible, indent=2)}

Please evaluate the following criteria on a scale of 1-5 (where 5 is excellent):

1. **Historical Accuracy** (Plausible version): Is the plausible version historically accurate? Are the facts, timeframes, and context correct?

2. **Anachronism Validity** (Anachronistic version): Is the anachronistic version clearly temporally impossible? Would someone with historical knowledge detect this as anachronistic?

3. **Linguistic Quality**: Are both sentences well-formed, natural, and fluent? Do they sound realistic and professional?

4. **Task Appropriateness**: Does this pair effectively test anachronism detection skills? Is the contrast clear and meaningful?

5. **Difficulty Level**: Is this an appropriate difficulty level - not too obvious but not requiring obscure historical knowledge?

6. **Format Compliance**: Do both samples follow the exact JSON format requirements?

Please provide:
- A score (1-5) for each criterion
- Brief comments explaining your reasoning
- An overall recommendation: APPROVE, MINOR_ISSUES, or REJECT
- Specific suggestions for improvement if applicable

Format your response as JSON:
{{
  "scores": {{
    "historical_accuracy": X,
    "anachronism_validity": X, 
    "linguistic_quality": X,
    "task_appropriateness": X,
    "difficulty_level": X,
    "format_compliance": X
  }},
  "overall_score": X.X,
  "recommendation": "APPROVE/MINOR_ISSUES/REJECT",
  "comments": {{
    "historical_accuracy": "...",
    "anachronism_validity": "...", 
    "linguistic_quality": "...",
    "task_appropriateness": "...",
    "difficulty_level": "...",
    "format_compliance": "..."
  }},
  "suggestions": ["...", "..."],
  "summary": "Overall assessment..."
}}"""

        try:
            response = self.client.chat.completions.create(
                model="o3-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert historian and dataset quality reviewer. Provide detailed, objective assessments of anachronism detection samples.",
                    },
                    {"role": "user", "content": review_prompt},
                ],
                max_completion_tokens=1500,
            )

            # Parse the JSON response
            review_text = response.choices[0].message.content.strip()

            # Extract JSON from response
            try:
                # Find JSON block in response
                start_idx = review_text.find("{")
                end_idx = review_text.rfind("}") + 1
                json_str = review_text[start_idx:end_idx]
                review_data = json.loads(json_str)

                return review_data

            except json.JSONDecodeError:
                return {
                    "error": "Failed to parse review response",
                    "raw_response": review_text,
                }

        except Exception as e:
            return {"error": f"API call failed: {e}"}

    def select_review_samples(
        self, samples: List[Dict], n: int = 10
    ) -> List[Tuple[Dict, Dict]]:
        """Select n random sample pairs for review."""

        # Group samples into pairs (anachronistic + plausible)
        pairs = []
        anachronistic_samples = [s for s in samples if s["target_scores"]["Yes"] == 1]
        plausible_samples = [s for s in samples if s["target_scores"]["No"] == 1]

        # Create pairs by matching indices (assuming they were generated as pairs)
        min_pairs = min(len(anachronistic_samples), len(plausible_samples))
        for i in range(min_pairs):
            pairs.append((anachronistic_samples[i], plausible_samples[i]))

        # Randomly select n pairs
        selected_pairs = random.sample(pairs, min(n, len(pairs)))
        return selected_pairs

    def conduct_batch_review(self, sample_pairs: List[Tuple[Dict, Dict]]) -> List[Dict]:
        """Conduct review of multiple sample pairs."""

        reviews = []

        for i, (anachronistic, plausible) in enumerate(sample_pairs):
            print(f"Reviewing pair {i+1}/{len(sample_pairs)}...")

            review = self.review_sample_pair(anachronistic, plausible)
            review["pair_id"] = i + 1
            review["anachronistic_sample"] = anachronistic
            review["plausible_sample"] = plausible

            reviews.append(review)

            # Add delay between reviews
            if i < len(sample_pairs) - 1:
                print("  Waiting between reviews...")
                import time

                time.sleep(3)

        return reviews


def load_demo_samples() -> List[Dict]:
    """Load the demo samples for review."""
    with open(
        "/Users/kyle/Documents/ws/post-hoc-reasoning/anachronisms_demo_samples.json",
        "r",
    ) as f:
        data = json.load(f)
        return data["examples"]


def analyze_review_results(reviews: List[Dict]) -> Dict:
    """Analyze the review results and provide summary statistics."""

    if not reviews:
        return {"error": "No reviews to analyze"}

    # Filter out error reviews
    valid_reviews = [r for r in reviews if "scores" in r]

    if not valid_reviews:
        return {"error": "No valid reviews found"}

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
        scores = [r["scores"][criterion] for r in valid_reviews if "scores" in r]
        avg_scores[criterion] = sum(scores) / len(scores) if scores else 0

    # Count recommendations
    recommendations = [r["recommendation"] for r in valid_reviews]
    recommendation_counts = {
        "APPROVE": recommendations.count("APPROVE"),
        "MINOR_ISSUES": recommendations.count("MINOR_ISSUES"),
        "REJECT": recommendations.count("REJECT"),
    }

    # Calculate overall statistics
    overall_scores = [r["overall_score"] for r in valid_reviews if "overall_score" in r]
    avg_overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0

    return {
        "total_reviews": len(reviews),
        "valid_reviews": len(valid_reviews),
        "average_scores": avg_scores,
        "average_overall_score": avg_overall,
        "recommendation_counts": recommendation_counts,
        "approval_rate": (
            recommendation_counts["APPROVE"] / len(valid_reviews)
            if valid_reviews
            else 0
        ),
    }


def main():
    """Run o3 manual review of anachronism samples."""

    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found")
        return

    print("🔍 O3 Manual Review System for Anachronisms")
    print("=" * 50)

    # Load demo samples
    print("Loading demo samples...")
    try:
        samples = load_demo_samples()
        print(f"Loaded {len(samples)} demo samples")
    except FileNotFoundError:
        print(
            "❌ anachronisms_demo_samples.json not found. Please generate demo samples first."
        )
        return

    # Initialize review system
    reviewer = O3ReviewSystem(api_key)

    # Select samples for review (all 10 pairs from demo)
    print("Selecting sample pairs for review...")
    sample_pairs = reviewer.select_review_samples(samples, n=10)
    print(f"Selected {len(sample_pairs)} pairs for review")

    # Conduct reviews
    print("\n🧐 Conducting O3 reviews...")
    reviews = reviewer.conduct_batch_review(sample_pairs)

    # Analyze results
    print("\n📊 Analyzing review results...")
    analysis = analyze_review_results(reviews)

    # Save detailed review results
    review_data = {
        "metadata": {
            "review_date": "2025-07-28",
            "reviewer": "o3-mini",
            "samples_reviewed": len(sample_pairs),
            "total_samples": len(samples),
        },
        "reviews": reviews,
        "analysis": analysis,
    }

    print("Saving detailed review results...")
    with open(
        "/Users/kyle/Documents/ws/post-hoc-reasoning/anachronisms_o3_review_report.json",
        "w",
    ) as f:
        json.dump(review_data, f, indent=2)

    # Print summary
    print("\n✅ O3 Review Complete!")
    print(f"📄 Detailed report saved to: anachronisms_o3_review_report.json")

    if "error" not in analysis:
        print(f"\n📈 Summary Statistics:")
        print(f"Total Reviews: {analysis['total_reviews']}")
        print(f"Valid Reviews: {analysis['valid_reviews']}")
        print(f"Average Overall Score: {analysis['average_overall_score']:.2f}/5.0")
        print(f"Approval Rate: {analysis['approval_rate']:.1%}")

        print(f"\n📋 Recommendations:")
        for rec, count in analysis["recommendation_counts"].items():
            print(f"  {rec}: {count}")

        print(f"\n🎯 Average Scores by Criteria:")
        for criterion, score in analysis["average_scores"].items():
            print(f"  {criterion.replace('_', ' ').title()}: {score:.2f}/5.0")
    else:
        print(f"❌ Analysis error: {analysis['error']}")


if __name__ == "__main__":
    main()
