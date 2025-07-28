#!/usr/bin/env python3
"""
Anachronisms Component Database Extractor

Analyzes the existing 226 anachronisms samples to extract:
1. Historical figures mentioned
2. Modern technologies/concepts that are anachronistic
3. Historical events and periods
4. Geographic locations
5. Pattern templates for generation

This creates the foundation databases for LLM-based sample generation.
"""

import json
import re
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple


def load_existing_dataset() -> List[Dict]:
    """Load the existing anachronisms dataset."""
    with open(
        "/Users/kyle/Documents/ws/post-hoc-reasoning/data/anachronisms/anachronisms.json",
        "r",
    ) as f:
        data = json.load(f)
        return data["examples"]


def extract_historical_figures(samples: List[Dict]) -> Dict[str, List[str]]:
    """Extract historical figures mentioned in samples, categorized by era."""

    # Known historical figures with their approximate eras
    historical_figures = {
        "ancient": set(),
        "medieval": set(),
        "renaissance": set(),
        "colonial_american": set(),
        "modern": set(),
    }

    # Define era patterns and known figures
    era_patterns = {
        "ancient": [
            "Socrates",
            "Plato",
            "Aristotle",
            "Alexander the Great",
            "Julius Caesar",
            "Cleopatra",
            "Augustus",
            "Hannibal",
            "Hammurabi",
            "Sun Tzu",
            "Confucius",
            "Homer",
            "Archimedes",
            "Ptolemy",
            "Cyrus",
            "Darius",
            "Xerxes",
            "Pericles",
            "Spartacus",
            "Nero",
            "Marcus Aurelius",
            "Caesar Octavian Augustus",
            "Pyrrhus",
            "Surena",
            "Attila",
        ],
        "medieval": [
            "Charlemagne",
            "William the Conqueror",
            "Richard the Lionheart",
            "Joan of Arc",
            "King John",
            "Ivan the Terrible",
            "Genghis Khan",
            "Marco Polo",
            "Eric the Red",
            "King Richard",
            "Saladin",
        ],
        "renaissance": [
            "Leonardo da Vinci",
            "Michelangelo",
            "Shakespeare",
            "Galileo",
            "Christopher Columbus",
            "Vasco de Gama",
            "Machiavelli",
            "Copernicus",
            "Tycho Brahe",
            "Palestrina",
            "John Fletcher",
        ],
        "colonial_american": [
            "George Washington",
            "Benjamin Franklin",
            "Thomas Jefferson",
            "Abraham Lincoln",
            "John Adams",
            "Alexander Hamilton",
            "Aaron Burr",
            "Lewis and Clark",
            "Pocahontas",
            "Squanto",
            "Ponce De Leon",
        ],
        "modern": [
            "Napoleon",
            "Theodore Roosevelt",
            "Woodrow Wilson",
            "Dwight Eisenhower",
            "George Washington Carver",
            "Nikola Tesla",
            "Thomas Edison",
            "Charles Darwin",
            "Marie Curie",
            "Albert Einstein",
            "Anne Frank",
        ],
    }

    # Extract figures from sample text
    all_text = " ".join([sample["input"] for sample in samples])

    for era, figures in era_patterns.items():
        for figure in figures:
            if figure in all_text:
                historical_figures[era].add(figure)

    # Convert sets to sorted lists
    return {era: sorted(list(figures)) for era, figures in historical_figures.items()}


def extract_modern_technologies(samples: List[Dict]) -> Dict[str, List[str]]:
    """Extract modern technologies/concepts that appear in anachronistic contexts."""

    technology_categories = {
        "computing": set(),
        "communication": set(),
        "transportation": set(),
        "weapons_tools": set(),
        "materials": set(),
        "entertainment": set(),
        "institutions": set(),
        "science": set(),
    }

    # Technology patterns to look for
    tech_patterns = {
        "computing": [
            "laptop",
            "computer",
            "smartphone",
            "cell phone",
            "phone",
            "calculator",
            "GPS",
            "internet",
            "email",
            "software",
            "programming",
            "digital",
            "AI",
            "artificial intelligence",
            "machine learning",
            "blockchain",
            "video",
            "streaming",
            "5G",
            "WiFi",
            "Bluetooth",
            "VCR",
            "DVD",
            "e-book",
            "ebook",
            "online",
            "website",
            "app",
            "download",
        ],
        "communication": [
            "telephone",
            "telegram",
            "radio",
            "television",
            "satellite",
            "broadcast",
            "livestream",
            "podcast",
            "social media",
            "Twitter",
            "Facebook",
            "Instagram",
            "YouTube",
            "TikTok",
            "Snapchat",
        ],
        "transportation": [
            "car",
            "automobile",
            "truck",
            "bus",
            "airplane",
            "helicopter",
            "rocket",
            "spacecraft",
            "submarine",
            "motorcycle",
            "bicycle",
            "train",
            "railway",
            "Ferrari",
            "Tesla",
            "Model T",
            "Cybertruck",
        ],
        "weapons_tools": [
            "nuclear",
            "atomic",
            "radar",
            "laser",
            "missile",
            "drone",
            "jackhammer",
            "chainsaw",
            "power drill",
            "electric",
            "battery",
        ],
        "materials": [
            "plastic",
            "synthetic",
            "titanium",
            "aluminum",
            "silicon",
            "carbon fiber",
            "kevlar",
            "teflon",
            "velcro",
            "polyester",
        ],
        "entertainment": [
            "movie",
            "film",
            "cinema",
            "television show",
            "video game",
            "rock music",
            "hip hop",
            "jazz",
            "blues",
            "country music",
            "pop music",
            "DJ",
            "concert",
            "album",
            "CD",
            "MP3",
        ],
        "institutions": [
            "corporation",
            "stock market",
            "bank",
            "insurance",
            "franchise",
            "university",
            "democracy",
            "republic",
            "constitution",
            "bill of rights",
        ],
        "science": [
            "DNA",
            "genetics",
            "evolution",
            "quantum",
            "relativity",
            "psychology",
            "psychiatry",
            "medicine",
            "vaccine",
            "antibiotic",
        ],
    }

    # Extract from sample text
    all_text = " ".join([sample["input"] for sample in samples]).lower()

    for category, technologies in tech_patterns.items():
        for tech in technologies:
            if tech.lower() in all_text:
                technology_categories[category].add(tech)

    return {
        category: sorted(list(techs))
        for category, techs in technology_categories.items()
    }


def extract_historical_events_periods(samples: List[Dict]) -> Dict[str, List[str]]:
    """Extract historical events, periods, and locations mentioned."""

    historical_elements = {
        "events": set(),
        "periods": set(),
        "locations": set(),
        "institutions": set(),
    }

    # Patterns to extract
    element_patterns = {
        "events": [
            "World War I",
            "World War II",
            "Civil War",
            "Revolutionary War",
            "Crusade",
            "Black Death",
            "Great Depression",
            "Cold War",
            "Salem Witch Trials",
            "Boston Tea Party",
            "Pearl Harbor",
            "Boxer Rebellion",
            "Bay of Pigs",
            "Gulf War",
            "Vietnam War",
        ],
        "periods": [
            "Renaissance",
            "Enlightenment",
            "Middle Ages",
            "Dark Ages",
            "Industrial Revolution",
            "Stone Age",
            "Bronze Age",
            "Iron Age",
            "Prohibition",
            "Great Depression",
            "Roaring Twenties",
            "Victorian Era",
            "Colonial Period",
            "Antebellum",
            "Reconstruction",
        ],
        "locations": [
            "Rome",
            "Athens",
            "Egypt",
            "Babylon",
            "Constantinople",
            "Jerusalem",
            "Mecca",
            "Beijing",
            "London",
            "Paris",
            "Madrid",
            "Florence",
            "Venice",
            "Greenland",
            "Iceland",
            "America",
            "Coliseum",
            "Pyramids",
            "Great Wall",
            "Notre Dame",
            "Hagia Sophia",
        ],
        "institutions": [
            "Roman Empire",
            "Byzantine Empire",
            "Ottoman Empire",
            "British Empire",
            "Holy Roman Empire",
            "Catholic Church",
            "Protestant Church",
            "Congress",
            "Parliament",
            "Senate",
            "Supreme Court",
        ],
    }

    # Extract from sample text
    all_text = " ".join([sample["input"] for sample in samples])

    for category, elements in element_patterns.items():
        for element in elements:
            if element in all_text:
                historical_elements[category].add(element)

    return {
        category: sorted(list(elements))
        for category, elements in historical_elements.items()
    }


def analyze_anachronism_patterns(samples: List[Dict]) -> Dict[str, List[str]]:
    """Analyze patterns in how anachronisms are constructed."""

    patterns = {
        "technology_displacement": [],
        "temporal_figure_displacement": [],
        "cultural_anachronisms": [],
        "scientific_anachronisms": [],
        "event_timeline_errors": [],
    }

    # Group samples into pairs (anachronistic vs plausible)
    sample_pairs = []
    for i in range(0, len(samples), 2):
        if i + 1 < len(samples):
            anachronistic = (
                samples[i]
                if samples[i]["target_scores"]["Yes"] == 1
                else samples[i + 1]
            )
            plausible = (
                samples[i + 1]
                if samples[i]["target_scores"]["Yes"] == 1
                else samples[i]
            )
            sample_pairs.append((anachronistic, plausible))

    # Analyze each pair to identify pattern types
    technology_keywords = [
        "used",
        "laptop",
        "phone",
        "computer",
        "GPS",
        "internet",
        "digital",
    ]
    figure_keywords = ["collaborated", "met", "wrote to", "worked with"]
    cultural_keywords = ["fan of", "listened to", "watched", "played"]

    for anachronistic, plausible in sample_pairs[
        :50
    ]:  # Analyze first 50 pairs as examples
        ana_text = anachronistic["input"].lower()

        if any(tech in ana_text for tech in technology_keywords):
            patterns["technology_displacement"].append(
                {
                    "anachronistic": anachronistic["input"],
                    "plausible": plausible["input"],
                    "pattern_type": "technology_substitution",
                }
            )
        elif any(fig in ana_text for fig in figure_keywords):
            patterns["temporal_figure_displacement"].append(
                {
                    "anachronistic": anachronistic["input"],
                    "plausible": plausible["input"],
                    "pattern_type": "impossible_collaboration",
                }
            )
        elif any(cult in ana_text for cult in cultural_keywords):
            patterns["cultural_anachronisms"].append(
                {
                    "anachronistic": anachronistic["input"],
                    "plausible": plausible["input"],
                    "pattern_type": "modern_culture_in_past",
                }
            )

    return patterns


def extract_sentence_templates(samples: List[Dict]) -> List[str]:
    """Extract sentence structure templates for generation."""

    templates = []

    # Analyze sentence structures
    sample_pairs = []
    for i in range(0, len(samples), 2):
        if i + 1 < len(samples):
            sample_pairs.append((samples[i]["input"], samples[i + 1]["input"]))

    # Extract common patterns
    common_patterns = [
        "{figure} used {technology/tool} to {action}",
        "{figure} {verb} {object/person} {context}",
        "{figure} was a fan of {modern_thing}",
        "{figure} collaborated with {other_figure} on {project}",
        "The {historical_object} was made of {material}",
        "{figure} {action} during {time_period}",
        "{event} occurred during {wrong_time_period}",
        "{figure} invented {anachronistic_invention}",
        "{figure} participated in {wrong_event}",
        "{modern_institution} was {action} during {historical_period}",
    ]

    return common_patterns


def main():
    """Extract and save component databases from existing anachronisms dataset."""
    print("Loading existing anachronisms dataset...")
    samples = load_existing_dataset()
    print(f"Loaded {len(samples)} samples")

    print("\nExtracting historical figures by era...")
    historical_figures = extract_historical_figures(samples)
    for era, figures in historical_figures.items():
        print(f"{era}: {len(figures)} figures - {figures[:5]}...")

    print("\nExtracting modern technologies by category...")
    technologies = extract_modern_technologies(samples)
    for category, techs in technologies.items():
        print(f"{category}: {len(techs)} technologies - {techs[:3]}...")

    print("\nExtracting historical elements...")
    historical_elements = extract_historical_events_periods(samples)
    for category, elements in historical_elements.items():
        print(f"{category}: {len(elements)} elements - {elements[:3]}...")

    print("\nAnalyzing anachronism patterns...")
    patterns = analyze_anachronism_patterns(samples)
    for pattern_type, examples in patterns.items():
        print(f"{pattern_type}: {len(examples)} examples")

    print("\nExtracting sentence templates...")
    templates = extract_sentence_templates(samples)
    print(f"Extracted {len(templates)} common templates")

    # Save extracted components
    components_data = {
        "historical_figures": historical_figures,
        "modern_technologies": technologies,
        "historical_elements": historical_elements,
        "anachronism_patterns": patterns,
        "sentence_templates": templates,
        "extraction_metadata": {
            "source_samples": len(samples),
            "total_figures": sum(len(figs) for figs in historical_figures.values()),
            "total_technologies": sum(len(techs) for techs in technologies.values()),
            "extraction_date": "2025-07-28",
        },
    }

    print("\nSaving component databases...")
    with open(
        "/Users/kyle/Documents/ws/post-hoc-reasoning/claude-scratchpad/anachronisms_components.json",
        "w",
    ) as f:
        json.dump(components_data, f, indent=2)

    print("Component extraction complete!")
    print(f"Files created:")
    print(f"- anachronisms_components.json")

    # Print summary statistics
    print(f"\n=== EXTRACTION SUMMARY ===")
    print(
        f"Historical Figures: {sum(len(figs) for figs in historical_figures.values())}"
    )
    print(f"Modern Technologies: {sum(len(techs) for techs in technologies.values())}")
    print(
        f"Historical Elements: {sum(len(elems) for elems in historical_elements.values())}"
    )
    print(f"Pattern Examples: {sum(len(patterns[pt]) for pt in patterns)}")
    print(f"Templates: {len(templates)}")


if __name__ == "__main__":
    main()
