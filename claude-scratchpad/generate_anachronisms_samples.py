#!/usr/bin/env python3
"""
Anachronisms Dataset Generator

Generates 974 new anachronisms samples using LLM-assisted generation
with template-based patterns and quality validation.

Target: 974 new samples (expand from 226 to 1200 total)
"""

import json
import os
import random
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set seed for reproducibility
random.seed(42)


class AnachronismTemplates:
    """Generation templates for different types of anachronisms."""

    def __init__(self):
        # Load component databases
        with open(
            "/Users/kyle/Documents/ws/post-hoc-reasoning/claude-scratchpad/anachronisms_components.json",
            "r",
        ) as f:
            self.components = json.load(f)

        # Enhanced historical figures database
        self.historical_figures = {
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
                "Herodotus",
                "Thucydides",
                "Pericles",
                "Spartacus",
                "Nero",
                "Marcus Aurelius",
                "Cyrus the Great",
                "Darius",
                "Xerxes",
                "Leonidas",
                "Solon",
            ],
            "medieval": [
                "Charlemagne",
                "William the Conqueror",
                "Richard the Lionheart",
                "Joan of Arc",
                "Saladin",
                "Frederick Barbarossa",
                "King Arthur",
                "Genghis Khan",
                "Kublai Khan",
                "Marco Polo",
                "Thomas Aquinas",
                "Dante Alighieri",
                "Geoffrey Chaucer",
                "El Cid",
                "Roland",
                "Eric the Red",
                "Leif Erikson",
                "Otto I",
                "Henry II",
                "Eleanor of Aquitaine",
            ],
            "renaissance": [
                "Leonardo da Vinci",
                "Michelangelo",
                "Raphael",
                "Shakespeare",
                "Galileo Galilei",
                "Christopher Columbus",
                "Vasco da Gama",
                "Machiavelli",
                "Copernicus",
                "Johannes Kepler",
                "Tycho Brahe",
                "Erasmus",
                "Martin Luther",
                "John Calvin",
                "Gutenberg",
                "Donatello",
                "Botticelli",
                "Titian",
                "Dürer",
                "Montaigne",
            ],
            "colonial_american": [
                "George Washington",
                "Benjamin Franklin",
                "Thomas Jefferson",
                "John Adams",
                "Alexander Hamilton",
                "Aaron Burr",
                "Paul Revere",
                "Patrick Henry",
                "Samuel Adams",
                "John Hancock",
                "Benedict Arnold",
                "Marquis de Lafayette",
                "Nathan Hale",
                "Ethan Allen",
                "Daniel Boone",
                "Pocahontas",
                "Squanto",
                "Captain John Smith",
                "William Penn",
                "Roger Williams",
            ],
            "enlightenment": [
                "Voltaire",
                "Rousseau",
                "John Locke",
                "David Hume",
                "Immanuel Kant",
                "Adam Smith",
                "Montesquieu",
                "Diderot",
                "Newton",
                "Leibniz",
                "Descartes",
                "Pascal",
                "Spinoza",
                "Berkeley",
                "Edmund Burke",
            ],
            "industrial_modern": [
                "Napoleon Bonaparte",
                "Abraham Lincoln",
                "Charles Darwin",
                "Karl Marx",
                "Sigmund Freud",
                "Albert Einstein",
                "Marie Curie",
                "Nikola Tesla",
                "Thomas Edison",
                "Alexander Graham Bell",
                "Henry Ford",
                "Wright Brothers",
                "Theodore Roosevelt",
                "Winston Churchill",
                "Franklin D. Roosevelt",
                "Gandhi",
            ],
        }

        # Enhanced modern technologies database
        self.modern_technologies = {
            "computing": [
                "laptop",
                "computer",
                "smartphone",
                "tablet",
                "smartwatch",
                "iPhone",
                "internet",
                "email",
                "website",
                "social media",
                "GPS",
                "WiFi",
                "Bluetooth",
                "artificial intelligence",
                "machine learning",
                "blockchain",
                "cryptocurrency",
                "virtual reality",
                "augmented reality",
                "3D printing",
                "cloud computing",
                "software",
                "app",
                "algorithm",
                "database",
                "programming",
                "coding",
            ],
            "communication": [
                "telephone",
                "cell phone",
                "television",
                "radio",
                "satellite",
                "telegram",
                "video call",
                "text message",
                "podcast",
                "livestream",
                "YouTube",
                "Twitter",
                "Facebook",
                "Instagram",
                "TikTok",
                "Zoom",
                "Skype",
                "email",
                "instant messaging",
                "video chat",
                "social media",
                "blog",
            ],
            "transportation": [
                "automobile",
                "car",
                "truck",
                "motorcycle",
                "airplane",
                "helicopter",
                "rocket",
                "spacecraft",
                "submarine",
                "train",
                "metro",
                "bus",
                "bicycle",
                "electric car",
                "hybrid car",
                "Tesla",
                "Ferrari",
                "Uber",
                "ride-sharing",
                "GPS navigation",
                "self-driving car",
                "drone delivery",
            ],
            "entertainment": [
                "movie",
                "cinema",
                "television show",
                "video game",
                "streaming",
                "Netflix",
                "Spotify",
                "iTunes",
                "DVD",
                "Blu-ray",
                "VHS",
                "cassette",
                "CD",
                "MP3",
                "digital music",
                "online gaming",
                "virtual reality game",
                "social media",
                "YouTube",
                "podcast",
                "audiobook",
                "e-book",
            ],
            "modern_concepts": [
                "democracy",
                "human rights",
                "feminism",
                "environmentalism",
                "psychology",
                "sociology",
                "economics",
                "stock market",
                "corporation",
                "startup",
                "venture capital",
                "crowdfunding",
                "cryptocurrency",
                "globalization",
                "climate change",
                "sustainability",
                "renewable energy",
                "nuclear power",
            ],
            "medical_science": [
                "vaccine",
                "antibiotic",
                "anesthesia",
                "X-ray",
                "MRI",
                "CT scan",
                "DNA testing",
                "genetic engineering",
                "organ transplant",
                "chemotherapy",
                "laser surgery",
                "robotic surgery",
                "telemedicine",
                "mental health therapy",
                "psychology",
                "psychiatry",
                "neuroscience",
                "stem cell research",
            ],
        }

        # Historical events and periods
        self.historical_events = [
            "World War I",
            "World War II",
            "American Civil War",
            "Revolutionary War",
            "French Revolution",
            "Russian Revolution",
            "Industrial Revolution",
            "Protestant Reformation",
            "Renaissance",
            "Enlightenment",
            "Crusades",
            "Black Death",
            "Fall of Rome",
            "Rise of Christianity",
            "Viking Age",
            "Age of Exploration",
            "Colonial Period",
            "Great Depression",
            "Cold War",
            "Salem Witch Trials",
            "Boston Tea Party",
            "Declaration of Independence",
        ]

        # Generation templates
        self.templates = {
            "technology_displacement": [
                "{figure} used {modern_tech} to {action}",
                "{figure} invented the {modern_tech}",
                "{figure} recorded {content} using {modern_tech}",
                "{figure} communicated with {other} via {modern_tech}",
                "{figure} traveled to {location} using {modern_tech}",
                "{figure} {action} using his {modern_tech}",
                "The {historical_object} was equipped with {modern_tech}",
                "{figure} developed the first {modern_tech}",
                "{figure} used {modern_tech} to solve {problem}",
                "{figure} broadcasted {content} on {modern_tech}",
            ],
            "temporal_displacement": [
                "{early_figure} collaborated with {later_figure} on {project}",
                "{early_figure} met {later_figure} at {location}",
                "{early_figure} wrote to {later_figure} about {topic}",
                "{early_figure} and {later_figure} were {relationship}",
                "{early_figure} influenced {later_figure}'s work on {subject}",
                "{early_figure} debated {later_figure} about {topic}",
                "{early_figure} studied under {later_figure}",
                "{early_figure} competed against {later_figure} in {activity}",
                "{early_figure} commissioned {later_figure} to {action}",
                "{early_figure} attended {later_figure}'s {event}",
            ],
            "cultural_anachronisms": [
                "{figure} was a fan of {modern_cultural_element}",
                "{figure} listened to {modern_music} while {action}",
                "{figure} played {modern_game} in his spare time",
                "{figure} watched {modern_entertainment} for inspiration",
                "{figure} enjoyed {modern_food} as his favorite meal",
                "{figure} collected {modern_items} as a hobby",
                "{figure} participated in the {modern_activity} movement",
                "{figure} was influenced by {modern_ideology}",
                "{figure} celebrated {modern_holiday} with {activity}",
                "{figure} followed the {modern_trend} fashion",
            ],
            "scientific_anachronisms": [
                "{figure} discovered {modern_scientific_concept}",
                "{figure} used {modern_scientific_method} to study {subject}",
                "{figure} applied {modern_theory} to explain {phenomenon}",
                "{figure} conducted {modern_experiment} to prove {hypothesis}",
                "{figure} used {modern_material} in his {creation}",
                "{figure} treated patients with {modern_medicine}",
                "{figure} observed {phenomenon} using {modern_instrument}",
                "{figure} published research on {modern_scientific_field}",
                "{figure} developed a theory of {modern_scientific_concept}",
                "{figure} synthesized {modern_compound} in his laboratory",
            ],
            "institutional_anachronisms": [
                "{figure} was elected to {modern_institution}",
                "{figure} founded the {modern_organization}",
                "{figure} graduated from {modern_university}",
                "{figure} worked for {modern_company}",
                "{figure} was certified by {modern_authority}",
                "{figure} applied for {modern_legal_concept}",
                "{figure} sued {other} under {modern_law}",
                "{figure} voted in the {modern_election}",
                "{figure} received funding from {modern_funding_source}",
                "{figure} was regulated by {modern_regulatory_body}",
            ],
        }


class AnachronismGenerator:
    """Main generator for anachronisms using LLM assistance."""

    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.templates = AnachronismTemplates()
        self.generated_samples = []
        self.validation_cache = {}

    def generate_sample_pair(
        self, category: str, complexity: str = "moderate"
    ) -> Optional[Dict]:
        """Generate a single anachronistic/plausible sample pair."""

        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                # Generate context and base scenario
                context = self._generate_historical_context(category)

                if not context:
                    continue

                # Generate anachronistic version using LLM
                anachronistic = self._generate_anachronistic_version(
                    context, category, complexity
                )

                if not anachronistic:
                    continue

                # Generate plausible version using LLM
                plausible = self._generate_plausible_version(
                    anachronistic, context, category
                )

                if not plausible:
                    continue

                # Validate the pair
                if self._validate_sample_pair(anachronistic, plausible):
                    return {
                        "anachronistic": {
                            "input": anachronistic,
                            "target_scores": {"Yes": 1, "No": 0},
                        },
                        "plausible": {
                            "input": plausible,
                            "target_scores": {"Yes": 0, "No": 1},
                        },
                        "metadata": {
                            "category": category,
                            "complexity": complexity,
                            "context": context,
                        },
                    }

            except Exception as e:
                print(f"Error generating sample (attempt {attempt + 1}): {e}")
                time.sleep(1)  # Brief pause before retry

        return None

    def _generate_historical_context(self, category: str) -> Optional[Dict]:
        """Generate historical context for the anachronism."""

        # Select historical period and figure
        period = random.choice(list(self.templates.historical_figures.keys()))
        figure = random.choice(self.templates.historical_figures[period])

        # Select appropriate modern element based on category
        if category == "technology_displacement":
            tech_category = random.choice(
                list(self.templates.modern_technologies.keys())
            )
            modern_element = random.choice(
                self.templates.modern_technologies[tech_category]
            )
            element_type = "technology"
        elif category == "temporal_displacement":
            # Select another figure from a different, later period
            later_periods = [
                p
                for p in self.templates.historical_figures.keys()
                if p != period and self._is_later_period(period, p)
            ]
            if not later_periods:
                return None
            later_period = random.choice(later_periods)
            modern_element = random.choice(
                self.templates.historical_figures[later_period]
            )
            element_type = "figure"
        elif category == "cultural_anachronisms":
            modern_element = random.choice(
                self.templates.modern_technologies["entertainment"]
            )
            element_type = "cultural"
        elif category == "scientific_anachronisms":
            modern_element = random.choice(
                self.templates.modern_technologies["medical_science"]
            )
            element_type = "scientific"
        else:  # institutional_anachronisms
            modern_element = random.choice(
                self.templates.modern_technologies["modern_concepts"]
            )
            element_type = "institutional"

        return {
            "figure": figure,
            "period": period,
            "modern_element": modern_element,
            "element_type": element_type,
            "category": category,
        }

    def _is_later_period(self, earlier: str, later: str) -> bool:
        """Check if one period comes after another chronologically."""
        period_order = [
            "ancient",
            "medieval",
            "renaissance",
            "colonial_american",
            "enlightenment",
            "industrial_modern",
        ]
        try:
            return period_order.index(later) > period_order.index(earlier)
        except ValueError:
            return False

    def generate_batch_samples(self, category: str, batch_size: int = 10) -> List[Dict]:
        """Generate a batch of sample pairs using o3."""

        # Load existing samples to show as examples
        existing_samples = load_existing_dataset()
        example_pairs = self._get_example_pairs(existing_samples, category, 3)

        prompt = self._create_batch_generation_prompt(
            category, batch_size, example_pairs
        )

        try:
            print(f"    Sending request to o3-mini for {batch_size} pairs...")
            response = self.client.chat.completions.create(
                model="o3-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert historian and dataset creator. You must generate anachronism detection samples that exactly match the format and style of the provided examples.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=4000,
            )

            print(f"    Received response, parsing...")
            parsed_samples = self._parse_batch_response(
                response.choices[0].message.content, category
            )
            print(f"    Parsed {len(parsed_samples)} samples from response")
            return parsed_samples

        except Exception as e:
            print(f"Error generating batch: {e}")
            return []

    def _get_example_pairs(
        self, existing_samples: List[Dict], category: str, count: int
    ) -> List[Tuple[Dict, Dict]]:
        """Get example pairs from existing dataset that match the category."""

        pairs = []
        for i in range(0, len(existing_samples) - 1, 2):
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
            ]
            return any(word in text_lower for word in tech_words)
        elif category == "temporal_displacement":
            # Look for samples with people from different eras
            return any(
                name in text
                for name in ["Einstein", "Newton", "Darwin", "Franklin", "Washington"]
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
            ]
            return any(word in text_lower for word in inst_words)

    def _create_batch_generation_prompt(
        self, category: str, batch_size: int, example_pairs: List[Tuple[Dict, Dict]]
    ) -> str:
        """Create a detailed prompt for batch generation using o3."""

        # Format example pairs to show exact structure
        examples_text = "EXACT EXAMPLES FROM THE DATASET:\n\n"
        for i, (anachronistic, plausible) in enumerate(example_pairs, 1):
            examples_text += f"Example {i}:\n"
            examples_text += (
                f"Anachronistic version:\n{json.dumps(anachronistic, indent=2)}\n\n"
            )
            examples_text += (
                f"Plausible version:\n{json.dumps(plausible, indent=2)}\n\n"
            )

        category_descriptions = {
            "technology_displacement": "Historical figures using modern technology that didn't exist in their time",
            "temporal_displacement": "People from different historical eras interacting impossibly",
            "cultural_anachronisms": "Historical figures engaging with modern culture, entertainment, or lifestyle",
            "scientific_anachronisms": "Historical figures using modern scientific knowledge or medical advances",
            "institutional_anachronisms": "Historical figures interacting with modern institutions or legal concepts",
        }

        prompt = f"""TASK: Generate {batch_size} pairs of anachronism detection samples for the category: {category}

CATEGORY DESCRIPTION: {category_descriptions.get(category, category)}

{examples_text}

CRITICAL REQUIREMENTS:
1. EXACT FORMAT: Each sample must have EXACTLY the same JSON structure as the examples above
2. EXACT FIELD NAMES: Use "input" and "target_scores" with "Yes" and "No" keys only
3. EXACT SCORING: Anachronistic samples get {{"Yes": 1, "No": 0}}, plausible samples get {{"Yes": 0, "No": 1}}
4. PAIRED STRUCTURE: Generate pairs where one version is anachronistic and one is historically plausible
5. SIMILAR LENGTH: Keep sentences roughly the same length as the examples (20-150 words)
6. NATURAL LANGUAGE: Write in natural, fluent English that sounds realistic
7. CLEAR ANACHRONISMS: Make the temporal impossibility obvious to someone with historical knowledge
8. HISTORICAL ACCURACY: Ensure the plausible versions are factually correct

SPECIFIC INSTRUCTIONS FOR {category.upper()}:
- Focus on {category_descriptions.get(category, category)}
- Use different historical figures and time periods for variety
- Make anachronisms clear but not cartoonishly obvious
- Ensure both versions of each pair cover the same basic scenario

OUTPUT FORMAT:
Generate exactly {batch_size} pairs. For each pair, output the anachronistic version first, then the plausible version. Use the EXACT JSON format shown in the examples.

Sample 1 (Anachronistic):
{{
  "input": "[Your anachronistic sentence here]",
  "target_scores": {{
    "Yes": 1,
    "No": 0
  }}
}}

Sample 1 (Plausible):
{{
  "input": "[Your plausible sentence here]", 
  "target_scores": {{
    "Yes": 0,
    "No": 1
  }}
}}

Sample 2 (Anachronistic):
[Continue pattern...]

Generate all {batch_size} pairs now:"""

        return prompt

    def _parse_batch_response(self, response_text: str, category: str) -> List[Dict]:
        """Parse the batch response from o3 into individual samples."""

        samples = []

        # Split response into JSON blocks
        json_blocks = []
        current_block = ""
        in_json = False
        brace_count = 0

        for line in response_text.split("\n"):
            line = line.strip()
            if line.startswith("{"):
                in_json = True
                brace_count = 0
                current_block = line
                brace_count += line.count("{") - line.count("}")
            elif in_json:
                current_block += "\n" + line
                brace_count += line.count("{") - line.count("}")

                if brace_count <= 0:
                    try:
                        parsed = json.loads(current_block)
                        if "input" in parsed and "target_scores" in parsed:
                            json_blocks.append(parsed)
                    except json.JSONDecodeError:
                        pass
                    in_json = False
                    current_block = ""

        return json_blocks

    def _create_anachronistic_prompt(
        self, context: Dict, category: str, complexity: str
    ) -> str:
        """Create prompt for generating anachronistic statement."""

        figure = context["figure"]
        period = context["period"]
        modern_element = context["modern_element"]
        element_type = context["element_type"]

        base_prompt = f"""Create an anachronistic statement about the historical figure {figure} from the {period} period.

The statement should:
1. Include the anachronistic element: "{modern_element}" ({element_type})
2. Be {complexity} in complexity (not too obvious, requires some historical knowledge)
3. Sound plausible but be temporally impossible
4. Be a single, clear sentence
5. Maintain historical context while including the anachronistic element

Context: {figure} lived in the {period} period, long before {modern_element} existed.

Example format: "{figure} used {modern_element} to [historical action appropriate to the figure]"

Anachronistic statement:"""

        if category == "temporal_displacement":
            base_prompt = f"""Create an anachronistic statement showing {figure} ({period} period) interacting with {modern_element} (a person from a much later time period).

The statement should show them collaborating, meeting, or communicating in a way that would be chronologically impossible.

Make it {complexity} complexity - not immediately obvious to someone without historical knowledge.

Anachronistic statement:"""

        return base_prompt

    def _validate_sample_pair(self, anachronistic: str, plausible: str) -> bool:
        """Validate that the sample pair meets quality criteria."""

        # Basic checks
        if not anachronistic or not plausible:
            return False

        if len(anachronistic) < 20 or len(plausible) < 20:
            return False

        if len(anachronistic) > 200 or len(plausible) > 200:
            return False

        # Check that they're different enough
        if anachronistic.lower() == plausible.lower():
            return False

        # Use GPT-4 for quality validation (with caching)
        cache_key = (anachronistic, plausible)
        if cache_key in self.validation_cache:
            return self.validation_cache[cache_key]

        validation_prompt = f"""Evaluate this anachronism detection sample pair:

Anachronistic: "{anachronistic}"
Plausible: "{plausible}"

Rate on a scale of 1-5:
1. Is the anachronistic version clearly temporally impossible? (1=no, 5=yes)
2. Is the plausible version historically accurate? (1=no, 5=yes)
3. Are both sentences well-formed and natural? (1=no, 5=yes)
4. Is this an appropriate difficulty level? (1=too easy/hard, 5=just right)

Respond with only four numbers separated by commas (e.g., "4,5,4,4"), then "PASS" or "FAIL":"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a quality validator for anachronism detection tasks. Be strict about historical accuracy and temporal impossibilities.",
                    },
                    {"role": "user", "content": validation_prompt},
                ],
                max_tokens=50,
                temperature=0.1,
            )

            result = response.choices[0].message.content.strip()
            is_valid = "PASS" in result.upper()

            # Cache the result
            self.validation_cache[cache_key] = is_valid

            return is_valid

        except Exception as e:
            print(f"Error validating sample: {e}")
            return False

    def generate_samples_by_category(
        self, category: str, target_count: int, batch_size: int = 10
    ) -> List[Dict]:
        """Generate samples for a specific category using batch generation."""

        print(
            f"Generating {target_count} samples for {category} in batches of {batch_size} pairs..."
        )

        samples = []
        # Each batch generates batch_size pairs (2 * batch_size samples)
        pairs_needed = target_count // 2
        batches_needed = (pairs_needed + batch_size - 1) // batch_size  # Round up

        for batch_num in range(batches_needed):
            print(
                f"  Generating batch {batch_num + 1}/{batches_needed} for {category}..."
            )

            # Calculate how many pairs needed for this batch
            remaining_pairs = pairs_needed - (len(samples) // 2)
            current_batch_pairs = min(batch_size, remaining_pairs)

            if current_batch_pairs <= 0:
                break

            # Generate batch using o3
            batch_samples = self.generate_batch_samples(category, current_batch_pairs)

            if batch_samples:
                samples.extend(batch_samples)
                print(
                    f"    Successfully generated {len(batch_samples)} samples ({len(batch_samples)//2} pairs)"
                )
            else:
                print(f"    Failed to generate batch {batch_num + 1}")

            # Add small delay between batches to avoid rate limiting
            time.sleep(2)

            if len(samples) >= target_count:
                break

        print(f"  Generated {len(samples)}/{target_count} samples for {category}")
        return samples[:target_count]


def load_existing_dataset() -> List[Dict]:
    """Load existing anachronisms dataset."""
    with open(
        "/Users/kyle/Documents/ws/post-hoc-reasoning/data/anachronisms/anachronisms.json",
        "r",
    ) as f:
        data = json.load(f)
        return data["examples"]


def main():
    """Generate 974 new anachronisms samples."""

    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in .env file")
        return

    print("Initializing Anachronisms Generator...")
    generator = AnachronismGenerator(api_key)

    print("Loading existing dataset...")
    existing_samples = load_existing_dataset()
    print(f"Found {len(existing_samples)} existing samples")

    # Target distribution for 974 new samples
    target_distribution = {
        "technology_displacement": 341,  # 35%
        "temporal_displacement": 244,  # 25%
        "cultural_anachronisms": 195,  # 20%
        "scientific_anachronisms": 146,  # 15%
        "institutional_anachronisms": 48,  # 5%
    }

    print(f"Target generation: {sum(target_distribution.values())} samples")
    print("Distribution:", target_distribution)

    # Generate samples for each category using batch generation
    all_new_samples = []
    batch_size = (
        5  # Generate in batches of 5 pairs (10 samples) - smaller for stability
    )

    for category, target_count in target_distribution.items():
        print(f"\n=== GENERATING {category.upper()} ===")
        category_samples = generator.generate_samples_by_category(
            category, target_count, batch_size
        )
        all_new_samples.extend(category_samples)
        print(
            f"✓ Generated {len(category_samples)}/{target_count} samples for {category}"
        )

    # Shuffle the combined samples
    random.shuffle(all_new_samples)

    print(f"\nGenerated {len(all_new_samples)} total new samples")

    # Save new samples only
    print("Saving anachronisms_new_samples.json...")
    new_samples_data = {"examples": all_new_samples}
    with open(
        "/Users/kyle/Documents/ws/post-hoc-reasoning/anachronisms_new_samples.json", "w"
    ) as f:
        json.dump(new_samples_data, f, indent=2)

    # Create combined dataset
    print("Creating combined dataset...")
    combined_samples = existing_samples + all_new_samples
    random.shuffle(combined_samples)

    print(f"Combined dataset has {len(combined_samples)} samples")

    # Save combined dataset
    print("Saving anachronisms_new_full.json...")
    combined_data = {"examples": combined_samples}
    with open(
        "/Users/kyle/Documents/ws/post-hoc-reasoning/anachronisms_new_full.json", "w"
    ) as f:
        json.dump(combined_data, f, indent=2)

    print("\nGeneration complete!")
    print(f"Files created:")
    print(f"- anachronisms_new_samples.json ({len(all_new_samples)} samples)")
    print(f"- anachronisms_new_full.json ({len(combined_samples)} samples)")

    # Summary statistics
    print(f"\n=== GENERATION SUMMARY ===")
    for category, target in target_distribution.items():
        actual = len(
            [
                s
                for s in all_new_samples
                if "metadata" in s and s.get("metadata", {}).get("category") == category
            ]
        )
        print(f"{category}: {actual}/{target} samples")


if __name__ == "__main__":
    main()
