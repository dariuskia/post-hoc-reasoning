#!/usr/bin/env python3
"""
Logical Deduction Dataset Generator

Generates 900 new logical deduction samples using systematic matrix generation
across 5 domains: books, fruits, golf, cars, and birds.

Target: 300 new scenarios × 3 questions each = 900 total samples
Final dataset: 300 existing + 900 new = 1200 samples
"""

import json
import random
import re
from collections import defaultdict
from itertools import permutations
from typing import Dict, List, Optional, Set, Tuple

# Set seed for reproducibility
random.seed(42)

# Domain-specific object databases
OBJECTS = {
    "books": {
        "context": "On a shelf, there are three books",
        "object_type": "book",
        "colors": [
            "black",
            "orange",
            "blue",
            "purple",
            "red",
            "yellow",
            "white",
            "green",
            "brown",
            "gray",
            "silver",
            "gold",
            "pink",
            "maroon",
            "navy",
            "teal",
            "violet",
            "crimson",
            "ivory",
            "copper",
        ],
    },
    "fruits": {
        "context": "A fruit stand sells three fruits",
        "object_type": "fruit",
        "items": [
            "apples",
            "cantaloupes",
            "kiwis",
            "loquats",
            "mangoes",
            "oranges",
            "peaches",
            "pears",
            "plums",
            "watermelons",
            "grapes",
            "berries",
            "bananas",
            "lemons",
            "limes",
            "cherries",
            "apricots",
            "papayas",
            "coconuts",
            "avocados",
        ],
    },
    "golf": {
        "context": "In a golf tournament, there were three golfers",
        "object_type": "golfer",
        "names": [
            "Amy",
            "Ana",
            "Joe",
            "Dan",
            "Eve",
            "Ben",
            "Cal",
            "Ivy",
            "Max",
            "Sue",
            "Wei",
            "Arjun",
            "Sofia",
            "Dmitri",
            "Yuki",
            "Priya",
            "Omar",
            "Elena",
            "Hassan",
            "Maya",
            "Finn",
            "Zara",
            "Kai",
            "Lucia",
            "Ravi",
            "Nora",
            "Sasha",
            "Camila",
            "Ahmed",
            "Iris",
        ],
    },
    "cars": {
        "context": "In an antique car show, there are three vehicles",
        "object_type": "vehicle",
        "types": [
            "bus",
            "sedan",
            "truck",
            "hatchback",
            "limousine",
            "station wagon",
            "SUV",
            "convertible",
            "coupe",
            "van",
            "pickup",
            "motorcycle",
            "minivan",
            "roadster",
            "jeep",
            "wagon",
            "compact",
            "sports car",
        ],
    },
    "birds": {
        "context": "On a branch, there are three birds",
        "object_type": "bird",
        "species": [
            "robin",
            "raven",
            "quail",
            "hawk",
            "cardinal",
            "eagle",
            "sparrow",
            "owl",
            "finch",
            "jay",
            "wren",
            "dove",
            "crow",
            "pigeon",
            "parrot",
            "falcon",
            "heron",
            "pelican",
            "penguin",
            "ostrich",
        ],
    },
}

# Constraint templates by domain type
CONSTRAINT_TEMPLATES = {
    "spatial": [  # Books and birds
        "{B} is to the right of {A}. {C} is to the right of {B}.",
        "{B} is to the left of {C}. {A} is to the left of {B}.",
        "{A} is the leftmost. {B} is to the right of {A}.",
        "{C} is the rightmost. {B} is to the left of {C}.",
    ],
    "price": [  # Fruits
        "{A} costs more than {B}. {B} costs more than {C}.",
        "{C} is the cheapest. {A} costs more than {C}.",
        "{A} is the most expensive. {B} costs less than {A}.",
        "{B} costs more than {C}. {A} costs more than {B}.",
        "{C} costs less than {B}. {B} costs less than {A}.",
        "{A} is the most expensive. {C} is the cheapest.",
    ],
    "age": [  # Cars
        "{A} is older than {B}. {B} is older than {C}.",
        "{C} is the newest. {A} is older than {C}.",
        "{A} is the oldest. {B} is newer than {C}.",
        "{B} is older than {C}. {A} is older than {B}.",
        "{C} is newer than {B}. {B} is newer than {A}.",
        "{A} is the oldest. {C} is the newest.",
    ],
    "ranking": [  # Golf
        "{A} finished above {B}. {B} finished above {C}.",
        "{C} finished last. {A} finished above {C}.",
        "{A} finished first. {B} finished below {A}.",
        "{B} finished above {C}. {A} finished above {B}.",
        "{C} finished below {B}. {B} finished below {A}.",
        "{A} finished first. {C} finished last.",
    ],
}

# Question templates by domain
QUESTION_TEMPLATES = {
    "spatial": {
        "first": "{obj} is the leftmost.",
        "second": "{obj} is the second from the left.",
        "third": "{obj} is the rightmost.",
    },
    "price": {
        "first": "{obj} is the most expensive.",
        "second": "{obj} is the second-most expensive.",
        "third": "{obj} is the cheapest.",
    },
    "age": {
        "first": "{obj} is the oldest.",
        "second": "{obj} is the second-oldest.",
        "third": "{obj} is the newest.",
    },
    "ranking": {
        "first": "{obj} finished first.",
        "second": "{obj} finished second.",
        "third": "{obj} finished last.",
    },
}


def load_existing_dataset() -> List[Dict]:
    """Load the existing logical deduction dataset."""
    with open(
        "/Users/kyle/Documents/ws/post-hoc-reasoning/data/logical_deduction/logical_deduction.json",
        "r",
    ) as f:
        return json.load(f)


def normalize_scenario(scenario_input: str) -> str:
    """Normalize scenario for duplicate detection."""
    # Remove specific names/colors and focus on structure
    normalized = re.sub(
        r"\b(a|an|the)\s+\w+\s+(book|fruit|golfer|vehicle|bird)",
        r"OBJ",
        scenario_input.lower(),
    )
    # Remove extra whitespace
    return " ".join(normalized.split())


def create_existing_scenarios_set(existing_data: List[Dict]) -> Set[str]:
    """Create set of existing scenarios to avoid duplicates."""
    scenarios = set()
    for item in existing_data:
        normalized = normalize_scenario(item["input"])
        scenarios.add(normalized)
    return scenarios


class ConstraintSolver:
    """Solves logical constraints to determine valid object ordering."""

    def __init__(self):
        self.constraints = []
        self.objects = []

    def parse_constraints(
        self, template: str, obj_mapping: Dict[str, str]
    ) -> List[Tuple[str, str, str]]:
        """Parse constraint template into relationship tuples."""
        constraints = []
        filled_template = template.format(**obj_mapping)

        # Create reverse mapping to find original object names
        reverse_mapping = {v: k for k, v in obj_mapping.items()}

        # Helper function to extract object references
        def extract_object_refs(text):
            object_refs = []
            # Try to match against our known objects first
            for obj_name in obj_mapping.values():
                if obj_name in text:
                    object_refs.append(obj_name)
            return object_refs

        # Helper function to clean object names
        def clean_object_name(name):
            return name.strip().rstrip(".")

        # Split filled template into sentences
        sentences = filled_template.split(". ")

        # Parse spatial relationships
        for sentence in sentences:
            if "is to the right of" in sentence:
                match = re.search(r"^(.+?) is to the right of (.+?)$", sentence.strip())
                if match:
                    right_obj = clean_object_name(match.group(1))
                    left_obj = clean_object_name(match.group(2))
                    constraints.append((left_obj, "before", right_obj))

            if "is to the left of" in sentence:
                match = re.search(r"^(.+?) is to the left of (.+?)$", sentence.strip())
                if match:
                    left_obj = clean_object_name(match.group(1))
                    right_obj = clean_object_name(match.group(2))
                    constraints.append((left_obj, "before", right_obj))

        # Parse absolute positions
        for sentence in sentences:
            sentence = sentence.strip()

            if "is the leftmost" in sentence:
                match = re.search(r"^(.+?) is the leftmost\.?$", sentence)
                if match:
                    obj = clean_object_name(match.group(1))
                    constraints.append((obj, "position", "first"))

            if "is the rightmost" in sentence:
                match = re.search(r"^(.+?) is the rightmost$", sentence)
                if match:
                    obj = clean_object_name(match.group(1))
                    constraints.append((obj, "position", "third"))

        # Parse comparative and absolute relationships
        for sentence in sentences:
            sentence = sentence.strip()

            # Comparative relationships
            if "costs more than" in sentence:
                match = re.search(r"^(.+?) costs more than (.+?)$", sentence)
                if match:
                    constraints.append(
                        (
                            clean_object_name(match.group(1)),
                            "better_than",
                            clean_object_name(match.group(2)),
                        )
                    )

            if "finished above" in sentence:
                match = re.search(r"^(.+?) finished above (.+?)$", sentence)
                if match:
                    constraints.append(
                        (
                            clean_object_name(match.group(1)),
                            "better_than",
                            clean_object_name(match.group(2)),
                        )
                    )

            if "is older than" in sentence:
                match = re.search(r"^(.+?) is older than (.+?)$", sentence)
                if match:
                    constraints.append(
                        (
                            clean_object_name(match.group(1)),
                            "better_than",
                            clean_object_name(match.group(2)),
                        )
                    )

            if "costs less than" in sentence:
                match = re.search(r"^(.+?) costs less than (.+?)$", sentence)
                if match:
                    constraints.append(
                        (
                            clean_object_name(match.group(2)),
                            "better_than",
                            clean_object_name(match.group(1)),
                        )
                    )

            if "finished below" in sentence:
                match = re.search(r"^(.+?) finished below (.+?)$", sentence)
                if match:
                    constraints.append(
                        (
                            clean_object_name(match.group(2)),
                            "better_than",
                            clean_object_name(match.group(1)),
                        )
                    )

            if "is newer than" in sentence:
                match = re.search(r"^(.+?) is newer than (.+?)$", sentence)
                if match:
                    constraints.append(
                        (
                            clean_object_name(match.group(2)),
                            "better_than",
                            clean_object_name(match.group(1)),
                        )
                    )

            # Absolute rankings
            if "is the most expensive" in sentence:
                match = re.search(r"^(.+?) is the most expensive$", sentence)
                if match:
                    constraints.append(
                        (clean_object_name(match.group(1)), "position", "first")
                    )

            if "finished first" in sentence:
                match = re.search(r"^(.+?) finished first$", sentence)
                if match:
                    constraints.append(
                        (clean_object_name(match.group(1)), "position", "first")
                    )

            if "is the oldest" in sentence:
                match = re.search(r"^(.+?) is the oldest$", sentence)
                if match:
                    constraints.append(
                        (clean_object_name(match.group(1)), "position", "first")
                    )

            if "is the cheapest" in sentence:
                match = re.search(r"^(.+?) is the cheapest$", sentence)
                if match:
                    constraints.append(
                        (clean_object_name(match.group(1)), "position", "third")
                    )

            if "finished last" in sentence:
                match = re.search(r"^(.+?) finished last$", sentence)
                if match:
                    constraints.append(
                        (clean_object_name(match.group(1)), "position", "third")
                    )

            if "is the newest" in sentence:
                match = re.search(r"^(.+?) is the newest$", sentence)
                if match:
                    constraints.append(
                        (clean_object_name(match.group(1)), "position", "third")
                    )

        return constraints

    def solve_ordering(
        self, constraints: List[Tuple[str, str, str]], objects: List[str]
    ) -> Optional[List[str]]:
        """Solve constraints to find valid total ordering."""
        # Try all permutations to find one that satisfies constraints
        for perm in permutations(objects):
            if self.validate_ordering(perm, constraints):
                return list(perm)
        return None

    def validate_ordering(
        self, ordering: Tuple[str], constraints: List[Tuple[str, str, str]]
    ) -> bool:
        """Check if ordering satisfies all constraints."""
        pos_map = {obj: i for i, obj in enumerate(ordering)}

        for constraint in constraints:
            obj1, relation, obj2_or_pos = constraint

            if relation == "before":  # obj1 comes before obj2
                if pos_map[obj1] >= pos_map[obj2_or_pos]:
                    return False
            elif relation == "better_than":  # obj1 ranks higher than obj2 (lower index)
                if pos_map[obj1] >= pos_map[obj2_or_pos]:
                    return False
            elif relation == "position":  # obj1 has specific position
                if obj2_or_pos == "first" and pos_map[obj1] != 0:
                    return False
                elif obj2_or_pos == "third" and pos_map[obj1] != 2:
                    return False

        return True


class DomainGenerator:
    """Base class for domain-specific scenario generation."""

    def __init__(self, domain_name: str, domain_config: Dict):
        self.domain = domain_name
        self.config = domain_config
        self.solver = ConstraintSolver()

        # Determine domain type for templates
        if domain_name in ["books", "birds"]:
            self.template_type = "spatial"
            self.question_type = "spatial"
        elif domain_name == "fruits":
            self.template_type = "price"
            self.question_type = "price"
        elif domain_name == "cars":
            self.template_type = "age"
            self.question_type = "age"
        elif domain_name == "golf":
            self.template_type = "ranking"
            self.question_type = "ranking"

    def get_objects(self) -> List[str]:
        """Get list of available objects for this domain."""
        if self.domain == "books":
            return [f"a {color} book" for color in self.config["colors"]]
        elif self.domain == "fruits":
            return self.config["items"]
        elif self.domain == "golf":
            return self.config["names"]
        elif self.domain == "cars":
            return [f"a {vehicle_type}" for vehicle_type in self.config["types"]]
        elif self.domain == "birds":
            return [f"a {species}" for species in self.config["species"]]

    def generate_scenario(self) -> Optional[Dict]:
        """Generate a single logical deduction scenario."""
        # Select 3 random objects
        available_objects = self.get_objects()
        selected_objects = random.sample(available_objects, 3)

        # Select constraint template
        template = random.choice(CONSTRAINT_TEMPLATES[self.template_type])

        # Create object mapping for template
        obj_mapping = {
            "A": selected_objects[0],
            "B": selected_objects[1],
            "C": selected_objects[2],
        }

        # Parse constraints
        constraints = self.solver.parse_constraints(template, obj_mapping)

        # Debug print for troubleshooting
        # if self.domain == "books":
        #     print(f"Template: {template}")
        #     print(f"Objects: {selected_objects}")
        #     print(f"Constraints: {constraints}")
        #     print(f"Filled template: {template.format(**obj_mapping)}")
        #     print("---")

        # Solve for valid ordering
        ordering = self.solver.solve_ordering(constraints, selected_objects)
        if not ordering:
            return None  # No valid solution

        # Generate input description
        input_desc = self.format_input_description(template, obj_mapping)

        # Generate target questions (3 separate question sets)
        question_sets = self.generate_target_questions(ordering)

        # Return 3 separate samples for this scenario
        samples = []
        for question_set in question_sets:
            samples.append({"input": input_desc, "target_scores": question_set})

        return samples

    def format_input_description(
        self, template: str, obj_mapping: Dict[str, str]
    ) -> str:
        """Format the input description for the scenario."""
        filled_template = template.format(**obj_mapping)
        context = self.config["context"]

        # Create object list
        objects_list = f"{obj_mapping['A']}, {obj_mapping['B']}, and {obj_mapping['C']}"

        return f"{context}: {objects_list}. {filled_template}"

    def generate_target_questions(self, ordering: List[str]) -> List[Dict[str, int]]:
        """Generate 3 separate question sets (one for each position)."""
        templates = QUESTION_TEMPLATES[self.question_type]
        question_sets = []

        # Generate separate question set for each position being tested
        for test_position in ["first", "second", "third"]:
            questions = {}
            for obj in ordering:
                question = templates[test_position].format(obj=obj)
                # Determine correct answer based on object's position in ordering
                obj_position = ordering.index(obj)
                position_index = ["first", "second", "third"].index(test_position)
                questions[question] = 1 if obj_position == position_index else 0
            question_sets.append(questions)

        return question_sets


def generate_domain_samples(
    domain_name: str, target_count: int, existing_scenarios: Set[str]
) -> List[Dict]:
    """Generate samples for a specific domain."""
    print(f"Generating {target_count} scenarios for {domain_name}...")

    domain_config = OBJECTS[domain_name]
    generator = DomainGenerator(domain_name, domain_config)

    samples = []
    attempts = 0
    # Books domain seems to have more constraint conflicts, increase attempts
    max_attempts = target_count * 50 if domain_name == "books" else target_count * 20

    while len(samples) < target_count and attempts < max_attempts:
        attempts += 1

        scenario_samples = generator.generate_scenario()
        if scenario_samples is None:
            continue

        # Check for duplicates using the first sample's input
        normalized = normalize_scenario(scenario_samples[0]["input"])
        if normalized in existing_scenarios:
            continue

        # Add all 3 samples from this scenario
        samples.extend(scenario_samples)
        existing_scenarios.add(normalized)

        if len(samples) >= target_count:
            break

    return samples[:target_count]


def main():
    """Generate 900 new logical deduction samples."""
    print("Loading existing dataset...")
    existing_data = load_existing_dataset()
    existing_scenarios = create_existing_scenarios_set(existing_data)

    print(f"Found {len(existing_data)} existing samples")
    print(f"Found {len(existing_scenarios)} unique existing scenarios")

    # Generate samples for each domain (180 each)
    all_new_samples = []
    domains = ["books", "fruits", "golf", "cars", "birds"]
    samples_per_domain = 180

    for domain in domains:
        domain_samples = generate_domain_samples(
            domain, samples_per_domain, existing_scenarios
        )
        all_new_samples.extend(domain_samples)
        print(f"Generated {len(domain_samples)} samples for {domain}")

    # Shuffle the combined samples
    random.shuffle(all_new_samples)

    print(f"\nGenerated {len(all_new_samples)} total new samples")

    # Save new samples
    print("Saving logical_deduction_new_samples.json...")
    with open(
        "/Users/kyle/Documents/ws/post-hoc-reasoning/logical_deduction_new_samples.json",
        "w",
    ) as f:
        json.dump(all_new_samples, f, indent=2)

    # Create combined dataset
    print("Creating combined dataset...")
    combined_data = existing_data + all_new_samples
    random.shuffle(combined_data)

    print(f"Combined dataset has {len(combined_data)} samples")

    # Save combined dataset
    print("Saving logical_deduction_new_full.json...")
    with open(
        "/Users/kyle/Documents/ws/post-hoc-reasoning/logical_deduction_new_full.json",
        "w",
    ) as f:
        json.dump(combined_data, f, indent=2)

    print("\nGeneration complete!")
    print(f"Files created:")
    print(f"- logical_deduction_new_samples.json ({len(all_new_samples)} samples)")
    print(f"- logical_deduction_new_full.json ({len(combined_data)} samples)")


if __name__ == "__main__":
    main()
