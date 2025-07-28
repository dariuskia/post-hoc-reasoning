#!/usr/bin/env python3
"""
Sports Understanding Dataset Generator

Generates 950 new sports understanding samples using systematic matrix generation
based on athlete-action combinations from the existing dataset analysis.

Target: 475 plausible + 475 implausible = 950 total samples
Final dataset: 250 existing + 950 new = 1200 samples with 50/50 balance
"""

import json
import random
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple

# Set seed for reproducibility
random.seed(42)

# Athlete database by sport (from analysis)
ATHLETES = {
    "basketball": [
        "Jonas Valanciunas",
        "LaMelo Ball",
        "Zach LaVine",
        "Malcolm Brogdon",
        "Draymond Green",
        "Fred VanVleet",
        "Kevin Durant",
        "Jayson Tatum",
        "Anthony Davis",
        "Kawhi Leonard",
        "Clint Capela",
        "Norman Powell",
        "Dejounte Murray",
        "Ben Simmons",
        "Kendrick Nunn",
        "Jamal Murray",
        "Domantas Sabonis",
        "De'Aaron Fox",
        "James Harden",
        "Trae Young",
        "Mitchell Robinson",
        "Collin Sexton",
        "Russell Westbrook",
        "Stephen Curry",
        "Mikal Bridges",
        "Jaylen Brown",
        "Malik Beasley",
        "LeBron James",
        "Giannis Antetokounmpo",
        "Nikola Jokic",
        "Luka Doncic",
        "Joel Embiid",
    ],
    "hockey": [
        "Elias Lindholm",
        "John Carlson",
        "John Tavares",
        "Robin Lehner",
        "Dougie Hamilton",
        "Frederik Andersen",
        "Nazem Kadri",
        "Jakub Vrana",
        "Tom Wilson",
        "Aleksander Barkov",
        "Jonathan Marchessault",
        "Connor McDavid",
        "Patrick Kane",
        "Steven Stamkos",
        "Patrice Bergeron",
        "Nathan MacKinnon",
        "Gabriel Landeskog",
        "Ryan Nugent-Hopkins",
        "Kyle Connor",
        "Teuvo Teravainen",
        "Sidney Crosby",
        "Alex Ovechkin",
        "Auston Matthews",
        "Leon Draisaitl",
        "Artemi Panarin",
        "Mika Zibanejad",
        "David Pastrnak",
        "Brad Marchand",
        "Cale Makar",
        "Erik Karlsson",
        "Victor Hedman",
        "Andrei Vasilevskiy",
        "Igor Shesterkin",
        "Jonathan Huberdeau",
        "Matthew Tkachuk",
        "Johnny Gaudreau",
        "Mark Stone",
        "Jack Hughes",
        "Kirill Kaprizov",
        "Sebastian Aho",
        "Timo Meier",
        "Kailer Yamamoto",
        "Claude Giroux",
    ],
    "football": [
        "Deshaun Watson",
        "T.Y. Hilton",
        "Robert Woods",
        "Julian Edelman",
        "DJ Chark",
        "Philip Rivers",
        "Drew Brees",
        "Ryan Tannehill",
        "Brandin Cooks",
        "Tyler Boyd",
        "Carson Wentz",
        "Sterling Shepard",
        "Josh Allen",
        "Calvin Ridley",
        "Marvin Jones",
        "Cooper Kupp",
        "Keenan Allen",
        "Mitchell Trubisky",
        "Joe Burrow",
        "Tom Brady",
        "Tyreek Hill",
        "Adam Thielen",
        "Amari Cooper",
        "DK Metcalf",
        "Stefon Diggs",
        "DeAndre Hopkins",
        "Davante Adams",
        "Mike Evans",
        "Chris Godwin",
        "Tyler Lockett",
        "CeeDee Lamb",
        "Justin Jefferson",
        "Ja'Marr Chase",
        "Jonathan Taylor",
        "Derrick Henry",
        "Alvin Kamara",
        "Dalvin Cook",
        "Nick Chubb",
        "Austin Ekeler",
        "Aaron Jones",
        "Travis Kelce",
        "George Kittle",
    ],
    "baseball": [
        "Mookie Betts",
        "Jack Flaherty",
        "Freddie Freeman",
        "Michael Conforto",
        "Corbin Burnes",
        "Gleyber Torres",
        "Juan Soto",
        "Fernando Tatis Jr.",
        "Gerrit Cole",
        "Luis Robert",
        "Mike Trout",
        "Walker Buehler",
        "Pete Alonso",
        "Anthony Rizzo",
        "Francisco Lindor",
        "George Springer",
        "Blake Snell",
        "Kyle Tucker",
        "Bryce Harper",
        "Ronald Acuna Jr.",
        "Vladimir Guerrero Jr.",
        "Bo Bichette",
        "Shohei Ohtani",
        "Aaron Judge",
        "Giancarlo Stanton",
        "Jose Altuve",
        "Alex Bregman",
        "Yordan Alvarez",
        "Kyle Schwarber",
        "Trea Turner",
        "Manny Machado",
        "Xander Bogaerts",
        "Rafael Devers",
        "Jose Ramirez",
    ],
    "soccer": [
        "Marcelo",
        "David Silva",
        "Carles Puyol",
        "Kwadwo Asamoah",
        "Giorgio Chiellini",
        "Gerard Pique",
        "Neymar",
        "Santi Cazorla",
        "Edinson Cavani",
        "Petr Cech",
        "Robert Lewandowski",
        "Wayne Rooney",
        "Mario Gomez",
        "Andres Iniesta",
        "Yaya Toure",
        "Mario Balotelli",
        "Toni Kroos",
        "Sergio Busquets",
        "Sergio Ramos",
        "Angel Di Maria",
        "Lionel Messi",
        "Cristiano Ronaldo",
        "Kevin De Bruyne",
        "Mohamed Salah",
        "Sadio Mane",
        "Virgil van Dijk",
        "Kylian Mbappe",
        "Erling Haaland",
        "Karim Benzema",
        "Luka Modric",
        "N'Golo Kante",
        "Paul Pogba",
        "Bruno Fernandes",
        "Harry Kane",
        "Son Heung-min",
        "Raheem Sterling",
        "Riyad Mahrez",
        "Jadon Sancho",
        "Marcus Rashford",
        "Arjen Robben",
        "Willian",
        "Pepe",
        "Pedro",
    ],
}

# Action database by sport (from analysis)
ACTIONS = {
    "basketball": [
        "beat the buzzer",
        "hit the buzzer beater",
        "scored a reverse layup",
        "scored a reverse dunk",
        "scored a windmill dunk",
        "dunked the ball",
        "banked the shot",
        "scored the easy layup",
        "took a three",
        "shot from beyond the arc",
        "launched the half court shot",
        "took a side-step three",
        "took a turnaround jumper",
        "airballed the shot",
        "hit nothing but net",
        "eurostepped to the basket",
        "drove into the restricted area",
        "called for the screen",
        "set the hard screen",
        "committed a three second violation",
        "was called for the goal tend",
        "took a charge",
        "committed a blocking foul",
        "beat the shot clock",
    ],
    "hockey": [
        "shot the puck",
        "passed the puck",
        "lost control of the puck",
        "took a backhand shot",
        "wristed a shot",
        "backhanded a shot",
        "crossed the blue line",
        "skated behind the net",
        "skated backwards",
        "entered the attacking zone",
        "earned a trip to the penalty box",
        "spent time in the penalty box",
        "killed the powerplay",
        "scored on the power play",
        "scored in the third period",
        "went five hole",
        "was called for icing",
        "was called for slashing",
        "launched the desperation heave",
    ],
    "football": [
        "threw a touchdown",
        "caught the screen pass",
        "hit the screen pass",
        "caught the back shoulder fade",
        "hit the slant pass",
        "launched a hail mary",
        "hit the wheel route",
        "converted the first down",
        "went for it on fourth down",
        "got into the endzone",
        "gained five yards",
        "changed direction in the backfield",
        "ran out of bounds",
        "fumbled the ball",
        "was flagged on the play",
        "drew a flag on the play",
        "scored a touchdown",
        "took the snap",
    ],
    "baseball": [
        "hit a walkoff homer",
        "hit a triple",
        "hit a double",
        "hit a single",
        "got a base hit",
        "hit into a double play",
        "walked to first base",
        "was out at home",
        "was out at second",
        "was out at first",
        "was safe at first",
        "stepped on first base",
        "got on base",
        "struck out the side",
        "threw to first base",
        "worked a full count",
        "watched the pitch go by",
        "walked on ball four",
        "took ball four",
        "grounded out to second base",
    ],
    "soccer": [
        "got on the end of a through ball",
        "did a maradona on the defender",
        "nutmegged the defender",
        "maradona'd the defender",
        "performed a give and go",
        "did a double stepover",
        "scored a freekick",
        "scored a penalty kick",
        "scored a bicycle kick",
        "took a left footed shot",
        "shot with the left foot",
        "shot from the six yard line",
        "shot from outside the eighteen",
        "took a throw in",
        "scored a corner kick",
        "earned an indirect kick",
        "performed a slide tackle",
        "committed a handball",
        "went in studs up",
        "earned a red card",
        "was flagged on the play",
        "scored in extra time",
        "scored in added time",
        "scored a header goal",
    ],
}

# Tournament contexts by sport (from analysis)
TOURNAMENTS = {
    "basketball": [
        "NBA Championship",
        "Eastern Conference Finals",
        "Western Conference Finals",
        "NBA Finals",
        "Conference Semifinals",
        "First Round",
    ],
    "hockey": [
        "Stanley Cup",
        "Stanley Cup Finals",
        "Conference Finals",
        "Stanley Cup Playoffs",
        "Winter Classic",
    ],
    "football": [
        "Superbowl",
        "AFC Championship",
        "NFC Championship",
        "AFC Divisional",
        "NFC Divisional",
        "Wild Card Round",
    ],
    "baseball": [
        "World Series",
        "League Championship",
        "Division Series",
        "Wild Card Game",
        "All-Star Game",
    ],
    "soccer": [
        "Champions League",
        "FA Cup",
        "European Cup",
        "Champions League Final",
        "Premier League",
        "World Cup",
        "UEFA Euro",
    ],
}


def load_existing_dataset() -> List[Dict]:
    """Load the existing sports understanding dataset."""
    with open(
        "/Users/kyle/Documents/ws/post-hoc-reasoning/tmp/sports_understanding_raw.json",
        "r",
    ) as f:
        data = json.load(f)
        return data["examples"]


def extract_sentence_content(input_text: str) -> str:
    """Extract the main sentence content from input text."""
    # Extract sentence from "Is the following sentence plausible? \"SENTENCE\""
    import re

    match = re.search(r'"([^"]+)"', input_text)
    if match:
        return match.group(1)
    return input_text.strip()


def create_existing_sentences_set(existing_data: List[Dict]) -> Set[str]:
    """Create a set of existing sentences to avoid duplicates."""
    sentences = set()
    for item in existing_data:
        sentence = extract_sentence_content(item["input"])
        sentences.add(sentence.lower())
    return sentences


def generate_plausible_samples(
    target_count: int, existing_sentences: Set[str]
) -> List[Dict]:
    """Generate plausible samples (same sport athlete-action pairs)."""
    samples = []
    attempts = 0
    max_attempts = target_count * 10

    while len(samples) < target_count and attempts < max_attempts:
        attempts += 1

        # Randomly select a sport
        sport = random.choice(list(ATHLETES.keys()))
        athlete = random.choice(ATHLETES[sport])
        action = random.choice(ACTIONS[sport])

        # 20% chance to add tournament context
        if random.random() < 0.20:
            tournament = random.choice(TOURNAMENTS[sport])
            sentence = f"{athlete} {action} in the {tournament}."
        else:
            sentence = f"{athlete} {action}."

        # Check for duplicates
        if sentence.lower() not in existing_sentences:
            formatted_input = f'Is the following sentence plausible? "{sentence}"'
            samples.append({"input": formatted_input, "target": "yes"})
            existing_sentences.add(sentence.lower())

    return samples


def generate_implausible_samples(
    target_count: int, existing_sentences: Set[str]
) -> List[Dict]:
    """Generate implausible samples (cross-sport athlete-action pairs)."""
    samples = []
    attempts = 0
    max_attempts = target_count * 10

    sports_list = list(ATHLETES.keys())

    while len(samples) < target_count and attempts < max_attempts:
        attempts += 1

        # Select two different sports
        athlete_sport = random.choice(sports_list)
        action_sport = random.choice([s for s in sports_list if s != athlete_sport])

        athlete = random.choice(ATHLETES[athlete_sport])
        action = random.choice(ACTIONS[action_sport])

        # 15% chance to add wrong tournament context for extra confusion
        if random.random() < 0.15:
            tournament_sport = random.choice(
                [s for s in sports_list if s != athlete_sport]
            )
            tournament = random.choice(TOURNAMENTS[tournament_sport])
            sentence = f"{athlete} {action} in the {tournament}."
        else:
            sentence = f"{athlete} {action}."

        # Check for duplicates
        if sentence.lower() not in existing_sentences:
            formatted_input = f'Is the following sentence plausible? "{sentence}"'
            samples.append({"input": formatted_input, "target": "no"})
            existing_sentences.add(sentence.lower())

    return samples


def apply_difficulty_distribution(samples: List[Dict]) -> List[Dict]:
    """Apply difficulty and technical terminology patterns."""
    # Technical terms by sport for moderate complexity
    technical_terms = {
        "basketball": [
            "eurostepped",
            "three second violation",
            "goal tend",
            "shot clock",
        ],
        "hockey": ["blue line", "penalty box", "powerplay", "five hole", "icing"],
        "football": ["back shoulder fade", "slant pass", "hail mary", "wheel route"],
        "baseball": [
            "walkoff homer",
            "struck out the side",
            "full count",
            "double play",
        ],
        "soccer": ["maradona", "nutmegged", "bicycle kick", "studs up", "give and go"],
    }

    # Track complexity
    for sample in samples:
        sentence = extract_sentence_content(sample["input"]).lower()

        # Check for technical terminology (moderate complexity)
        has_technical = any(
            term in sentence for terms in technical_terms.values() for term in terms
        )

        # Check for tournament context (can add complexity)
        has_tournament = any(
            tournament.lower() in sentence
            for tournaments in TOURNAMENTS.values()
            for tournament in tournaments
        )

        if has_technical and has_tournament:
            sample["complexity"] = "complex"
        elif has_technical:
            sample["complexity"] = "moderate"
        else:
            sample["complexity"] = "simple"

    return samples


def main():
    """Generate 950 new sports understanding samples."""
    print("Loading existing dataset...")
    existing_data = load_existing_dataset()
    existing_sentences = create_existing_sentences_set(existing_data)

    print(f"Found {len(existing_data)} existing samples")
    print(f"Found {len(existing_sentences)} unique sentences")

    # Generate samples
    print("\nGenerating 475 plausible samples...")
    plausible_samples = generate_plausible_samples(475, existing_sentences)
    print(f"Generated {len(plausible_samples)} plausible samples")

    print("\nGenerating 475 implausible samples...")
    implausible_samples = generate_implausible_samples(475, existing_sentences)
    print(f"Generated {len(implausible_samples)} implausible samples")

    # Combine and shuffle
    new_samples = plausible_samples + implausible_samples
    random.shuffle(new_samples)

    # Apply difficulty distribution
    print("\nApplying difficulty distribution...")
    new_samples = apply_difficulty_distribution(new_samples)

    # Statistics
    complexity_stats = defaultdict(int)
    target_stats = defaultdict(int)
    for sample in new_samples:
        complexity_stats[sample.get("complexity", "unknown")] += 1
        target_stats[sample["target"]] += 1

    print(f"\nGenerated {len(new_samples)} total new samples")
    print(f"Target distribution: {dict(target_stats)}")
    print(f"Complexity distribution: {dict(complexity_stats)}")

    # Save new samples
    print("\nSaving sports_understanding_new_samples.json...")
    with open(
        "/Users/kyle/Documents/ws/post-hoc-reasoning/sports_understanding_new_samples.json",
        "w",
    ) as f:
        json.dump(new_samples, f, indent=2)

    # Create combined dataset
    print("Creating combined dataset...")
    combined_data = existing_data + new_samples
    random.shuffle(combined_data)

    print(f"Combined dataset has {len(combined_data)} samples")

    # Combined statistics
    combined_target_stats = defaultdict(int)
    for sample in combined_data:
        combined_target_stats[sample["target"]] += 1

    print(f"Combined target distribution: {dict(combined_target_stats)}")

    # Save combined dataset
    print("Saving sports_understanding_new_full.json...")
    with open(
        "/Users/kyle/Documents/ws/post-hoc-reasoning/sports_understanding_new_full.json",
        "w",
    ) as f:
        json.dump(combined_data, f, indent=2)

    print("\nGeneration complete!")
    print(f"Files created:")
    print(f"- sports_understanding_new_samples.json ({len(new_samples)} samples)")
    print(f"- sports_understanding_new_full.json ({len(combined_data)} samples)")


if __name__ == "__main__":
    main()
