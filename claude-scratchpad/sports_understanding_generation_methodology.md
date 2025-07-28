# Sports Understanding Dataset Generation Methodology

**Generated:** 2025-07-28  
**Target:** 950 new samples (475 plausible + 475 implausible)  
**Final Dataset:** 1200 samples (250 original + 950 new)

## Overview

This document describes the systematic approach used to expand the Sports Understanding dataset from 250 to 1200 samples while maintaining high quality and achieving perfect 50/50 label balance.

## 1. Initial Analysis Phase

### 1.1 Original Dataset Structure
- **Source:** `/tmp/sports_understanding_raw.json`
- **Format:** `{"canary": "...", "examples": [{"input": "Is the following sentence plausible? \"SENTENCE\"", "target": "yes/no"}]}`
- **Total samples:** 250
- **Label distribution:** 135 implausible (54%) + 115 plausible (46%)
- **Unique sentences:** 248 (2 duplicates found)

### 1.2 Comprehensive Analysis
Created detailed analysis in `claude-scratchpad/sports_understanding_analysis.md` covering:
- **Athletes by sport:** 191 unique athletes across 5 sports
- **Actions by sport:** 104 unique sport-specific actions  
- **Tournament contexts:** 56 examples (22.4%) included championship/tournament references
- **Mismatch patterns:** 4 types creating implausible examples
- **Difficulty levels:** Simple (80%), Moderate (20%), Complex (minimal)

## 2. Component Database Extraction

### 2.1 Athletes Database
Extracted and categorized by sport:
```python
ATHLETES = {
    "basketball": [32 players] # LeBron James, Stephen Curry, etc.
    "hockey": [43 players]     # Connor McDavid, Sidney Crosby, etc. 
    "football": [42 players]   # Tom Brady, Aaron Rodgers, etc.
    "baseball": [34 players]   # Mike Trout, Mookie Betts, etc.
    "soccer": [43 players]     # Lionel Messi, Cristiano Ronaldo, etc.
}
```

### 2.2 Actions Database  
Sport-specific action vocabulary:
```python
ACTIONS = {
    "basketball": [24 actions] # "beat the buzzer", "eurostepped to the basket"
    "hockey": [19 actions]     # "shot the puck", "crossed the blue line"
    "football": [18 actions]   # "threw a touchdown", "caught the screen pass"
    "baseball": [20 actions]   # "hit a walkoff homer", "struck out the side"
    "soccer": [24 actions]     # "scored a freekick", "nutmegged the defender"
}
```

### 2.3 Tournament Contexts
Championship/tournament references by sport:
```python
TOURNAMENTS = {
    "basketball": ["NBA Championship", "Eastern Conference Finals", ...]
    "hockey": ["Stanley Cup", "Stanley Cup Finals", ...]
    "football": ["Superbowl", "AFC Championship", ...]
    "baseball": ["World Series", "League Championship", ...]
    "soccer": ["Champions League", "FA Cup", ...]
}
```

## 3. Matrix Generation Strategy

### 3.1 Combinatorial Space Analysis
**Available combinations:**
- **Plausible:** Same-sport athlete-action pairs
  - Basketball: 32 × 24 = 768 combinations
  - Hockey: 43 × 19 = 817 combinations  
  - Football: 42 × 18 = 756 combinations
  - Baseball: 34 × 20 = 680 combinations
  - Soccer: 43 × 24 = 1,032 combinations
  - **Total plausible space:** 4,053 combinations

- **Implausible:** Cross-sport athlete-action pairs
  - Each sport's athletes × other sports' actions
  - **Total implausible space:** ~16,000 combinations

**Capacity verification:** 21× overcapacity available for target 950 samples

### 3.2 Systematic Generation Algorithm
```python
def generate_plausible_samples(target_count: int):
    # Same-sport athlete-action combinations
    for each target sample:
        sport = random.choice(['basketball', 'hockey', 'football', 'baseball', 'soccer'])
        athlete = random.choice(ATHLETES[sport])
        action = random.choice(ACTIONS[sport])
        
        # 20% chance to add tournament context
        if random.random() < 0.20:
            tournament = random.choice(TOURNAMENTS[sport])
            sentence = f"{athlete} {action} in the {tournament}."
        else:
            sentence = f"{athlete} {action}."

def generate_implausible_samples(target_count: int):
    # Cross-sport athlete-action combinations  
    for each target sample:
        athlete_sport = random.choice(sports_list)
        action_sport = random.choice([s for s in sports_list if s != athlete_sport])
        
        athlete = random.choice(ATHLETES[athlete_sport])
        action = random.choice(ACTIONS[action_sport])
        
        # 15% chance for wrong tournament context (extra confusion)
        if random.random() < 0.15:
            tournament_sport = random.choice([s for s in sports_list if s != athlete_sport])
            tournament = random.choice(TOURNAMENTS[tournament_sport])
            sentence = f"{athlete} {action} in the {tournament}."
```

## 4. Quality Control Measures

### 4.1 Duplicate Prevention
- Extract sentences from existing dataset using regex: `r'"([^"]+)"'`
- Maintain `existing_sentences` set for real-time duplicate checking
- Check lowercase normalized versions to catch case variations

### 4.2 Format Consistency
Ensure all generated samples match original format:
```
Input: 'Is the following sentence plausible? "SENTENCE"'
Target: "yes" or "no"
```

### 4.3 Difficulty Distribution
Applied complexity classification based on analysis:
- **Simple (78%):** Basic cross-sport mismatches, obvious equipment/location errors
- **Moderate (18%):** Technical terminology from wrong sport  
- **Complex (4%):** Multiple mismatch layers + tournament context

## 5. Implementation Details

### 5.1 Core Script Structure
File: `claude-scratchpad/generate_sports_samples.py`

**Key functions:**
- `load_existing_dataset()`: Parse JSON structure with "examples" key
- `extract_sentence_content()`: Extract sentences using regex
- `create_existing_sentences_set()`: Build duplicate checking set
- `generate_plausible_samples()`: Same-sport combinations
- `generate_implausible_samples()`: Cross-sport combinations  
- `apply_difficulty_distribution()`: Classify complexity levels

### 5.2 Random Seed Management
```python
random.seed(42)  # Reproducible generation
```

### 5.3 Balance Achievement
- **Target:** 475 plausible + 475 implausible = 950 total
- **Shuffle:** `random.shuffle(new_samples)` for random order
- **Validation:** Count target distribution in final output

## 6. Results Validation

### 6.1 Generated Output
**Files created:**
- `sports_understanding_new_samples.json` (950 new samples)
- `sports_understanding_new_full.json` (1200 total samples)

**Final distributions:**
- **New samples:** 475 yes (50.0%) + 475 no (50.0%) ✅
- **Combined dataset:** 590 yes (49.2%) + 610 no (50.8%) ✅
- **Complexity:** 741 simple (78%) + 172 moderate (18%) + 37 complex (4%) ✅

### 6.2 Quality Examples
**Plausible samples:**
- `"Is the following sentence plausible? "Aaron Jones gained five yards."` (football player, football action)

**Implausible samples:**  
- `"Is the following sentence plausible? "Trae Young threw a touchdown."` (basketball player, football action)

**Complex samples:**
- `"Is the following sentence plausible? "Sidney Crosby shot from the six yard line in the Western Conference Finals."` (hockey player, soccer action, basketball tournament)

## 7. Key Success Factors

### 7.1 Systematic Approach
- **Matrix generation** leveraged combinatorial overcapacity
- **Component databases** preserved authentic athlete-sport associations
- **Pattern-based generation** maintained quality characteristics from analysis

### 7.2 Quality Preservation
- **Sport boundaries** maintained clear separation between domains
- **Authentic terminology** used legitimate sport-specific vocabulary
- **Natural language** ensured grammatically correct, realistic sentences

### 7.3 Balance Achievement
- **Perfect 50/50** in new samples corrected original 54/46 skew
- **Near-perfect overall** balance in combined 1200-sample dataset
- **Proportional complexity** distribution matching original patterns

## 8. Reproducibility

### 8.1 Seed Control
Fixed random seed ensures reproducible results across runs.

### 8.2 Component Databases
All athlete, action, and tournament databases documented in script for future reference.

### 8.3 Validation Pipeline
Systematic checking process documented for verifying:
- Format consistency
- Duplicate prevention  
- Logical correctness
- Distribution targets

This methodology successfully scaled the Sports Understanding dataset by 4× while maintaining quality, authenticity, and achieving perfect label balance through systematic matrix generation.