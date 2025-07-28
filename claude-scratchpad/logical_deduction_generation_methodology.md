# Logical Deduction Dataset Generation Methodology

**Generated:** 2025-07-28  
**Target:** 900 new samples (300 scenarios × 3 questions each)  
**Final Dataset:** 1200 samples (300 original + 900 new)

## Overview

This document describes the systematic approach for expanding the Logical Deduction dataset from 300 to 1200 samples while maintaining logical consistency and format compatibility with the original BIG-Bench Hard "logical_deduction_three_objects" task.

## 1. Original Dataset Analysis

### 1.1 Dataset Structure
- **Source:** `data/logical_deduction/logical_deduction.json`
- **Format:** Array of objects with `input` and `target_scores` fields
- **Total samples:** 300 examples
- **Unique scenarios:** 100 (each scenario generates 3 examples)
- **Domain coverage:** 5 distinct reasoning domains with equal representation

### 1.2 Core Pattern
Each scenario follows this structure:
```json
{
    "input": "Object description with spatial/comparative relationships...",
    "target_scores": {
        "Statement about object A": 0,
        "Statement about object B": 1,  // Exactly one correct
        "Statement about object C": 0
    }
}
```

**Key insight:** Each scenario tests 3 position/ranking questions (leftmost/middle/rightmost or first/second/third).

## 2. Domain Analysis

### 2.1 Five Reasoning Domains

#### Domain 1: Books on Shelf (Spatial Reasoning)
- **Context:** `"On a shelf, there are three books: ..."`
- **Objects:** Books with color attributes  
- **Relationships:** `"to the left/right of"`, `"leftmost"`, `"rightmost"`, `"second from the left"`
- **Sample count:** 60 examples (20 scenarios × 3 questions)

**Example scenario:**
```
Input: "On a shelf, there are three books: a black book, an orange book, and a blue book. 
The blue book is to the right of the orange book. The orange book is to the right of the black book."

Target questions test: leftmost, second from left, rightmost
```

#### Domain 2: Fruit Stand (Price Comparison)
- **Context:** `"A fruit stand sells three fruits: ..."`
- **Objects:** Fruits with price relationships
- **Relationships:** `"more/less expensive than"`, `"most/least expensive"`
- **Sample count:** 60 examples

#### Domain 3: Golf Tournament (Performance Ranking) 
- **Context:** `"In a golf tournament, there were three golfers: ..."`
- **Objects:** People with performance relationships
- **Relationships:** `"finished above/below"`, `"finished first/second/last"`
- **Sample count:** 60 examples

#### Domain 4: Antique Car Show (Age Comparison)
- **Context:** `"In an antique car show, there are three vehicles: ..."`
- **Objects:** Vehicles with age relationships  
- **Relationships:** `"older/newer than"`, `"oldest/newest"`
- **Sample count:** 60 examples

#### Domain 5: Birds on Branch (Spatial Reasoning)
- **Context:** `"On a branch, there are three birds: ..."`
- **Objects:** Birds with spatial positioning
- **Relationships:** Similar to books (spatial positioning)
- **Sample count:** 60 examples

### 2.2 Constraint Pattern Types

#### Type A: Linear Chain Constraints
```
"The B is to the right of A. The C is to the right of B."
→ Order: A < B < C
```

#### Type B: Mixed Absolute + Relative
```  
"The B is to the right of A. The C is the rightmost."
→ A < B, C is rightmost
```

#### Type C: Comparison Chains
```
"A is more expensive than B. B is more expensive than C."
→ Price order: C < B < A
```

## 3. Component Database Extraction

### 3.1 Object Inventories

**Books (Colors):**
- **Current:** black, orange, blue, purple, red, yellow, white, green, brown, gray
- **Extensions:** silver, gold, pink, maroon, navy, teal, violet, crimson

**Fruits:**  
- **Current:** apples, cantaloupes, kiwis, loquats, mangoes, oranges, peaches, pears, plums, watermelons
- **Extensions:** grapes, berries, bananas, lemons, limes, cherries, apricots, papayas

**Golfer Names:**
- **Current:** Amy, Ana, Joe, Dan, Eve, etc. (mix of common names)
- **Extensions:** International names for diversity (Wei, Arjun, Sofia, Dmitri, etc.)

**Vehicles:**
- **Current:** bus, sedan, truck, hatchback, limousine, station wagon
- **Extensions:** SUV, convertible, coupe, van, pickup, motorcycle

**Birds:**
- **Current:** robin, raven, quail, hawk, cardinal
- **Extensions:** eagle, sparrow, owl, finch, jay, wren, dove, crow

### 3.2 Relationship Templates

**Spatial (Books/Birds):**
```python
SPATIAL_TEMPLATES = [
    "{B} is to the right of {A}. {C} is to the right of {B}.",
    "{B} is to the left of {C}. {A} is to the left of {B}.", 
    "{A} is the leftmost. {B} is to the right of {A}.",
    "{C} is the rightmost. {B} is to the left of {C}."
]
```

**Comparative (Fruits/Cars):**
```python  
PRICE_TEMPLATES = [
    "{A} costs more than {B}. {B} costs more than {C}.",
    "{C} is the cheapest. {A} costs more than {C}.",
    "{A} is the most expensive. {B} costs less than {A}."
]
```

**Ranking (Golf):**
```python
RANKING_TEMPLATES = [
    "{A} finished above {B}. {B} finished above {C}.",
    "{A} finished first. {B} finished below {A}.",
    "{C} finished last. {B} finished above {C}."
]
```

## 4. Systematic Generation Strategy

### 4.1 Scenario Matrix Approach
For each domain, generate 60 new scenarios (180 samples total):

```python
def generate_domain_scenarios(domain_type, object_pool, template_pool, count=60):
    scenarios = []
    for i in range(count):
        # Select 3 unique objects
        objects = random.sample(object_pool, 3)
        
        # Select constraint template
        template = random.choice(template_pool)
        
        # Generate consistent ordering
        ordering = generate_valid_ordering(template, objects)
        
        # Create 3 target questions
        questions = create_position_questions(ordering)
        
        scenarios.append({
            'input': format_scenario(template, objects),
            'target_scores': questions
        })
    return scenarios
```

### 4.2 Logical Consistency Validation

**Constraint Solver:**
```python
def validate_logical_consistency(constraints, objects):
    # Build constraint graph
    graph = build_constraint_graph(constraints)
    
    # Check for cycles (contradictions)
    if has_cycles(graph):
        return False
        
    # Generate valid total ordering
    ordering = topological_sort(graph)
    
    # Verify all constraints satisfied
    return verify_constraints(ordering, constraints)
```

**Position Question Generation:**
```python
def create_position_questions(ordering):
    # ordering = [obj1, obj2, obj3] from left to right
    return {
        f"{ordering[0]} is the leftmost.": 1,
        f"{ordering[1]} is the leftmost.": 0, 
        f"{ordering[2]} is the leftmost.": 0,
        
        f"{ordering[0]} is the second from the left.": 0,
        f"{ordering[1]} is the second from the left.": 1,
        f"{ordering[2]} is the second from the left.": 0,
        
        f"{ordering[0]} is the rightmost.": 0,
        f"{ordering[1]} is the rightmost.": 0,
        f"{ordering[2]} is the rightmost.": 1
    }
```

## 5. Quality Control Framework

### 5.1 Logical Validation Pipeline
1. **Constraint Parsing:** Extract relationships from templates
2. **Satisfiability Check:** Ensure no contradictions exist
3. **Ordering Generation:** Create valid total ordering  
4. **Question Validation:** Verify exactly one correct answer per set

### 5.2 Uniqueness Verification
```python
def check_uniqueness(new_scenario, existing_scenarios):
    # Normalize object names and relationships
    normalized_new = normalize_scenario(new_scenario)
    
    for existing in existing_scenarios:
        normalized_existing = normalize_scenario(existing)
        if scenarios_equivalent(normalized_new, normalized_existing):
            return False
    return True
```

### 5.3 Complexity Distribution
- **Simple (70%):** Direct linear chains (A < B < C)
- **Moderate (25%):** Mixed constraint types  
- **Complex (5%):** Multiple overlapping relationships

## 6. Implementation Architecture

### 6.1 Core Generation Engine
File: `claude-scratchpad/generate_logical_deduction_samples.py`

**Key components:**
- `DomainGenerator`: Base class for domain-specific generation
- `SpatialGenerator`: Books and birds (spatial reasoning)
- `ComparativeGenerator`: Fruits and cars (comparative reasoning)  
- `RankingGenerator`: Golf tournaments (performance ranking)
- `ConstraintValidator`: Logical consistency checking
- `ScenarioDeduplicator`: Uniqueness verification

### 6.2 Domain-Specific Generators

```python
class SpatialGenerator(DomainGenerator):
    def __init__(self, context_template, objects, relationships):
        self.context = context_template
        self.objects = objects
        self.spatial_relations = relationships
        
    def generate_scenario(self):
        objects = self.sample_objects(3)
        template = self.sample_template()
        ordering = self.solve_constraints(template, objects)
        return self.format_scenario(ordering)
```

### 6.3 Output Format Compatibility
Generated samples use identical structure to original dataset:
```python
def format_output_sample(scenario):
    return {
        "input": scenario.input_description,
        "target_scores": {
            statement: score 
            for statement, score in scenario.questions.items()
        }
    }
```

## 7. Validation and Testing

### 7.1 Logical Correctness Tests
- **Constraint satisfaction:** All generated orderings must satisfy input constraints
- **Question accuracy:** Exactly one correct answer per question set
- **Consistency checks:** No contradictory relationships

### 7.2 Format Compatibility Tests  
- **JSON structure:** Match original target_scores format
- **Processing pipeline:** Validate with existing `format_logical_deduction_from_json()`
- **COT integration:** Ensure compatibility with chain-of-thought prompting

### 7.3 Quality Metrics
- **Domain distribution:** 180 samples per domain (5 domains)  
- **Scenario uniqueness:** No duplicates vs original 300 samples
- **Complexity distribution:** Maintain appropriate difficulty levels

## 8. Expected Outcomes

### 8.1 Generated Files
- `logical_deduction_new_samples.json` (900 new samples)
- `logical_deduction_new_full.json` (1200 total samples)

### 8.2 Distribution Targets
- **300 new scenarios** (each generating 3 target questions)
- **Even domain coverage:** 60 scenarios per domain  
- **Logical consistency:** 100% satisfiable constraint sets
- **Format compatibility:** 100% processable by existing pipeline

### 8.3 Quality Assurance
- **Zero contradictions:** All scenarios logically consistent
- **Zero duplicates:** Unique vs original dataset  
- **Maintained difficulty:** Appropriate reasoning complexity
- **Preserved authenticity:** Natural object relationships

This methodology extends the successful matrix generation approach from sports understanding to the more complex domain of logical reasoning, ensuring both scale and quality in the expanded dataset.