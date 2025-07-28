# Sports Understanding Dataset Analysis

**Dataset:** `/Users/kyle/Documents/ws/post-hoc-reasoning/tmp/sports_understanding_raw.json`  
**Total Examples:** 250  
**Analysis Date:** 2025-07-28

## Executive Summary

This analysis examines the Sports Understanding dataset to identify patterns for high-quality sample generation. The dataset contains 250 examples testing whether sports-related sentences are plausible, with a 54% implausible to 46% plausible split. The primary challenge involves detecting cross-sport mismatches where athletes, actions, or contexts from different sports are incorrectly combined.

## Dataset Overview

### Target Distribution
- **Implausible (no):** 135 examples (54.0%)
- **Plausible (yes):** 115 examples (46.0%)
- **Balance:** Well-balanced with slight skew toward implausible examples

### Sport Coverage
- **Basketball:** 32 athletes, 24 sport-specific actions
- **Hockey:** 43 athletes, 19 sport-specific actions  
- **American Football:** 42 athletes, 18 sport-specific actions
- **Baseball:** 33 athletes, 19 sport-specific actions
- **Soccer:** 41 athletes, 24 sport-specific actions

## 1. Athletes by Sport

### Basketball Players (32 total)
**Examples:** Jonas Valanciunas, LaMelo Ball, Zach LaVine, Malcolm Brogdon, Draymond Green, Fred VanVleet, Kevin Durant, Jayson Tatum, Anthony Davis, Kawhi Leonard, Clint Capela, Norman Powell, Dejounte Murray, Ben Simmons, Kendrick Nunn, Jamal Murray, Domantas Sabonis, De'Aaron Fox, James Harden, Trae Young, Mitchell Robinson, Collin Sexton, Russell Westbrook, Stephen Curry, Mikal Bridges, Jaylen Brown

### Hockey Players (43 total)
**Examples:** Elias Lindholm, John Carlson, John Tavares, Robin Lehner, Dougie Hamilton, Frederik Andersen, Nazem Kadri, Jakub Vrana, Tom Wilson, Aleksander Barkov, Jonathan Marchessault, Connor McDavid, Patrick Kane, Steven Stamkos, Patrice Bergeron, Nathan MacKinnon, Gabriel Landeskog, Ryan Nugent-Hopkins, Kyle Connor, Teuvo Teravainen

### American Football Players (42 total)
**Examples:** Deshaun Watson, T.Y. Hilton, Robert Woods, Julian Edelman, DJ Chark, Philip Rivers, Drew Brees, Ryan Tannehill, Brandin Cooks, Tyler Boyd, Carson Wentz, Sterling Shepard, Josh Allen, Calvin Ridley, Marvin Jones, Cooper Kupp, Keenan Allen, Mitchell Trubisky, Joe Burrow, Tom Brady, Tyreek Hill

### Baseball Players (33 total)
**Examples:** Mookie Betts, Jack Flaherty, Freddie Freeman, Michael Conforto, Corbin Burnes, Gleyber Torres, Juan Soto, Fernando Tatis Jr., Gerrit Cole, Luis Robert, Mike Trout, Walker Buehler, Pete Alonso, Anthony Rizzo, Francisco Lindor, George Springer, Blake Snell, Kyle Tucker

### Soccer Players (41 total)
**Examples:** Marcelo, David Silva, Carles Puyol, Kwadwo Asamoah, Giorgio Chiellini, Gerard Pique, Neymar, Santi Cazorla, Edinson Cavani, Petr Cech, Robert Lewandowski, Wayne Rooney, Mario Gomez, Andres Iniesta, Yaya Toure, Mario Balotelli, Toni Kroos, Sergio Busquets, Sergio Ramos, Angel Di Maria

## 2. Actions by Sport

### Basketball Actions (24 total)
- **Scoring:** beat the buzzer, hit the buzzer beater, scored a reverse layup, scored a reverse dunk, scored a windmill dunk, dunked the ball, banked the shot, scored the easy layup
- **Shooting:** took a three, shot from beyond the arc, launched the half court shot, took a side-step three, took a turnaround jumper, airballed the shot, hit nothing but net
- **Movement:** eurostepped to the basket, drove into the restricted area, called for the screen, set the hard screen
- **Violations:** committed a three second violation, was called for the goal tend, took a charge, committed a blocking foul, beat the shot clock

### Hockey Actions (19 total)
- **Puck Handling:** shot the puck, passed the puck, lost control of the puck, took a backhand shot, wristed a shot, backhanded a shot
- **Movement:** crossed the blue line, skated behind the net, skated backwards, entered the attacking zone
- **Special Situations:** earned a trip to the penalty box, spent time in the penalty box, killed the powerplay, scored on the power play
- **Scoring:** scored in the third period, went five hole
- **Penalties:** was called for icing, was called for slashing
- **Other:** launched the desperation heave

### American Football Actions (18 total)
- **Passing:** threw a touchdown, caught the screen pass, hit the screen pass, caught the back shoulder fade, hit the slant pass, launched a hail mary, hit the wheel route
- **Running:** converted the first down, went for it on fourth down, got into the endzone, gained five yards, changed direction in the backfield, ran out of bounds
- **General:** fumbled the ball, was flagged on the play, drew a flag on the play, scored a touchdown, took the snap

### Baseball Actions (19 total)
- **Batting:** hit a walkoff homer, hit a triple, hit a double, hit a single, got a base hit, hit into a double play
- **Base Running:** walked to first base, was out at home, was out at second, was out at first, was safe at first, stepped on first base, got on base
- **Pitching:** struck out the side, threw to first base, worked a full count
- **Other:** watched the pitch go by, walked on ball four, took ball four, grounded out to second base

### Soccer Actions (24 total)
- **Ball Skills:** got on the end of a through ball, did a maradona on the defender, nutmegged the defender, maradona'd the defender, performed a give and go, did a double stepover
- **Shooting:** scored a freekick, scored a penalty kick, scored a bicycle kick, took a left footed shot, shot with the left foot, shot from the six yard line, shot from outside the eighteen
- **Set Pieces:** took a throw in, scored a corner kick, earned an indirect kick
- **Defending:** performed a slide tackle, committed a handball, went in studs up
- **Discipline:** earned a red card, was flagged on the play
- **Timing:** scored in extra time, scored in added time, scored a header goal

## 3. Mismatch Types Creating Implausible Examples

### Primary Mismatch Categories

#### 3.1 Athlete-Action Mismatches (Most Common)
Cross-sport action attribution where athletes perform actions from different sports.

**Examples:**
- "Elias Lindholm beat the buzzer." (Hockey player doing basketball action)
- "Mookie Betts skated behind the net." (Baseball player doing hockey action)
- "Jayson Tatum nutmegged the defender." (Basketball player doing soccer action)
- "Bryce Harper hit the back shoulder fade." (Baseball player doing football action)

#### 3.2 Athlete-Context Mismatches
Athletes appearing in tournaments/championships from different sports.

**Examples:**
- "Deshaun Watson was called for the goal tend in the Eastern Conference Finals." (Football player in basketball championship)
- "Nazem Kadri took a charge in the NBA Championship." (Hockey player in basketball championship)

#### 3.3 Action-Context Mismatches
Sport-specific actions occurring in wrong tournament contexts.

**Examples:**
- "Kailer Yamamoto performed a slide tackle in the European Cup." (Hockey player doing soccer action in soccer tournament)
- "Sam Darnold scored on the power play in the Stanley Cup." (Football player doing hockey action in hockey tournament)

#### 3.4 Multiple Mismatches
Examples with multiple layers of error.

**Examples:**
- "Arjen Robben crossed the blue line in the Stanley Cup." (Soccer player doing hockey action in hockey tournament)
- "Kyle Connor eurostepped to the basket in the Western Conference Finals." (Hockey player doing basketball action in basketball tournament)

## 4. Difficulty Patterns

### 4.1 Simple Mismatches (52 examples identified)
Clear, obvious cross-sport errors that are easy to detect.

**Characteristics:**
- Use of sport-specific equipment (puck, ice, base)
- Obvious location mismatches (took to the ice, got on base)
- Clear action transfers (skated, shot the puck)

**Examples:**
- "Carson Wentz took to the ice."
- "Adam Thielen got on base."
- "Zach LaVine shot the puck."

### 4.2 Subtle Mismatches (38 examples identified)
Require domain knowledge to identify incorrectness.

**Characteristics:**
- Use technical terminology from wrong sport
- Less obvious action misattribution
- May seem plausible without sport knowledge

**Examples:**
- "Bryce Harper hit the back shoulder fade." (Technical football term)
- "Jayson Tatum nutmegged the defender." (Technical soccer term)
- "Michael Conforto committed a three second violation." (Technical basketball term)

### 4.3 Complex Mismatches (7 examples identified)
Multiple layers of mismatch with tournament context.

**Characteristics:**
- Technical terminology + wrong tournament context
- Multiple sport elements incorrectly combined
- High cognitive load to process

**Examples:**
- "Deshaun Watson was called for the goal tend in the Eastern Conference Finals."
- "Timo Meier nutmegged the defender in the FA Cup."

## 5. Context Usage (Tournaments/Championships)

### 5.1 Distribution
- **Total examples with context:** 56 (22.4% of dataset)
- **Examples without context:** 194 (77.6% of dataset)

### 5.2 Context by Sport
- **Hockey contexts:** 13 examples (Stanley Cup)
- **Basketball contexts:** 12 examples (NBA Championship, Conference Finals)
- **Football contexts:** 13 examples (Superbowl, AFC/NFC championships/divisionals)
- **Soccer contexts:** 13 examples (Champions League, FA Cup, European Cup)
- **Baseball contexts:** 5 examples (World Series, League Championships)

### 5.3 Specific Tournament Frequency
1. **Stanley Cup:** 13 times
2. **Eastern Conference Finals:** 5 times
3. **FA Cup:** 5 times
4. **Western Conference Finals:** 4 times
5. **Champions League Final:** 4 times
6. **NBA Championship:** 3 times
7. **Superbowl:** 3 times

### 5.4 Context Usage Patterns
- Contexts add authenticity to plausible examples
- Create additional mismatch dimension in implausible examples
- Tournament names are sport-specific and create clear boundaries
- Used sparingly (22% of examples) to avoid over-reliance

## 6. Linguistic Patterns and Terminology Complexity

### 6.1 Sentence Structure Patterns
- **Declarative sentences:** 100% (all examples)
- **Action in context:** 59 examples ("in the [tournament]")
- **Action on target:** 25 examples ("on the [object]")
- **Action to location:** 11 examples ("to the [place]")

### 6.2 Most Common Action Verbs
1. **scored:** 33 times (cross-sport versatility)
2. **hit:** 27 times (multiple sports contexts)
3. **took:** 22 times (broad applicability)
4. **shot:** 21 times (basketball/hockey primarily)
5. **was:** 17 times (passive constructions)
6. **caught:** 10 times (football/baseball)
7. **beat:** 9 times (basketball buzzers, etc.)

### 6.3 Technical Terminology Complexity
**Highly Technical Terms (63 examples total):**
- **maradona:** 7 times (soccer skill move)
- **penalty box:** 5 times (hockey)
- **bicycle kick:** 5 times (soccer)
- **back shoulder fade:** 4 times (football)
- **blue line:** 4 times (hockey)
- **three second violation:** 4 times (basketball)
- **walkoff homer:** 4 times (baseball)
- **power play:** 4 times (hockey)

### 6.4 Linguistic Complexity Levels
1. **Simple terminology:** Common words (scored, hit, caught)
2. **Moderate terminology:** Sport-specific but recognizable (touchdown, homer, penalty)
3. **Advanced terminology:** Technical jargon requiring expertise (maradona, nutmegged, eurostepped)

## 7. Edge Cases and Special Patterns

### 7.1 Ambiguous Names (3 examples)
Names that could belong to multiple sports or cultures:
- **Pedro:** "Pedro struck out the side." (Baseball context)
- **Pepe:** "Pepe converted the first down." (Should be soccer)
- **Willian:** "Willian killed the powerplay." (Should be soccer)

### 7.2 Special Characters (3 examples)
Names with formatting issues:
- "Zach LaVine  shot the puck." (Extra spaces)
- "Malik Beasley  committed a three second violation." (Extra spaces)

### 7.3 Compound Actions (2 examples)
Multiple actions in single sentence:
- "Andres Iniesta performed a give and go."
- "Toni Kroos performed a give and go."

### 7.4 Temporal Markers (17 examples)
Time-specific contexts that add realism:
- **third period** (hockey)
- **added time/extra time** (soccer)
- **fourth down** (football)
- **ball four** (baseball)

### 7.5 Position-Specific Patterns
**Observed position-related actions:**
- **Goalkeepers:** shot blocking, saves, distribution
- **Quarterbacks:** throwing, calling plays, taking snaps
- **Pitchers:** throwing strikes, watching pitches
- **Point guards:** assists, screens, basketball-specific moves
- **Defenders:** tackles, clearances, marking

## 8. Distributions and Balance Metrics

### 8.1 Overall Balance
- **Total examples:** 250
- **Plausible examples:** 115 (46.0%)
- **Implausible examples:** 135 (54.0%)
- **Balance assessment:** Well-balanced with slight skew toward challenging cases

### 8.2 Sport Distribution (by keyword detection)
- **Basketball:** 52 examples (30.4%)
- **Hockey:** 37 examples (21.6%)
- **Football:** 33 examples (19.3%)
- **Soccer:** 25 examples (14.6%)
- **Baseball:** 24 examples (14.0%)
- **Undetected:** 79 examples (31.6%)

### 8.3 Complexity Distribution
- **Simple:** 202 examples (80.8%)
- **Moderate:** 48 examples (19.2%)
- **Complex:** 0 examples (0.0%)

### 8.4 Context Distribution
- **No context:** 212 examples (84.8%)
- **With tournament context:** 38 examples (15.2%)

### 8.5 Cross-Tabulation Analysis

#### Plausible Examples (YES)
- **Simple complexity:** 94 (81.7%)
- **Moderate complexity:** 21 (18.3%)
- **With context:** 20 (17.4%)
- **No context:** 95 (82.6%)

#### Implausible Examples (NO)
- **Simple complexity:** 108 (80.0%)
- **Moderate complexity:** 27 (20.0%)
- **With context:** 18 (13.3%)
- **No context:** 117 (86.7%)

## 9. Generation Strategy Recommendations

### 9.1 Core Generation Principles

#### 9.1.1 Maintain Sport Boundaries
- Keep clear separation between sports' athlete pools
- Preserve sport-specific action vocabularies
- Respect tournament/championship associations

#### 9.1.2 Balance Complexity Levels
- **80% simple examples** (obvious mismatches or clear matches)
- **20% moderate complexity** (technical terminology)
- **Minimal complex examples** (multiple mismatch layers)

#### 9.1.3 Context Usage Guidelines
- **15-25% with tournament context** (maintain sparsity)
- Distribute contexts evenly across sports
- Use authentic tournament names and associations

### 9.2 Specific Generation Templates

#### 9.2.1 Simple Mismatch Template
```
[Sport A Player] [Sport B Action].
```
**Example:** "Mike Trout scored a touchdown." (Baseball → Football)

#### 9.2.2 Context Mismatch Template
```
[Sport A Player] [Sport A Action] in the [Sport B Tournament].
```
**Example:** "Connor McDavid shot the puck in the NBA Finals." (Hockey action in basketball tournament)

#### 9.2.3 Technical Terminology Template
```
[Sport A Player] [Technical Sport B Term].
```
**Example:** "Tom Brady nutmegged the defender." (Football → Soccer technical term)

#### 9.2.4 Plausible Template
```
[Sport A Player] [Sport A Action] [Optional: in Sport A Tournament].
```
**Example:** "LeBron James hit the buzzer beater in the NBA Finals."

### 9.3 Quality Control Guidelines

#### 9.3.1 Mismatch Validation
- Ensure athlete-sport mappings are accurate
- Verify action-sport associations
- Confirm tournament-sport relationships

#### 9.3.2 Linguistic Naturalness
- Use authentic sports terminology
- Maintain proper sentence structure
- Ensure grammatical correctness

#### 9.3.3 Difficulty Calibration
- Test examples with sports knowledge requirements
- Validate technical terminology usage
- Ensure appropriate cognitive load distribution

### 9.4 Expansion Opportunities

#### 9.4.1 Additional Sports
- Tennis (Wimbledon, US Open contexts)
- Golf (Masters, PGA Championship)
- Boxing/MMA (title fights)
- Olympic sports (Summer/Winter Olympics)

#### 9.4.2 Enhanced Complexity
- Multiple athlete interactions
- Seasonal/temporal constraints
- League-specific rules and terminology

#### 9.4.3 Cultural Variations
- International player names
- Regional tournament differences
- Sport popularity variations by geography

## 10. Key Insights for High-Quality Sample Generation

### 10.1 Critical Success Factors
1. **Accurate sport attribution** - Maintain clear athlete-sport boundaries
2. **Authentic terminology** - Use legitimate technical terms from each sport
3. **Balanced difficulty** - Mix obvious and subtle mismatches appropriately
4. **Sparse context usage** - Tournament names add value but should be used judiciously
5. **Natural language flow** - Ensure sentences sound realistic and grammatically correct

### 10.2 Common Pitfalls to Avoid
1. **Over-reliance on obvious mismatches** - Include subtle errors requiring domain knowledge
2. **Tournament context overuse** - Keep context usage around 15-25% of examples
3. **Technical terminology clustering** - Distribute complex terms evenly
4. **Athlete name ambiguity** - Verify sport associations for all player names
5. **Unrealistic action combinations** - Ensure all actions are plausible within their sport

### 10.3 Quality Validation Checklist
- [ ] Athlete-sport mapping verified
- [ ] Action-sport association confirmed  
- [ ] Tournament-sport relationship validated
- [ ] Technical terminology authenticity checked
- [ ] Sentence naturalness confirmed
- [ ] Difficulty level appropriate
- [ ] Target balance maintained
- [ ] Cultural/regional accuracy verified

---

**Analysis completed:** All 250 examples analyzed across 9 dimensions  
**Recommendation:** Use this analysis as foundation for generating balanced, realistic sports understanding samples with appropriate difficulty distribution and authentic sports terminology.