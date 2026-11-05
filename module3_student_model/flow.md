# Module 3: Demo and Engine Flow

This document follows the calls made by `demo.py` and shows each step's input, operation/formula, and output. The simulator holds the hidden student truth; `StudentModelEngine` sees only emitted `Event` objects.

## 1. Demo setup and loop

### 1.1 Build the course and classroom

**Call:** `astar_catalog()` → `make_classroom(catalog, scenario, n_students, seed)` → `ClassroomSimulator(catalog, profiles, seed)` → `StudentModelEngine(catalog, sim.student_ids())`

| Input | Operation | Output |
|---|---|---|
| Scenario, student count, seed | Build course concepts/misconceptions; generate student profiles and initialize simulator truth; initialize estimator trackers for registered student IDs. | A course catalog, simulated students, simulator, and engine. The engine is not given simulator truth. |

The demo defaults to 24 students, seed 1, and scenario `mixed`. Its action sequence is:

`diagnostic quiz → lecture → example → diagnostic quiz → peer discussion → exercise → diagnostic quiz`

The demo loops over every concept and every action:

`events = sim.step(action, concept_id)`  
`engine.update_many(events)`

**Input:** one teaching action and concept.  
**Output:** zero or more events from the simulator; each event is processed in order by the engine. A `NEXT_CONCEPT` action returns no events.

## 2. Simulator: action to observable event (`simulator.py`)

### 2.1 Initial hidden state

For student `s`, concept `c`:

- Draw true mastery: `known[s,c] = (random < profile.prior_knowledge[c])`.
- If the student has a misconception linked to `c`, force `known[s,c] = False`.
- Initialize the hidden misconception set and engagement from the profile.

**Input:** profiles, catalog, random seed.  
**Output:** simulator-only `TrueState` for each student. These truth values are not event fields.

### 2.2 Learning opportunity

For a student not already knowing the concept and holding no linked misconception:

`prereq_factor = 1` if there are no prerequisites; otherwise `0.3 + 0.7 × (known prerequisites / number of prerequisites)`

`p_learn_true = min(0.95, action_rate × learning_speed × (0.4 + 0.6 × engagement) × prereq_factor)`

A random draw below `p_learn_true` changes the simulator's hidden mastery to known. A linked misconception blocks this learning attempt.

Action rates: lecture `0.12`, example `0.18`, peer discussion `0.15`, peer learning `0.20`, TA intervention `0.30`, revision `0.20`, diagnostic quiz/assessment `0.03`, exercise `0.12`.

**Input:** hidden state, action, student profile.  
**Output:** possibly updated hidden mastery; not directly exposed to the engine.

### 2.3 Teaching action (`_teach`)

**Input:** student, concept, teaching action, optional targeted misconception.

1. Try the hidden learning transition above.
2. For each linked misconception, remove it with probability:
   - Targeted: `0.5 × (0.4 + 0.6 × engagement)`
   - General: `0.06 × (0.4 + 0.6 × engagement)`
3. Emit an `INSTRUCTION` event.
4. With probability `participation_tendency × engagement × 0.6`, emit a voluntary `QUESTION` or `DISCUSSION` event too.

**Output:** instruction event and possibly a contribution event.

### 2.4 Answer action (`_answer`)

For quiz, assessment, or exercise:

- No response with probability `(1 - engagement) × 0.5`; emit `NO_RESPONSE`.
- Otherwise, if a tested misconception is held, its wrong option is selected with probability `0.7`.
- If not expressing a misconception:
  - Known student: correct with probability `1 - profile.slip`.
  - Unknown student: correct with probability `profile.guess`; an unknown student doing an exercise may receive a hint with probability `0.5`, making the guess probability `0.5`.
- If wrong, select a distractor and record its linked misconception ID if applicable.
- Emit `QUIZ_ANSWER` or `EXERCISE`, including correctness, selected/available misconception IDs, distractor count, hint flag, and stated confidence.
- After generating an answer, the simulator may learn from the practice action using the action's practice rate and the same learning formula.

Stated confidence:

`clip(profile.confidence + (0.1 if known else -0.05) + Normal(0, 0.08), 0, 1)`

**Output:** answer event, or no-response event. The event describes the observation, not the true knowledge state.

### 2.5 Contribution and engagement

A contribution's intent is generated from hidden state: 15% unclear; otherwise a held misconception can signal with probability 0.6; otherwise confusion if not known, or clarification/example request if known. Confidence is sampled by the simulator. The event carries intent, NLU-like confidence, signals and example text.

After each action, simulator engagement changes:

`engagement' = clip(engagement + action_effect + 0.1 × (profile_engagement - (engagement + action_effect)) + Normal(0, 0.02), 0.05, 1)`

**Output:** updated simulator-only engagement for the next action.

## 3. Event processing (`engine.py`)

### 3.1 Common prelude

**Call:** `engine.update(event)`; `update_many(events)` calls `update` once per event.

**Input:** one event with student ID, concept ID, event type and optional event-specific evidence.  
**Operation:** reject unknown concepts, register the student, then call `BehaviourTracker.update(event)`. Route by event type.  
**Output:** tracker state is updated in place; no estimate is returned by `update`.

### 3.2 Instruction

`strength = ACTION_LEARNING_WEIGHT[event.action]` (or configured default if action is missing).

**Calls:** `EnsembleTracker.expose(student, concept, strength)` and `MisconceptionTracker.apply_instruction(...)`.

- Each BKT member: `p' = p + (1-p) × (p_learn × strength)`.
- Each misconception: `P(M)' = clip(P(M) × (1 - rate × strength))`; `rate` is targeted or general.

**Output:** mastery and misconception beliefs move according to instruction; observation count does not increase.

### 3.3 Quiz or exercise

An event must contain `correct` or `score`. Before mastery is updated, the engine calls `MisconceptionTracker.update_from_answer` using the ensemble's current mean mastery. A hinted event uses `p_guess_hint`; otherwise it uses `p_guess`.

Then `EnsembleTracker.observe` updates every BKT member using the answer/score and increments the event observation count once.

**Output:** updated misconception beliefs, ensemble mastery values, and observation count.

### 3.4 Question or discussion

Let `c = event.nlu_confidence` (default `0.5`). If intent is `UNCLEAR`, misconception-signal confidence is halved: `signal_c = c × 0.5`; otherwise `signal_c = c`. Each valid misconception signal updates that misconception belief.

For recognized intent, the engine sends a soft score to the BKT ensemble:

`score = 0.5 + Δ × c`

`Δ`: confusion `-0.3`, misconception signal `-0.2`, clarification `+0.1`, example request `0`. `UNCLEAR` has no mastery update.

**Output:** confidence-weighted misconception and/or mastery evidence. A soft-score BKT observation increments the observation count.

### 3.5 Hint and no response

- `HINT`: behavior tracker only. Mastery changes only when a later answer has `hint_used=True`.
- `NO_RESPONSE`: behavior tracker only; it is not treated as a wrong answer.

**Output:** no mastery or misconception update from these event types alone.

## 4. Mastery and uncertainty (`bkt.py`, `uncertainty.py`)

### 4.1 Sample an ensemble

`EngineConfig.ensemble_size` defaults to `15`. Each member gets a plausible parameter sample:

`x ~ Beta(μκ, (1-μ)κ)`

The configured base parameter is `μ`; `κ=4` for `p_init`, `κ=30` for other parameters. Guess/slip samples are redrawn until `G + S < 0.95`; sampled hint guess is at least the sampled ordinary guess. A seeded generator makes the ensemble reproducible.

**Input:** base BKT parameters and seed.  
**Output:** 15 BKT trackers, each with its own parameters and per-student/concept mastery probability.

### 4.2 Binary answer update in each BKT member

Let `p = P(known)` before the answer. For a correct answer:

`p_obs = p(1-S) / [p(1-S) + (1-p)G]`

For a wrong answer:

`p_obs = pS / [pS + (1-p)(1-G)]`

For a hinted response use `G = G_h`. For a score `s`:

`L_known = s(1-S) + (1-s)S`  
`L_unknown = sG + (1-s)(1-G)`  
`p_obs = p L_known / [p L_known + (1-p)L_unknown]`

After the observation:

`p_next = clip(p_obs + (1-p_obs)T, 10^-6, 1-10^-6)`

All probabilities are clipped away from exactly 0 and 1.

**Input:** current member belief, correctness or score, hint flag, member parameters.  
**Output:** member's new mastery probability.

### 4.3 Helpers not called by the demo/engine path

- `prior_from_prerequisites(base_prior, prerequisite_mastery, weight)` returns `clip(base_prior × ((1-weight) + weight × mean(prerequisite_mastery)))`; it is optional and is not automatically applied by `StudentModelEngine`.
- `sequence_log_likelihood(sequence, params)` computes `Σ log(P(observed answer_t | current p_t, G, S))`, updating `p_t` after each answer. `fit_params_grid` maximizes the summed sequence log-likelihood over candidate parameter grids. These are parameter-fitting utilities, not calls in the demo or live event-update path.
- `binary_entropy(p) = -p log₂(p) - (1-p)log₂(1-p)` is an available helper; the engine's reported uncertainty is ensemble standard deviation, not entropy.

### 4.4 Ensemble readout

For member beliefs `p_1, ..., p_K` (`K=15` by default):

`mastery = mean(p_k) = (Σ p_k)/K`

`uncertainty = population_std(p_k) = sqrt(Σ(p_k - mastery)^2 / K)`

`n_observations = number of calls to observe`; instruction-only `expose` does not count.

**Output:** one mastery estimate, ensemble spread, and observation count per student/concept. Spread reflects member disagreement; it is not a calibrated probability of error.

## 5. Misconception probabilities (`misconceptions.py`)

Each student/misconception pair starts at configured prior `r` (default `0.15`). The Bayesian update is:

`r' = r L_held / [r L_held + (1-r)L_not]`

For answer evidence, using pre-answer mastery:

`q = p_mastery(1-S) + (1-p_mastery)G`; `w = 1-q`; `e = express` (default `0.7`); `nd = max(n_distractors, 1)`.

| Evidence for misconception M | `L_held` | `L_not` |
|---|---:|---:|
| Correct | `(1-e)q` | `q` |
| Select M's wrong option | `e + (1-e)w/nd` | `w/nd` |
| Generic wrong option (no misconception ID) | `(1-e)w/nd` | `w/nd` |
| Select a different misconception's option | No update to M | No update to M |

Only misconceptions available in that question are tested. NLU signal confidence `c` uses:

`L_held = c × signal_if_held + (1-c)`  
`L_not = c × signal_if_not_held + (1-c)`

Defaults are `signal_if_held=0.6` and `signal_if_not_held=0.1`; at confidence zero the probability is unchanged.

**Input:** answer choice or NLU signal, prior misconception belief and evidence parameters.  
**Output:** updated `P(M)` plus counts of linked-option picks, times tested, and NLU signals.

## 6. Behaviour estimates (`behaviour.py`)

Student behavior starts with engagement `0.7`, confidence `0.5`, zero counts, and calibration EMA `0`.

### Engagement

`engagement' = (1-r) × engagement + r × signal`, with `r=0.2`.

Signals: no response `0`; quiz/exercise answer `0.7`; hint `0.8`; question/discussion `1`. Instruction increments an opportunity but does not nudge engagement.

### Participation

`participation = min(1, (contributions + a)/(opportunities + a + b))`

Default Beta prior: `a=b=1`. Instruction is an opportunity; voluntary question/discussion is a contribution.

### Confidence and calibration

`confidence' = (1-r_c) × confidence + r_c × stated_confidence`, with `r_c=0.3`.

`gap = stated_confidence - outcome`; outcome is score if present, otherwise `1` for correct and `0` for wrong.

`w_n = max(1/n, 0.25)`  
`calibration' = (1-w_n) × calibration + w_n × gap`

**Output:** engagement, participation, confidence, and overconfidence (calibration EMA). `confidently_wrong=True` when at least 3 outcomes exist and calibration EMA > `0.25`.

## 7. Individual estimates (`engine.py`)

**Call:** `engine.get_student(student_id)`.

For every catalog concept, `_concept_estimate` reads:

- `p_mastery = EnsembleTracker.mean(...)`
- `uncertainty = EnsembleTracker.std(...)`
- `n_observations = EnsembleTracker.n_observations(...)`
- `misconceptions = MisconceptionTracker.for_concept(...)`
- `explanation = mastery/uncertainty/count + evidence text for each misconception with P(M) >= 0.3`

The engine adds the student's engagement, participation, confidence, overconfidence and confidently-wrong flag.

**Input:** registered student ID.  
**Output:** `StudentEstimate` containing per-concept `ConceptEstimate`s and behavior metrics.

## 8. Group/class summary and RL state (`aggregate.py`)

### 8.1 `summarise(students, concept_ids)`

For each concept:

- `mean_mastery = mean(student mastery)`
- `std_mastery = population standard deviation(student mastery)`
- `mean_uncertainty = mean(student uncertainty)`
- `fraction_below_threshold = mean(masteryp < 0.4)`
- `misconception_prevalence[M] = mean(P_student(M) >= 0.5)`

For each student:

- Left behind if mean mastery over all concepts `< 0.4`.
- High uncertainty if mean uncertainty over all concepts `> 0.15`.
- Confidently wrong if the behavior tracker flag is true.

For the group:

- Mean engagement and participation are arithmetic means.
- Participation Gini: sort participation as `x_(i)` ascending, then  
  `G = 2Σ(i × x_(i))/(nΣx_(i)) - (n+1)/n`; return `0` if empty or total participation is zero.

**Input:** one or more student estimates and concept IDs.  
**Output:** `GroupSummary`. Empty student input is rejected.

### 8.2 Discretised state

`_level(v, (a,b)) = 0` if `v<a`; `1` if `a<=v<b`; otherwise `2`.

For a selected concept, the state is:

1. Mean mastery: cuts `(0.4, 0.7)`.
2. Worst misconception prevalence: cuts `(0.15, 0.4)`.
3. Mean engagement: cuts `(0.4, 0.7)`.
4. Participation balance: `2 - level(Gini, (0.3, 0.5))`.
5. Fraction below mastery threshold: cuts `(0.1, 0.3)`.

**Output:** five integers, each in `{0,1,2}`: a maximum of `3^5 = 243` possible states.

## 9. Demo output (`demo.py`)

**Calls:** after all concept/action events are processed, the demo calls `engine.get_class_summary()`, prints per-concept mean mastery/fraction below `0.4`/most prevalent misconception, class engagement/Gini/left-behind/confidently-wrong lists, then calls `engine.discretised_state("astar")`.

For up to five students it calls `engine.get_student(sid)` and separately reads `sim.true_mastery(...)` and `sim.true_misconceptions(...)`.

**Input:** estimates and summaries from the engine; hidden truth read separately from the simulator for demonstration/evaluation.  
**Output:** printed comparison of estimated beliefs with simulator truth. The truth comparison is not an input to the student model.

## Call-flow summary

```text
demo.main
  ├─ astar_catalog + make_classroom
  ├─ ClassroomSimulator.step(action, concept)
  │    ├─ _teach → instruction / optional contribution
  │    └─ _answer → answer / no-response
  ├─ StudentModelEngine.update_many(events)
  │    └─ update(event)
  │         ├─ BehaviourTracker.update
  │         ├─ INSTRUCTION → EnsembleTracker.expose + misconception teaching update
  │         ├─ QUIZ/EXERCISE → misconception answer update + EnsembleTracker.observe
  │         └─ QUESTION/DISCUSSION → misconception signal + optional soft BKT score
  ├─ get_class_summary → get_student(s) → summarise
  ├─ discretised_state → summarise → discretise
  └─ print estimates; read simulator truth separately for comparison
```
