# Module 3: Reasoning under Uncertainty (Probabilistic Student Modelling)

This module estimates the following for every student and concept:

- how likely the student is to have **mastered** the concept,
- how **certain** the system is about that estimate,
- which **misconceptions** the student probably holds (tracked separately from mastery),
- the student's **engagement**, **participation** and **confidence**, and whether they are **confidently wrong**.

The estimates are updated after every classroom event: quiz answers, exercises, hints, questions, discussion and instruction. The module also includes a **classroom simulator**. It holds each student's true hidden state and produces the noisy events the estimator has to interpret. The estimator never sees the true state. It infers knowledge only from observable behaviour, as a teacher would.

```
classroom event ──► StudentModelEngine.update(event)
                        ├── BKT ensemble ........ mastery + uncertainty per (student, concept)
                        ├── misconception model . P(misconception) per (student, misconception)
                        └── behaviour model ..... engagement, participation, confidence
                        ▼
      get_student(id) · get_group_summary(ids) · get_class_summary() · discretised_state(concept)
```

| Requirement | How it is handled |
| --- | --- |
| Estimate mastery with BKT or a similar model | `bkt.py`: BKT for each student and concept |
| Update after answers, clarifications, discussion, hints and exercises | `engine.py` handles every event type |
| Account for guessing and slips | BKT guess and slip parameters; a higher guess rate after a hint |
| Handle incomplete or unclear questions | Evidence is weighted by NLU confidence; unclear questions don't change mastery |
| Keep an uncertainty estimate per concept | `uncertainty.py`: ensemble spread plus the number of observations |
| Distinguish "not yet learned" from "holds a misconception" | `misconceptions.py`: a separate probability per misconception |

---

## Files

| File | Purpose |
| --- | --- |
| `bkt.py` | BKT updates, hints, partial credit, the learning step, parameter fitting |
| `uncertainty.py` | Ensemble of BKT trackers; the mean is the estimate and the spread is the uncertainty |
| `misconceptions.py` | Bayesian belief per misconception |
| `behaviour.py` | Engagement, participation, confidence, overconfidence |
| `aggregate.py` | Group and class summaries, participation Gini, the discretised RL state |
| `engine.py` | `StudentModelEngine`, the entry point for the other modules |
| `profiles.py` | `StudentProfile`, archetypes, scenario classrooms |
| `simulator.py` | `ClassroomSimulator` (true hidden state → events) |
| `sample_catalog.py` | Small 6-concept A* course for the demo and tests |
| `demo.py` | Runs one simulated lesson and prints the estimates next to the truth |
| `test_*.py` | Unit and integration tests |

Shared data formats (`Event`, `StudentEstimate`, `GroupSummary`, `CourseCatalog`) are in `shared/schema.py`.

---

## Theory

### 1. Bayesian Knowledge Tracing (BKT)

Knowledge of a concept is a hidden binary variable $L$ (known or not known). Answers are noisy signals of it.

| Symbol | Meaning | Default |
| --- | --- | --- |
| $P(L_0)$ | prior probability that the concept is known | 0.20 |
| $T$ | probability of learning at each opportunity | 0.15 |
| $G$ | probability of a correct answer without knowing (guess) | 0.20 |
| $S$ | probability of a wrong answer despite knowing (slip) | 0.10 |
| $G_h$ | guess probability after a hint | 0.50 |

The parameters must satisfy $G + S < 1$.

**Observation update (Bayes' rule):**

$$
P(L \mid \text{correct}) = \frac{P(L)(1-S)}{P(L)(1-S) + (1-P(L))G}
\qquad
P(L \mid \text{wrong}) = \frac{P(L)S}{P(L)S + (1-P(L))(1-G)}
$$

**Learning transition** after each answer:

$$
P(L_{t+1}) = P(L \mid \text{obs}) + \big(1 - P(L \mid \text{obs})\big)\,T
$$

**Hints.** A correct answer after a hint is weaker evidence, so the update uses $G_h$ in place of $G$.

**Partial credit.** For a score $s \in [0,1]$, the likelihoods are
$\ell_{\text{known}} = s(1-S) + (1-s)S$ and $\ell_{\text{unknown}} = sG + (1-s)(1-G)$, combined with Bayes' rule. A score of 0.5 leaves the estimate unchanged.

**Instruction without an answer.** Only the learning step is applied, scaled by the action's strength $w_a$:

$$
P(L) \leftarrow P(L) + (1 - P(L))\,T\,w_a
$$

| Action | $w_a$ |
| --- | --- |
| lecture | 0.5 |
| peer discussion | 0.6 |
| example | 0.7 |
| peer learning, revision | 0.8 |
| TA intervention | 1.0 |
| quiz, exercise, assessment | 0 (the learning step comes with the answer) |

Probabilities are clipped to $[10^{-6}, 1-10^{-6}]$.

Because of the learning transition, **the order of answers matters**. Two students who both score 2 out of 5 get different estimates: wrong-wrong-wrong-right-right ends much higher than right-right-wrong-wrong-wrong.

**Parameter fitting.** `fit_params_grid` searches a grid of parameter values for the set that maximises
$\sum_t \log\big[c_t\,p_t + (1-c_t)(1-p_t)\big]$, where $c_t$ is 1 for a correct answer and 0 otherwise, and $p_t = P(L_t)(1-S) + (1-P(L_t))G$.

**Optional prerequisite prior.** `prior_from_prerequisites` adjusts the prior using the average mastery of the concept's prerequisites, $\overline{m}$:

$$
P(L_0') = P(L_0)\big((1-w) + w\,\overline{m}\big), \quad w = 0.5
$$

### 2. Uncertainty: an ensemble of BKT models

A single probability can't tell "50% after many mixed answers" apart from "no evidence yet". The module therefore runs **$K = 15$ BKT trackers**, each with parameters sampled around the base values:

$$
\theta_k \sim \text{Beta}(\mu\kappa,\,(1-\mu)\kappa)
$$

Here $\mu$ is the base value and $\kappa$ controls the spread: $\kappa = 4$ for `p_init` (wide spread) and $\kappa = 30$ for the other parameters (narrow spread). Samples with $G + S \ge 0.95$ are rejected, and $G_h \ge G$ is enforced.

$$
\hat{m} = \frac{1}{K}\sum_k P_k(L) \qquad
u = \sqrt{\frac{1}{K}\sum_k \big(P_k(L) - \hat{m}\big)^2}
$$

With little evidence the members disagree and $u$ is large. Consistent evidence brings them together and $u$ shrinks. A student who never answers keeps a **high uncertainty** rather than a low mastery estimate, so the planner can tell "unknown" apart from "weak". The number of observations (`n_observations`) is reported as well.

### 3. Misconceptions: a separate Bayesian belief

A student who **hasn't learned** a concept and a student who holds a **specific misconception** both answer wrongly, but they need different teaching. For each student and misconception $M$, the module keeps a separate $P(M)$, with prior 0.15, updated by Bayes' rule:

$$
P(M \mid \text{obs}) = \frac{P(M)\,\ell_{\text{held}}}{P(M)\,\ell_{\text{held}} + (1-P(M))\,\ell_{\text{not held}}}
$$

**Answer model.**

- A student who holds $M$ picks $M$'s wrong option with probability $e = 0.7$, and otherwise behaves like a student without $M$.
- A student without $M$ answers correctly with probability $q = P(L)(1-S) + (1-P(L))G$. Otherwise they pick one of the $n_d$ wrong options at random.

Writing $w = 1 - q$:

| Observation | $\ell_{\text{held}}$ | $\ell_{\text{not held}}$ | Effect |
| --- | --- | --- | --- |
| Correct answer | $(1-e)q$ | $q$ | decreases |
| Picked **$M$'s** wrong option | $e + (1-e)\frac{w}{n_d}$ | $\frac{w}{n_d}$ | increases sharply |
| Picked a wrong option not linked to any misconception | $(1-e)\frac{w}{n_d}$ | $\frac{w}{n_d}$ | decreases |
| Picked **another** misconception's option | — | — | unchanged (misconceptions can co-exist) |

Only misconceptions that the question's wrong options test (`available_misconceptions`) are updated.

**Result.** Repeated wrong answers on unlinked options lead to low mastery and a low $P(M)$: the student hasn't learned the concept. Repeatedly choosing $M$'s option leads to low mastery and a high $P(M)$: the student holds misconception $M$.

**Teaching** reduces the belief: $P(M) \leftarrow P(M)(1 - r\,w_a)$, with $r = 0.35$ if the action targets $M$ and $0.05$ otherwise.

Each estimate carries an **explanation**. For example: `P(g_is_remaining_cost) = 0.96 (picked its wrong option 4 of 4 times, 0 NLU signals)`.

### 4. Questions and incomplete evidence

Questions and discussion come with an NLU intent, a confidence $c$, and any detected misconceptions.

**Misconception signal.** When the NLU detects $M$, the update uses
$\ell_{\text{held}} = 0.6c + (1-c)$ and $\ell_{\text{not held}} = 0.1c + (1-c)$.
At $c = 0$ the belief doesn't change. For unclear questions, $c$ is halved first.

**Soft mastery evidence.** A question is applied as a partial-credit observation with score $s = 0.5 + \Delta\cdot c$:

| Intent | $\Delta$ |
| --- | --- |
| confusion | −0.3 |
| misconception signal | −0.2 |
| clarification | +0.1 |
| example request | 0 |
| unclear | no mastery update |

This handles **incomplete questions**: vague questions move the estimates very little, and unclear ones don't move mastery at all.

### 5. Engagement, participation and confidence

**Engagement** is an exponential moving average: $\text{eng} \leftarrow 0.8\,\text{eng} + 0.2\,x$, starting at 0.7. The signal $x$ depends on the event:

| Event | $x$ |
| --- | --- |
| no response | 0 |
| answer | 0.7 |
| hint | 0.8 |
| question, discussion | 1.0 |

**Participation** has a Beta(1, 1) prior, where an opportunity is an instruction event and a contribution is a voluntary question or discussion:

$$
\text{part} = \min\left(1, \frac{\text{contributions} + 1}{\text{opportunities} + 2}\right)
$$

**Confidence** is a moving average (rate 0.3) of the confidence the student states.

**Overconfidence** is the averaged gap between stated confidence and outcome:

$$
\text{over} \leftarrow (1-w_n)\,\text{over} + w_n(\text{confidence} - \text{outcome}), \qquad w_n = \max(1/n,\,0.25)
$$

A student is flagged **confidently wrong** when they have given at least 3 answers and $\text{over} > 0.25$. The flag clears once the student becomes well calibrated again.

### 6. Group and class summaries

**For each concept**, the summary reports:

- mean and standard deviation of mastery,
- mean uncertainty,
- the fraction of students below 0.4 mastery,
- **misconception prevalence**: the fraction of students with $P(M) \ge 0.5$.

**For each group**, it reports:

- mean engagement and participation,
- the students left behind (mean mastery < 0.4),
- the students with high uncertainty (mean uncertainty > 0.15),
- the confidently wrong students,
- the **participation Gini coefficient**, with participation values sorted in ascending order. 0 means equal participation; 1 means one student does all the talking.

$$
G = \frac{2\sum_i i\,x_{(i)}}{n\sum_i x_{(i)}} - \frac{n+1}{n}
$$

### 7. Discretised state for reinforcement learning

`discretised_state(concept)` maps a summary to five levels, each 0, 1 or 2, giving $3^5 = 243$ states:

| Feature | Cut-offs |
| --- | --- |
| mean mastery | 0.4, 0.7 |
| worst misconception prevalence | 0.15, 0.4 |
| mean engagement | 0.4, 0.7 |
| participation balance | 2 − level(Gini; cuts 0.3, 0.5) |
| share of students left behind | 0.1, 0.3 |

---

## Classroom simulator

`ClassroomSimulator` holds each student's true state: which concepts they know, which misconceptions they hold, and their engagement. `step(action, concept, target_ids, target_misconception)` returns **events only**, never the true state.

**Learning.** A student who doesn't know a concept learns it from an action with probability

$$
p = \min\Big(0.95,\; r_a \cdot \text{speed} \cdot (0.4 + 0.6\,\text{eng}) \cdot \big(0.3 + 0.7 \cdot \text{fraction of prerequisites known}\big)\Big)
$$

| Action | Rate $r_a$ |
| --- | --- |
| diagnostic quiz, assessment | 0.03 |
| lecture | 0.12 |
| exercise | 0.12 |
| peer discussion | 0.15 |
| example | 0.18 |
| peer learning, revision | 0.20 |
| TA intervention | 0.30 |

A student who holds a misconception about the concept **can't learn it** until the misconception is removed. Each teaching action removes it with probability $r(0.4 + 0.6\,\text{eng})$, where $r = 0.5$ if the action is targeted at it and $0.06$ otherwise.

**Answers.**

- The student gives no response with probability $(1-\text{eng})\cdot 0.5$.
- A student who holds a misconception the question tests picks its wrong option with probability 0.7.
- Otherwise, the answer is correct with probability $1 - \text{slip}$ if the student knows the concept, or with the guess probability (0.5 after a hint) if not.
- Stated confidence is the profile's confidence ± a small shift that depends on knowledge, plus noise.

**Questions.** During teaching, a student contributes with probability $\text{participation} \times \text{eng} \times 0.6$. The intent reflects the true state:

- **unclear** 15% of the time;
- otherwise, if the student holds a misconception, a **misconception signal** with probability 0.6;
- otherwise, **confusion** if they don't know the concept, or a **clarification** or **example request** if they do.

Each contribution includes question text, for example "Isn't it true that path cost is the estimated remaining cost to the goal?", which the NLU module can analyse.

**Engagement** drifts with each action, pulls back towards the student's baseline, and gets a little noise:

| Action | Effect |
| --- | --- |
| lecture | −0.04 |
| assessment, revision | −0.02 |
| diagnostic quiz | −0.01 |
| exercise | 0 |
| example | +0.02 |
| TA intervention | +0.04 |
| peer discussion, peer learning | +0.05 |

All randomness comes from one seeded generator, and sets are iterated in sorted order, so a given seed always gives the same run.

---

## Student profiles and scenarios

| Archetype | Prior | Speed | Confidence | Participation | Engagement | Misconception rate |
| --- | --- | --- | --- | --- | --- | --- |
| strong | 0.80 | 1.3 | 0.75 | 0.50 | 0.85 | 0.00 |
| average | 0.40 | 1.0 | 0.55 | 0.35 | 0.70 | 0.15 |
| struggling | 0.10 | 0.6 | 0.35 | 0.20 | 0.55 | 0.30 |
| quiet_knower | 0.80 | 1.2 | 0.40 | 0.05 | 0.75 | 0.00 |
| dominant | 0.50 | 1.0 | 0.85 | 0.90 | 0.90 | 0.15 |
| confident_wrong | 0.25 | 0.9 | 0.90 | 0.60 | 0.75 | 0.35 |
| disengaged | 0.30 | 0.8 | 0.40 | 0.10 | 0.30 | 0.15 |

Each student's prior for a concept is the archetype prior, multiplied by 0.8 if the concept has prerequisites, plus $\mathcal{N}(0, 0.08)$ noise.

| Scenario | Composition |
| --- | --- |
| `mixed` | all archetypes |
| `widespread_misconception` | about 70% of students hold one chosen misconception |
| `small_group_behind` | about 1/6 struggling |
| `dominant_talkers` | about 1/8 dominant |
| `confident_wrong` | about 1/5 confidently wrong |
| `quiet_knowers` | about 1/4 quiet knowers |

---

## Using the module

Run from the repository root.

```python
from module1_knowledge_graph.catalog import load_course_catalog
from module3_student_model import ClassroomSimulator, StudentModelEngine, make_classroom
from shared.schema import ActionType, Event, EventType

catalog = load_course_catalog()
engine = StudentModelEngine(catalog)
engine.update(Event(student_id="s01", concept_id="Path_Cost", type=EventType.QUIZ_ANSWER,
                    correct=False, chosen_misconception="g_is_remaining_cost", stated_confidence=0.9))

est = engine.get_student("s01").concepts["Path_Cost"]
est.p_mastery, est.uncertainty, est.n_observations, est.misconceptions, est.explanation

summary = engine.get_class_summary()
engine.discretised_state("Path_Cost")

profiles = make_classroom(catalog, "small_group_behind", n_students=24, seed=0)
sim = ClassroomSimulator(catalog, profiles, seed=0)
engine = StudentModelEngine(catalog, sim.student_ids())
engine.update_many(sim.step(ActionType.DIAGNOSTIC_QUIZ, "Path_Cost"))
sim.true_mastery("s03", "Path_Cost")   # ground truth, for evaluation only
```

**Event handling**

| Event | Effect |
| --- | --- |
| `INSTRUCTION` | learning step (scaled by $w_a$); misconception reduction; counts as a participation opportunity |
| `QUIZ_ANSWER`, `EXERCISE` | BKT update; misconception update; engagement and calibration update. Needs `correct` or `score`. |
| `QUESTION`, `DISCUSSION` | soft mastery evidence; misconception signals; counts as a contribution if voluntary |
| `HINT` | engagement only (affects mastery through `hint_used` on the next answer) |
| `NO_RESPONSE` | lowers engagement; **not** treated as a wrong answer |

**Configuration.** All parameters are in `EngineConfig`:

```python
from module3_student_model.engine import EngineConfig
from module3_student_model.bkt import BKTParams

config = EngineConfig(bkt=BKTParams(p_init=0.2, p_learn=0.15, p_guess=0.2, p_slip=0.1),
                      bkt_per_concept={"Astar": BKTParams(p_init=0.1)}, ensemble_size=15, seed=0)
```

`EngineConfig` also holds the misconception parameters (`MisconceptionParams`), the behaviour parameters (`BehaviourParams`) and the summary thresholds (`SummaryThresholds`).

---
