# Agentic Classroom Digital Twin

A simulated classroom for trying out teaching strategies before using them on real students. A teacher agent decides what to do next (lecture, quiz, worked example, peer discussion, TA help, revision, move on) for a class of simulated students who know different things, hold different misconceptions and speak up to different degrees. We then measure what actually helped, including whether it helped the weaker students or only the strong ones.

This is our project for the Artificial Intelligence course (Group 8, IIT Bhubaneswar, 2026). The simulated course is a short unit on search: BFS, DFS, path cost, priority queues, heuristics, UCS, greedy search and A*, 11 concepts in all.

Nothing here calls an LLM or the internet, and no API keys are needed. Module 5 of the brief (Generative AI) is not part of this repository.

## Setup

You need Python 3.12 or newer. We developed on 3.13 and also ran everything on 3.12; we haven't tried older versions. Check with `python --version` (or `python3 --version` on macOS and Linux).

Open a terminal in the project folder, the one that contains `main.py` and `requirements.txt`. In VS Code, Terminal > New Terminal does this for you.

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**Windows (Command Prompt)**

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

After activating, your prompt starts with `(.venv)`. You have to activate again each time you open a new terminal; the other two steps are one-off.

Check that everything works:

```bash
python example.py
```

It takes about 20 seconds and should end with `18/18 checks passed`.

## Running the simulation

Run all commands from the project folder.

```bash
python main.py                                   # one lesson, default settings
python main.py --scenario widespread_misconception
python main.py --policy rl --scenario small_group_behind
python main.py --compare                         # every policy on every scenario, about 2 minutes
```

`main.py` prints each teaching step with the reason it was chosen, the student groups the scheduler formed, what the language analysis made of student questions, a diagnosis for a struggling student, and a final summary. Outcomes are measured from the simulator's true student state, not from the model's estimates.

| Option | Default | Meaning |
|---|---|---|
| `--scenario` | `mixed` | Which kind of class to simulate (see below) |
| `--policy` | `plan` | Who chooses the next action (see below) |
| `--students` | 24 | Class size |
| `--seed` | 0 | Random seed; the same seed gives the same class and the same run |
| `--time-budget` | 80 | Class time available. A lecture or example costs 1, a quiz 0.5, a discussion or exercise 1.5 |
| `--no-csp` | off | Skip the group scheduler |
| `--no-nlu` | off | Skip the question analysis |
| `--compare` | off | Compare all policies on all scenarios instead of one run |
| `--seeds` | 3 | Number of seeds used by `--compare` |

Scenarios:

- `mixed`: a normal spread of strong, average and struggling students.
- `widespread_misconception`: about 70% of the class holds the same misconception (treating path cost g(n) as the remaining cost).
- `small_group_behind`: roughly one student in six is struggling.
- `dominant_talkers`: a few very talkative students dominate discussions.
- `confident_wrong`: some students are confident but hold misconceptions.
- `quiet_knowers`: a quarter of the class understands the material but rarely speaks.

Policies:

- `plan`: our planner. It searches for the cheapest sequence of lecture, quiz, discussion, exercise and assessment that fits the class's current state, and replans after every step.
- `heuristic`: a transparent set of if-then rules.
- `rl`: a Q-learning policy trained on the simulator.
- `fixed`: the same lecture, example, exercise, discussion, quiz routine for every concept.
- `threshold`: lecture and quiz, and repeat the lecture while the class average is below 0.6.

## What we found

Averages over all six scenarios and three seeds (24 students, time budget 80, `python main.py --compare`). "Known" is the share of student-concept pairs truly mastered across all 11 concepts. "Left behind" is the share of students who mastered less than 40% of the course. "Coverage" is the share of concepts the class got through.

| Policy | Known | Left behind | Coverage |
|---|---|---|---|
| rl | 0.611 | 0.204 | 0.717 |
| plan | 0.563 | 0.264 | 0.566 |
| fixed | 0.511 | 0.405 | 1.000 |
| heuristic | 0.464 | 0.465 | 0.409 |
| threshold | 0.459 | 0.475 | 1.000 |

The adaptive policies leave fewer students behind and end with more knowledge, but they spend time on weak spots and so don't finish the syllabus in the time available. The fixed plan covers everything and leaves 41% of students behind. The benefit is not uniform: in `dominant_talkers` the planner's lead over the fixed plan is small, and with seed 0 alone (the run in `example.py`) the fixed plan is slightly ahead on knowledge.

These are results on a simulator we built, not on real students. The RL README has a further caveat: a policy that simply gives seven worked examples per concept does about as well as the learned one, because worked examples are very effective in our simulator.

## How the pieces fit together

Each step of `main.py`:

1. The planner picks the next teaching action from the student model's estimates, using the chosen policy.
2. For peer-learning and TA actions, the classroom scheduler splits students into groups, TAs and time slots.
3. The simulator applies the action and returns events: quiz answers, questions, discussion contributions.
4. Student questions are re-labelled by the language-understanding pipeline.
5. The student model updates its estimates from the events.
6. For a struggling student, knowledge-graph search ranks the likely root causes and suggests resources.

## Where things are

The folder numbers follow our team split, not the module numbers in the brief.

| Brief | Folder | What it holds |
|---|---|---|
| Module 1: search | [`module1_knowledge_graph/`](module1_knowledge_graph/README.md) | Concept graph, diagnostic search, student-knowledge graph, the course catalog every module uses |
| Module 1: constraint satisfaction | [`module2_csp/`](module2_csp/README.md) | Classroom scheduling as a CSP, with explanations when it is infeasible |
| Module 2: planning | [`module4_planner/`](module4_planner/README.md) | Teaching actions, the planner, and the loop that runs a lesson |
| Module 3: uncertainty | [`module3_student_model/`](module3_student_model/README.md) | Bayesian knowledge tracing, misconception beliefs, and the classroom simulator |
| Module 4: reinforcement learning | [`module5_rl/`](module5_rl/README.md) | Q-learning and SARSA teaching policies, baselines, evaluation |
| Module 4: language understanding | [`module5_nlu/`](module5_nlu/README.md) | Question intent, concept and misconception detection |
| Module 5: generative AI | not included | |
| Shared data formats | `shared/schema.py` | Events, estimates and summaries passed between modules |

`example.py` is a quick end-to-end check of every module. `test_main.py` and the tests inside each folder are run together by `python -m pytest`.

## Tests

```bash
python -m pytest
```

All tests should pass (200 at the time of writing, in about 5 seconds).

## If something goes wrong

- **`ModuleNotFoundError: No module named 'shared'` (or `module1_knowledge_graph`, etc.)**: you are not in the project folder. `cd` into the folder with `main.py`. Run modules as `python -m folder.script`, not `python folder/script.py`.
- **`ModuleNotFoundError: No module named 'numpy'` (or `pydantic`, `pytest`)**: the virtual environment isn't active. Activate it again; the prompt should show `(.venv)`.
- **`python: command not found`** on macOS or Linux: use `python3` for the `venv` step. After activating, `python` works. On Windows, try `py` instead of `python`.
- **PowerShell says "running scripts is disabled on this system"**: run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again. Or use Command Prompt.
- **`FileNotFoundError` mentioning `module5_rl/results`**: run the command from the project folder.
