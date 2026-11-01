# AI Project

## Python version

Python 3.13.3

## Install requirements

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Then:

```bash
python -m pip install -r requirements.txt
```

## Run

Run every command from the repository root.

```bash
python example.py
python main.py --scenario widespread_misconception
python main.py --policy rl --scenario small_group_behind
python main.py --compare
python -m pytest
```

## Where each part of the brief lives

| Brief | Folder | What it does |
|---|---|---|
| Module 1: Search (graphs, diagnosis) | `module1_knowledge_graph/` | Concept, prerequisite, misconception and resource graphs; diagnostic search; student-knowledge graph; the course catalog every module uses |
| Module 1: Constraint satisfaction | `module2_csp/` | Classroom CSP for groups, TAs and time slots, with infeasibility explanations |
| Module 2: Planning | `module4_planner/` | Teaching actions with preconditions and effects, the lesson planner, and the executor |
| Module 3: Reasoning under uncertainty | `module3_student_model/` | BKT mastery, misconceptions, uncertainty, behaviour, and the classroom simulator |
| Module 4: Reinforcement learning | `module5_rl/` | Q-learning and SARSA teaching policies and baselines |
| Module 4: NLU | `module5_nlu/` | Question intent, concept and misconception detection |
| Module 5: Generative AI | not started | |
| Shared data formats | `shared/schema.py` | |
| Whole system | `main.py` | Runs all modules together |

Folder numbers follow the team numbers, not the module numbers in the brief.
