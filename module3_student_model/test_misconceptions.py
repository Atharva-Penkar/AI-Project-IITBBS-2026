from module3_student_model.misconceptions import MisconceptionTracker

MS = ["m1", "m2"]


def answer(t, correct, chosen=None, p_mastery=0.3, student="s"):
    t.update_from_answer(student, MS, correct=correct, chosen_misconception=chosen,
                         p_mastery=p_mastery, guess=0.2, slip=0.1)


def test_picking_linked_wrong_option_raises_belief_sharply():
    t = MisconceptionTracker()
    answer(t, False, chosen="m1")
    assert t.get("s", "m1") > 0.4
    answer(t, False, chosen="m1")
    assert t.get("s", "m1") > 0.7
    answer(t, False, chosen="m1")
    assert t.get("s", "m1") > 0.85
    assert t.get("s", "m2") == t.params.prior


def test_lacking_knowledge_is_not_a_misconception():
    t = MisconceptionTracker()
    for _ in range(5):
        answer(t, False, chosen=None, p_mastery=0.1)
    assert t.get("s", "m1") < 0.15
    assert t.get("s", "m2") < 0.15


def test_correct_answers_lower_belief():
    t = MisconceptionTracker()
    answer(t, False, chosen="m1")
    before = t.get("s", "m1")
    answer(t, True)
    assert t.get("s", "m1") < before


def test_question_that_cannot_reveal_a_misconception_leaves_it_alone():
    t = MisconceptionTracker()
    t.update_from_answer("s", MS, correct=True, chosen_misconception=None, p_mastery=0.5,
                         guess=0.2, slip=0.1, available_misconceptions=["m1"])
    assert t.get("s", "m2") == t.params.prior


def test_nlu_signal_is_scaled_by_confidence():
    sure, vague, none = MisconceptionTracker(), MisconceptionTracker(), MisconceptionTracker()
    sure.update_from_signal("s", "m1", 0.9)
    vague.update_from_signal("s", "m1", 0.2)
    none.update_from_signal("s", "m1", 0.0)
    assert sure.get("s", "m1") > vague.get("s", "m1") > none.get("s", "m1")
    assert none.get("s", "m1") == none.params.prior


def test_targeted_instruction_removes_misconception_faster():
    targeted, general = MisconceptionTracker(), MisconceptionTracker()
    for t in (targeted, general):
        answer(t, False, chosen="m1")
        answer(t, False, chosen="m1")
    targeted.apply_instruction("s", MS, target_misconception="m1", strength=1.0)
    general.apply_instruction("s", MS, target_misconception=None, strength=1.0)
    assert targeted.get("s", "m1") < general.get("s", "m1")