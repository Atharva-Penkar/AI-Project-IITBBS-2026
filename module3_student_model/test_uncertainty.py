from module3_student_model.uncertainty import EnsembleTracker, binary_entropy


def test_uncertainty_shrinks_with_consistent_evidence():
    t = EnsembleTracker(seed=1)
    t.observe("s", "c", correct=True)
    early = t.std("s", "c")
    for _ in range(6):
        t.observe("s", "c", correct=True)
    assert t.std("s", "c") < early
    assert t.mean("s", "c") > 0.95


def test_student_with_no_evidence_is_more_uncertain_than_tested_student():
    t = EnsembleTracker(seed=2)
    for _ in range(6):
        t.observe("tested", "c", correct=False)
    assert t.std("untested", "c") > t.std("tested", "c")
    assert t.n_observations("untested", "c") == 0
    assert t.n_observations("tested", "c") == 6


def test_same_seed_gives_identical_ensembles():
    a, b = EnsembleTracker(seed=5), EnsembleTracker(seed=5)
    a.observe("s", "c", correct=True)
    b.observe("s", "c", correct=True)
    assert a.mean("s", "c") == b.mean("s", "c")


def test_binary_entropy_peaks_at_half():
    assert binary_entropy(0.5) == 1.0
    assert binary_entropy(0.99) < 0.1