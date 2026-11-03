
import copy

from module2_csp.classroom_csp import format_result, make_demo_classroom, solve_classroom


def main() -> None:
    cfg = make_demo_classroom()

    print("=== Demo 1: normal classroom ===")
    res = solve_classroom(cfg)
    print(format_result(cfg, res))

    print("\n=== Demo 2: overloaded TAs (each TA may supervise only 1 group) ===")
    bad = copy.deepcopy(cfg)
    for t in bad.tas:
        t.max_groups = 1
    print(format_result(bad, solve_classroom(bad)))

    print("\n=== Demo 3: no revision of arrays, and S01 has weak prerequisites ===")
    bad2 = copy.deepcopy(cfg)
    bad2.activities = [a for a in bad2.activities if a.aid != "act_arrays"]
    bad2.students[0].mastery = {c: 0.1 for c in bad2.students[0].mastery}
    print(format_result(bad2, solve_classroom(bad2)))

    print("\n=== Demo 4: TA loads must be exactly equal (needs a multiple of 3 groups) ===")
    bad3 = copy.deepcopy(cfg)
    bad3.ta_balance_tolerance = 0
    print(format_result(bad3, solve_classroom(bad3)))


if __name__ == "__main__":
    main()
