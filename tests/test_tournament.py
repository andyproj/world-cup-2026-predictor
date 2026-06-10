from src.tournament import (
    WINNER_SLOTS,
    _fallback_third_assignment,
    parse_third_place_mapping_html,
    rank_group,
    SimulatedMatch,
)


def test_mapping_html_parser():
    cells = "".join(
        f"<td>{value}</td>"
        for value in [
            1,
            "E", "F", "G", "H", "I", "J", "K", "L",
            "3E", "3J", "3I", "3F", "3H", "3G", "3L", "3K",
        ]
    )
    # Repeat valid unique-looking rows only to exercise token parsing, then expect
    # the production validation to reject an incomplete table.
    html = f"<table><tr>{cells}</tr></table>"
    try:
        parse_third_place_mapping_html(html)
    except ValueError as exc:
        assert "495" in str(exc)
    else:
        raise AssertionError("Incomplete mapping table should be rejected")


def test_fallback_assignment_is_bijective_and_allowed():
    groups = set("EFGHIJKL")
    assignment = _fallback_third_assignment(groups)
    assert set(assignment) == set(WINNER_SLOTS)
    assert set(assignment.values()) == groups


def test_group_ranking_uses_points_then_head_to_head():
    teams = ["A", "B", "C", "D"]
    matches = [
        SimulatedMatch("A", "B", 1, 0),
        SimulatedMatch("A", "C", 0, 1),
        SimulatedMatch("A", "D", 2, 0),
        SimulatedMatch("B", "C", 2, 0),
        SimulatedMatch("B", "D", 1, 0),
        SimulatedMatch("C", "D", 0, 1),
    ]
    ranked, _ = rank_group(teams, matches, {team: 1500 for team in teams})
    assert ranked[0] == "A"  # A beat B head to head; both have six points.
    assert ranked[1] == "B"
