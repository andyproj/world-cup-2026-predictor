from src.elo import elo_three_way_probabilities
from src.poisson_model import poisson_probabilities


def test_elo_probabilities_sum_to_one():
    values = elo_three_way_probabilities(1700, 1500)
    assert abs(sum(values) - 1.0) < 1e-10


def test_stronger_team_has_higher_elo_win_probability():
    home, draw, away = elo_three_way_probabilities(1800, 1450)
    assert home > draw
    assert home > away


def test_poisson_probabilities_sum_to_one():
    home, draw, away, home_goals, away_goals = poisson_probabilities(1.8, 0.9)
    assert abs(home + draw + away - 1.0) < 1e-10
    assert home_goals >= 0
    assert away_goals >= 0
