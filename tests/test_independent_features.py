import unittest
import pandas as pd
from src.fixture_models.independent_features import build_feature_table


def match(key, hour, home_goals=2, away_goals=1):
    kickoff = pd.Timestamp('2025-01-01T12:00:00Z') + pd.Timedelta(hours=hour)
    return {'match_id': key, 'competition_key': 'fictional', 'season_code': 'demo',
            'home_team_id': 'Northbridge', 'away_team_id': key,
            'kickoff_at': kickoff, 'available_at': kickoff + pd.Timedelta(hours=8),
            'home_goals': home_goals, 'away_goals': away_goals,
            'home_shots': 12, 'away_shots': 7, 'home_shots_on_target': 6,
            'away_shots_on_target': 2, 'home_corners': 5, 'away_corners': 3}


class IndependentFeaturesTests(unittest.TestCase):
    def test_stats_wait_for_result_availability(self):
        frame = build_feature_table([match('one', 0), match('two', 4), match('three', 24)])
        self.assertEqual(frame.iloc[1].home_matches_before, 0)
        self.assertEqual(frame.iloc[2].home_matches_before, 2)
        self.assertEqual(frame.iloc[2].home_last_5_shots_for, 12)

    def test_future_result_changes_do_not_change_earlier_features(self):
        first = build_feature_table([match('one', 0), match('two', 24)])
        changed = build_feature_table([match('one', 0), match('two', 24, 0, 9)])
        columns = ['home_matches_before', 'elo_diff', 'home_last_5_goals_for', 'league_home_goals_mean']
        pd.testing.assert_frame_equal(first[columns], changed[columns])

    def test_simultaneous_matches_do_not_consume_each_others_results(self):
        first = build_feature_table([match('one', 0), match('two', 0)])
        changed = build_feature_table([match('one', 0, 0, 9), match('two', 0)])
        self.assertEqual(first.iloc[1].home_matches_before, 0)
        self.assertEqual(first.iloc[1].elo_diff, changed.iloc[1].elo_diff)


if __name__ == '__main__':
    unittest.main()
