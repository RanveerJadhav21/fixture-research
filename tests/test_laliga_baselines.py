import json
import unittest

import pandas as pd

from src.fixture_models.laliga_baselines import (
    PROBABILITY_COLUMNS,
    build_chronological_baselines,
    build_ios_backtest_seed,
    build_prospective_baselines,
    build_research_report,
    build_team_table,
)


def match(
    key,
    kickoff,
    home,
    away,
    home_goals,
    away_goals,
    season="2024-25",
    split="validation",
):
    outcome = "H" if home_goals > away_goals else "A" if home_goals < away_goals else "D"
    return {
        "source_record_key": key,
        "season_key": season,
        "research_split": split,
        "kickoff_at": kickoff,
        "home_team_key": f"club:{home}",
        "home_team_name": home,
        "away_team_key": f"club:{away}",
        "away_team_name": away,
        "home_score_ft": home_goals,
        "away_score_ft": away_goals,
        "outcome_ft": outcome,
    }


class LaLigaBaselineTests(unittest.TestCase):
    def setUp(self):
        self.matches = pd.DataFrame(
            [
                match("m1", "2024-08-01T18:00:00Z", "A", "B", 2, 0),
                match("m2", "2024-08-01T18:00:00Z", "C", "D", 0, 1),
                match("m3", "2024-08-08T18:00:00Z", "B", "C", 1, 1),
                match(
                    "m4",
                    "2025-08-01T18:00:00Z",
                    "E",
                    "A",
                    1,
                    0,
                    season="2025-26",
                    split="sealed_test",
                ),
            ]
        )

    def test_predictions_are_normalized_and_reproducible(self):
        first = build_chronological_baselines(self.matches)
        second = build_chronological_baselines(self.matches)
        pd.testing.assert_frame_equal(first, second)
        for columns in PROBABILITY_COLUMNS.values():
            sums = first[list(columns)].sum(axis=1)
            self.assertTrue(((sums - 1.0).abs() < 1e-12).all())
        cutoffs = first.dropna(subset=["data_cutoff"])
        self.assertTrue(
            (
                pd.to_datetime(cutoffs["data_cutoff"], utc=True)
                < pd.to_datetime(cutoffs["kickoff_at"], utc=True)
            ).all()
        )

    def test_same_kickoff_results_cannot_leak_across_matches(self):
        original = build_chronological_baselines(self.matches)
        changed = self.matches.copy()
        changed.loc[changed["source_record_key"] == "m1", ["home_score_ft", "away_score_ft"]] = [0, 8]
        changed.loc[changed["source_record_key"] == "m1", "outcome_ft"] = "A"
        rerun = build_chronological_baselines(changed)
        prediction_columns = [column for columns in PROBABILITY_COLUMNS.values() for column in columns]
        pd.testing.assert_series_equal(
            original.loc[original["source_record_key"] == "m2", prediction_columns].iloc[0],
            rerun.loc[rerun["source_record_key"] == "m2", prediction_columns].iloc[0],
        )

    def test_overlapping_result_cannot_leak_into_later_kickoff(self):
        matches = pd.DataFrame(
            [
                match("early", "2024-08-01T18:00:00Z", "A", "B", 2, 0),
                match("late", "2024-08-01T20:00:00Z", "A", "C", 1, 1),
            ]
        )
        original = build_chronological_baselines(matches)
        matches.loc[matches["source_record_key"] == "early", ["home_score_ft", "away_score_ft"]] = [0, 8]
        matches.loc[matches["source_record_key"] == "early", "outcome_ft"] = "A"
        rerun = build_chronological_baselines(matches)
        prediction_columns = [
            column for columns in PROBABILITY_COLUMNS.values() for column in columns
        ]
        pd.testing.assert_series_equal(
            original.loc[
                original["source_record_key"] == "late", prediction_columns
            ].iloc[0],
            rerun.loc[rerun["source_record_key"] == "late", prediction_columns].iloc[0],
        )
        self.assertTrue(pd.isna(original.loc[1, "data_cutoff"]))

    def test_future_result_change_does_not_change_prior_features(self):
        original = build_chronological_baselines(self.matches)
        changed = self.matches.copy()
        changed.loc[changed["source_record_key"] == "m3", ["home_score_ft", "away_score_ft"]] = [7, 0]
        changed.loc[changed["source_record_key"] == "m3", "outcome_ft"] = "H"
        rerun = build_chronological_baselines(changed)
        before_m3 = original[original["kickoff_at"] <= "2024-08-08T18:00:00+00:00"]
        before_m3_rerun = rerun[rerun["kickoff_at"] <= "2024-08-08T18:00:00+00:00"]
        comparable = [column for column in original.columns if column not in {"home_score_ft", "away_score_ft", "outcome_ft"}]
        pd.testing.assert_frame_equal(
            before_m3[comparable].reset_index(drop=True),
            before_m3_rerun[comparable].reset_index(drop=True),
        )

    def test_season_boundary_marks_new_team_and_reverts_elo(self):
        predictions = build_chronological_baselines(self.matches)
        row = predictions[predictions["source_record_key"] == "m4"].iloc[0]
        self.assertTrue(row["home_is_promoted_or_returning"])
        self.assertFalse(row["away_is_promoted_or_returning"])
        self.assertEqual(row["home_season_matches_before"], 0)
        self.assertNotEqual(row["away_elo_before"], 1500.0)

    def test_team_table_and_ios_seed_are_explicitly_non_public(self):
        teams = build_team_table(self.matches)
        self.assertEqual(len(teams), 5)
        predictions = build_chronological_baselines(self.matches)
        seed = build_ios_backtest_seed(predictions)
        self.assertTrue(seed["demo_only"])
        self.assertFalse(seed["publication_eligible"])
        self.assertEqual(len(seed["fixtures"]), 3)
        self.assertTrue(
            all(item["fixture_id"].startswith("fixture-research:") for item in seed["fixtures"])
        )

    def test_report_withholds_sealed_test_metrics(self):
        predictions = build_chronological_baselines(self.matches)
        report = build_research_report(predictions)
        self.assertEqual(report["sealed_test"]["metrics"], "WITHHELD")
        self.assertEqual(report["sealed_test"]["matches"], 1)
        self.assertNotIn("sealed_test", report["metrics"])
        json.dumps(report, allow_nan=False)

    def test_prospective_features_include_all_available_history_without_fake_results(self):
        schedule = pd.DataFrame(
            [
                {
                    "fixture_id": "sportmonks:1",
                    "source_record_key": "sportmonks:fixture:1",
                    "season_key": "2026-27",
                    "kickoff_at": "2026-08-15T18:00:00Z",
                    "home_team_key": "club:A",
                    "home_team_name": "A",
                    "away_team_key": "club:F",
                    "away_team_name": "F",
                }
            ]
        )
        result = build_prospective_baselines(
            self.matches,
            schedule,
            data_cutoff="2026-07-23T18:00:00Z",
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["league_matches_before"], len(self.matches))
        self.assertFalse(result.iloc[0]["home_is_promoted_or_returning"])
        self.assertTrue(result.iloc[0]["away_is_promoted_or_returning"])
        self.assertEqual(result.iloc[0]["home_season_matches_before"], 0)
        self.assertNotIn("outcome_ft", result.columns)
        for columns in PROBABILITY_COLUMNS.values():
            self.assertAlmostEqual(float(result.loc[0, list(columns)].sum()), 1.0)

    def test_prospective_cutoff_cannot_precede_historical_result_availability(self):
        schedule = pd.DataFrame(
            [
                {
                    "fixture_id": "sportmonks:1",
                    "source_record_key": "sportmonks:fixture:1",
                    "season_key": "2026-27",
                    "kickoff_at": "2026-08-15T18:00:00Z",
                    "home_team_key": "club:A",
                    "home_team_name": "A",
                    "away_team_key": "club:F",
                    "away_team_name": "F",
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "availability"):
            build_prospective_baselines(
                self.matches,
                schedule,
                data_cutoff="2025-08-01T19:00:00Z",
            )

    def test_operating_forecast_can_continue_current_season_without_reset(self):
        schedule = pd.DataFrame(
            [
                {
                    "fixture_id": "sportmonks:2",
                    "source_record_key": "sportmonks:fixture:2",
                    "season_key": "2025-26",
                    "kickoff_at": "2025-08-09T18:00:00Z",
                    "home_team_key": "club:E",
                    "home_team_name": "E",
                    "away_team_key": "club:A",
                    "away_team_name": "A",
                }
            ]
        )
        result = build_prospective_baselines(
            self.matches,
            schedule,
            data_cutoff="2025-08-02T00:00:00Z",
            allow_same_season=True,
        )
        self.assertEqual(result.iloc[0]["home_season_matches_before"], 1)
        self.assertEqual(result.iloc[0]["away_season_matches_before"], 1)
        self.assertTrue(result.iloc[0]["home_is_promoted_or_returning"])
        self.assertFalse(result.iloc[0]["away_is_promoted_or_returning"])


if __name__ == "__main__":
    unittest.main()
