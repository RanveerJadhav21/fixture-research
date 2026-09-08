"""Leakage-safe, chronological La Liga baseline models for Fixture.

These baselines are intentionally simple. They establish reproducible reference
points for Week 4 challengers; they are not approved public production models.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

OUTCOMES = ("H", "D", "A")
BASELINE_VERSION = "laliga-baselines-research-v1"
PROBABILITY_COLUMNS = {
    "empirical": ("empirical_home", "empirical_draw", "empirical_away"),
    "elo": ("elo_home", "elo_draw", "elo_away"),
    "poisson": ("poisson_home", "poisson_draw", "poisson_away"),
}

INITIAL_ELO = 1500.0
ELO_HOME_ADVANTAGE = 65.0
ELO_K_FACTOR = 20.0
ELO_SEASON_REVERSION = 0.25
RESULT_AVAILABILITY_LAG_HOURS = 3
EMPIRICAL_PRIOR = 1.0
POISSON_PRIOR_MATCHES = 5.0
INITIAL_HOME_GOALS = 1.40
INITIAL_AWAY_GOALS = 1.10


@dataclass
class TeamState:
    matches: int = 0
    season_matches: int = 0
    goals_for: int = 0
    goals_against: int = 0
    points: int = 0
    home_matches: int = 0
    home_goals_for: int = 0
    home_goals_against: int = 0
    away_matches: int = 0
    away_goals_for: int = 0
    away_goals_against: int = 0
    recent_goals_for: deque[int] = field(default_factory=lambda: deque(maxlen=5))
    recent_goals_against: deque[int] = field(default_factory=lambda: deque(maxlen=5))
    recent_points: deque[int] = field(default_factory=lambda: deque(maxlen=5))


@dataclass
class LeagueState:
    matches: int = 0
    home_goals: int = 0
    away_goals: int = 0
    outcome_counts: dict[str, int] = field(
        default_factory=lambda: {outcome: 0 for outcome in OUTCOMES}
    )


def _points(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def _mean(values: Iterable[int]) -> float:
    values = tuple(values)
    return float(sum(values) / len(values)) if values else 0.0


def _normalize(probabilities: Iterable[float]) -> tuple[float, float, float]:
    values = np.asarray(tuple(probabilities), dtype=float)
    values = np.clip(values, 1e-12, None)
    values /= values.sum()
    return tuple(float(value) for value in values)


def empirical_probabilities(state: LeagueState) -> tuple[float, float, float]:
    denominator = state.matches + EMPIRICAL_PRIOR * len(OUTCOMES)
    return _normalize(
        (state.outcome_counts[outcome] + EMPIRICAL_PRIOR) / denominator
        for outcome in OUTCOMES
    )


def elo_probabilities(
    home_elo: float,
    away_elo: float,
    empirical_draw_probability: float,
) -> tuple[float, float, float]:
    """Convert an Elo expected score into coherent home/draw/away probabilities."""

    expected_home = 1.0 / (
        1.0 + 10.0 ** (-(home_elo + ELO_HOME_ADVANTAGE - away_elo) / 400.0)
    )
    draw_probability = empirical_draw_probability * 4.0 * expected_home * (1.0 - expected_home)
    maximum_draw = max(0.0, 2.0 * min(expected_home, 1.0 - expected_home) - 1e-9)
    draw_probability = min(draw_probability, maximum_draw)
    home_probability = expected_home - 0.5 * draw_probability
    away_probability = 1.0 - expected_home - 0.5 * draw_probability
    return _normalize((home_probability, draw_probability, away_probability))


def _poisson_pmf(value: int, rate: float) -> float:
    return math.exp(-rate) * rate**value / math.factorial(value)


def poisson_outcome_probabilities(
    expected_home_goals: float,
    expected_away_goals: float,
    max_goals: int = 10,
) -> tuple[tuple[float, float, float], tuple[int, int]]:
    grid: list[tuple[int, int, float]] = []
    for home_goals in range(max_goals + 1):
        home_probability = _poisson_pmf(home_goals, expected_home_goals)
        for away_goals in range(max_goals + 1):
            probability = home_probability * _poisson_pmf(away_goals, expected_away_goals)
            grid.append((home_goals, away_goals, probability))
    total = sum(item[2] for item in grid)
    home = sum(item[2] for item in grid if item[0] > item[1]) / total
    draw = sum(item[2] for item in grid if item[0] == item[1]) / total
    away = sum(item[2] for item in grid if item[0] < item[1]) / total
    top = max(grid, key=lambda item: (item[2], -item[0] - item[1]))
    return _normalize((home, draw, away)), (top[0], top[1])


def _smoothed_rate(total: int, matches: int, prior_rate: float) -> float:
    return (total + POISSON_PRIOR_MATCHES * prior_rate) / (
        matches + POISSON_PRIOR_MATCHES
    )


def poisson_rates(
    league: LeagueState,
    home: TeamState,
    away: TeamState,
) -> tuple[float, float]:
    league_home_rate = (league.home_goals + 10.0 * INITIAL_HOME_GOALS) / (
        league.matches + 10.0
    )
    league_away_rate = (league.away_goals + 10.0 * INITIAL_AWAY_GOALS) / (
        league.matches + 10.0
    )

    home_attack = _smoothed_rate(home.home_goals_for, home.home_matches, league_home_rate)
    home_defence = _smoothed_rate(home.home_goals_against, home.home_matches, league_away_rate)
    away_attack = _smoothed_rate(away.away_goals_for, away.away_matches, league_away_rate)
    away_defence = _smoothed_rate(away.away_goals_against, away.away_matches, league_home_rate)

    expected_home = league_home_rate * (home_attack / league_home_rate) * (
        away_defence / league_home_rate
    )
    expected_away = league_away_rate * (away_attack / league_away_rate) * (
        home_defence / league_away_rate
    )
    return float(np.clip(expected_home, 0.10, 4.50)), float(
        np.clip(expected_away, 0.10, 4.50)
    )


def build_team_table(matches: pd.DataFrame) -> pd.DataFrame:
    home = matches[["home_team_key", "home_team_name", "season_key"]].rename(
        columns={"home_team_key": "team_key", "home_team_name": "team_name"}
    )
    away = matches[["away_team_key", "away_team_name", "season_key"]].rename(
        columns={"away_team_key": "team_key", "away_team_name": "team_name"}
    )
    appearances = pd.concat((home, away), ignore_index=True).drop_duplicates()
    names_per_key = appearances.groupby("team_key")["team_name"].nunique()
    if (names_per_key != 1).any():
        raise ValueError("A canonical team key maps to more than one display name")
    return (
        appearances.groupby(["team_key", "team_name"], as_index=False)
        .agg(
            first_season=("season_key", "min"),
            last_season=("season_key", "max"),
            seasons_in_sample=("season_key", "nunique"),
        )
        .sort_values("team_key")
        .reset_index(drop=True)
    )


def _season_team_sets(matches: pd.DataFrame) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for season, frame in matches.groupby("season_key", sort=False):
        result[str(season)] = set(frame["home_team_key"]) | set(frame["away_team_key"])
    return result


def _research_fixture_id(source_record_key: str) -> str:
    digest = hashlib.sha256(source_record_key.encode("utf-8")).hexdigest()[:24]
    return f"fixture-research:{digest}"


def _validate_matches(matches: pd.DataFrame) -> pd.DataFrame:
    required = {
        "season_key",
        "research_split",
        "kickoff_at",
        "home_team_key",
        "home_team_name",
        "away_team_key",
        "away_team_name",
        "home_score_ft",
        "away_score_ft",
        "outcome_ft",
        "source_record_key",
    }
    missing = sorted(required - set(matches.columns))
    if missing:
        raise ValueError(f"Missing canonical match fields: {missing}")
    frame = matches.copy()
    frame["kickoff_at"] = pd.to_datetime(frame["kickoff_at"], utc=True)
    frame = frame.sort_values(["kickoff_at", "source_record_key"]).reset_index(drop=True)
    if frame["source_record_key"].duplicated().any():
        raise ValueError("Duplicate source_record_key values")
    if not set(frame["outcome_ft"]).issubset(set(OUTCOMES)):
        raise ValueError("Unsupported outcome label")
    return frame


def _feature_values(state: TeamState, prefix: str) -> dict[str, float | int]:
    return {
        f"{prefix}_matches_before": state.matches,
        f"{prefix}_season_matches_before": state.season_matches,
        f"{prefix}_goals_for_per_match_before": state.goals_for / state.matches
        if state.matches
        else 0.0,
        f"{prefix}_goals_against_per_match_before": state.goals_against / state.matches
        if state.matches
        else 0.0,
        f"{prefix}_points_per_match_before": state.points / state.matches
        if state.matches
        else 0.0,
        f"{prefix}_last5_goals_for_before": _mean(state.recent_goals_for),
        f"{prefix}_last5_goals_against_before": _mean(state.recent_goals_against),
        f"{prefix}_last5_points_before": _mean(state.recent_points),
    }


def _update_team(state: TeamState, goals_for: int, goals_against: int, venue: str) -> None:
    points = _points(goals_for, goals_against)
    state.matches += 1
    state.season_matches += 1
    state.goals_for += goals_for
    state.goals_against += goals_against
    state.points += points
    state.recent_goals_for.append(goals_for)
    state.recent_goals_against.append(goals_against)
    state.recent_points.append(points)
    if venue == "home":
        state.home_matches += 1
        state.home_goals_for += goals_for
        state.home_goals_against += goals_against
    else:
        state.away_matches += 1
        state.away_goals_for += goals_for
        state.away_goals_against += goals_against


def _elo_margin_multiplier(goal_difference: int) -> float:
    return 1.0 if goal_difference <= 1 else math.log(goal_difference + 1.0)


def _apply_result(
    match_row: pd.Series,
    expected_home: float,
    team_states: defaultdict[str, TeamState],
    ratings: defaultdict[str, float],
    league: LeagueState,
) -> None:
    home_key = str(match_row["home_team_key"])
    away_key = str(match_row["away_team_key"])
    home_goals = int(match_row["home_score_ft"])
    away_goals = int(match_row["away_score_ft"])
    home_actual = (
        1.0
        if home_goals > away_goals
        else 0.5
        if home_goals == away_goals
        else 0.0
    )
    delta = (
        ELO_K_FACTOR
        * _elo_margin_multiplier(abs(home_goals - away_goals))
        * (home_actual - expected_home)
    )
    ratings[home_key] += delta
    ratings[away_key] -= delta
    _update_team(team_states[home_key], home_goals, away_goals, "home")
    _update_team(team_states[away_key], away_goals, home_goals, "away")
    league.matches += 1
    league.home_goals += home_goals
    league.away_goals += away_goals
    league.outcome_counts[str(match_row["outcome_ft"])] += 1


def build_chronological_baselines(matches: pd.DataFrame) -> pd.DataFrame:
    """Create one pre-kickoff feature/prediction row per match.

    All matches sharing a kickoff timestamp are forecast as a batch before any
    result at that timestamp updates state.
    """

    frame = _validate_matches(matches)
    season_teams = _season_team_sets(frame)
    seasons = list(dict.fromkeys(frame["season_key"].astype(str)))
    previous_season_teams: set[str] | None = None
    current_season: str | None = None
    team_states: defaultdict[str, TeamState] = defaultdict(TeamState)
    ratings: defaultdict[str, float] = defaultdict(lambda: INITIAL_ELO)
    league = LeagueState()
    rows: list[dict[str, object]] = []
    latest_included_kickoff_at: pd.Timestamp | None = None
    pending_results: list[tuple[pd.Timestamp, pd.Timestamp, pd.Series, float]] = []

    for kickoff_at, batch in frame.groupby("kickoff_at", sort=True):
        still_pending: list[tuple[pd.Timestamp, pd.Timestamp, pd.Series, float]] = []
        for available_at, result_kickoff, match_row, expected_home in pending_results:
            if available_at <= kickoff_at:
                _apply_result(match_row, expected_home, team_states, ratings, league)
                latest_included_kickoff_at = result_kickoff
            else:
                still_pending.append(
                    (available_at, result_kickoff, match_row, expected_home)
                )
        pending_results = still_pending

        season = str(batch.iloc[0]["season_key"])
        if batch["season_key"].astype(str).nunique() != 1:
            raise ValueError("A kickoff batch cannot span seasons")
        if season != current_season:
            if current_season is not None:
                for team_key in list(ratings):
                    ratings[team_key] = INITIAL_ELO + (
                        ratings[team_key] - INITIAL_ELO
                    ) * (1.0 - ELO_SEASON_REVERSION)
            for state in team_states.values():
                state.season_matches = 0
            previous_season_teams = (
                season_teams[seasons[seasons.index(season) - 1]]
                if season in seasons and seasons.index(season) > 0
                else None
            )
            current_season = season

        for _, match_row in batch.iterrows():
            home_key = str(match_row["home_team_key"])
            away_key = str(match_row["away_team_key"])
            home_state = team_states[home_key]
            away_state = team_states[away_key]
            empirical = empirical_probabilities(league)
            elo = elo_probabilities(ratings[home_key], ratings[away_key], empirical[1])
            expected_home, expected_away = poisson_rates(league, home_state, away_state)
            poisson, scoreline = poisson_outcome_probabilities(expected_home, expected_away)
            row: dict[str, object] = {
                "baseline_version": BASELINE_VERSION,
                "fixture_id": _research_fixture_id(str(match_row["source_record_key"])),
                "source_record_key": match_row["source_record_key"],
                "season_key": season,
                "research_split": match_row["research_split"],
                "kickoff_at": kickoff_at.isoformat(),
                "data_cutoff": latest_included_kickoff_at.isoformat()
                if latest_included_kickoff_at is not None
                else None,
                "home_team_key": home_key,
                "home_team_name": match_row["home_team_name"],
                "away_team_key": away_key,
                "away_team_name": match_row["away_team_name"],
                "home_is_promoted_or_returning": previous_season_teams is not None
                and home_key not in previous_season_teams,
                "away_is_promoted_or_returning": previous_season_teams is not None
                and away_key not in previous_season_teams,
                "league_matches_before": league.matches,
                "league_home_goals_per_match_before": (
                    league.home_goals / league.matches if league.matches else 0.0
                ),
                "league_away_goals_per_match_before": (
                    league.away_goals / league.matches if league.matches else 0.0
                ),
                "home_elo_before": ratings[home_key],
                "away_elo_before": ratings[away_key],
                "elo_home_advantage": ELO_HOME_ADVANTAGE,
                "expected_home_goals": expected_home,
                "expected_away_goals": expected_away,
                "predicted_home_score": scoreline[0],
                "predicted_away_score": scoreline[1],
                "outcome_ft": match_row["outcome_ft"],
                "home_score_ft": int(match_row["home_score_ft"]),
                "away_score_ft": int(match_row["away_score_ft"]),
            }
            row.update(_feature_values(home_state, "home"))
            row.update(_feature_values(away_state, "away"))
            for model_name, probabilities in (
                ("empirical", empirical),
                ("elo", elo),
                ("poisson", poisson),
            ):
                for column, value in zip(PROBABILITY_COLUMNS[model_name], probabilities):
                    row[column] = value
            rows.append(row)
            expected_home = 1.0 / (
                1.0
                + 10.0
                ** (-(ratings[home_key] + ELO_HOME_ADVANTAGE - ratings[away_key]) / 400.0)
            )
            pending_results.append(
                (
                    kickoff_at + pd.Timedelta(hours=RESULT_AVAILABILITY_LAG_HOURS),
                    kickoff_at,
                    match_row,
                    expected_home,
                )
            )

    return pd.DataFrame(rows)


def build_prospective_baselines(
    history: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    data_cutoff: str,
    allow_same_season: bool = False,
) -> pd.DataFrame:
    """Build frozen pre-match features for future fixtures without inventing results.

    Historical results are replayed with the same three-hour availability delay
    used in research. Future fixtures share the state known at ``data_cutoff``;
    they never update one another because their results do not yet exist.
    """

    history_frame = _validate_matches(history)
    required_schedule = {
        "fixture_id",
        "source_record_key",
        "season_key",
        "kickoff_at",
        "home_team_key",
        "home_team_name",
        "away_team_key",
        "away_team_name",
    }
    missing = sorted(required_schedule - set(schedule.columns))
    if missing:
        raise ValueError(f"Missing prospective schedule fields: {missing}")
    future = schedule.copy()
    future["kickoff_at"] = pd.to_datetime(future["kickoff_at"], utc=True)
    future = future.sort_values(["kickoff_at", "source_record_key"]).reset_index(drop=True)
    if future.empty:
        raise ValueError("At least one prospective fixture is required")
    if future["fixture_id"].astype(str).duplicated().any():
        raise ValueError("Duplicate prospective fixture IDs")
    if future["source_record_key"].astype(str).duplicated().any():
        raise ValueError("Duplicate prospective source record keys")
    if future["season_key"].astype(str).nunique() != 1:
        raise ValueError("A prospective run must contain exactly one season")
    if (future["home_team_key"] == future["away_team_key"]).any():
        raise ValueError("A prospective fixture cannot contain the same team twice")

    cutoff = pd.Timestamp(data_cutoff)
    if cutoff.tzinfo is None:
        raise ValueError("data_cutoff must include a timezone")
    cutoff = cutoff.tz_convert("UTC")
    if (future["kickoff_at"] <= cutoff).any():
        raise ValueError("Every prospective fixture must kick off after data_cutoff")

    season_teams = _season_team_sets(history_frame)
    seasons = list(dict.fromkeys(history_frame["season_key"].astype(str)))
    current_season: str | None = None
    team_states: defaultdict[str, TeamState] = defaultdict(TeamState)
    ratings: defaultdict[str, float] = defaultdict(lambda: INITIAL_ELO)
    league = LeagueState()
    latest_included_kickoff_at: pd.Timestamp | None = None
    pending_results: list[tuple[pd.Timestamp, pd.Timestamp, pd.Series, float]] = []

    for kickoff_at, batch in history_frame.groupby("kickoff_at", sort=True):
        still_pending: list[tuple[pd.Timestamp, pd.Timestamp, pd.Series, float]] = []
        for available_at, result_kickoff, match_row, expected_home in pending_results:
            if available_at <= kickoff_at:
                _apply_result(match_row, expected_home, team_states, ratings, league)
                latest_included_kickoff_at = result_kickoff
            else:
                still_pending.append(
                    (available_at, result_kickoff, match_row, expected_home)
                )
        pending_results = still_pending

        season = str(batch.iloc[0]["season_key"])
        if season != current_season:
            if current_season is not None:
                for team_key in list(ratings):
                    ratings[team_key] = INITIAL_ELO + (
                        ratings[team_key] - INITIAL_ELO
                    ) * (1.0 - ELO_SEASON_REVERSION)
            for state in team_states.values():
                state.season_matches = 0
            current_season = season

        for _, match_row in batch.iterrows():
            home_key = str(match_row["home_team_key"])
            away_key = str(match_row["away_team_key"])
            expected_home = 1.0 / (
                1.0
                + 10.0
                ** (-(ratings[home_key] + ELO_HOME_ADVANTAGE - ratings[away_key]) / 400.0)
            )
            pending_results.append(
                (
                    kickoff_at + pd.Timedelta(hours=RESULT_AVAILABILITY_LAG_HOURS),
                    kickoff_at,
                    match_row,
                    expected_home,
                )
            )

    still_pending = []
    for available_at, result_kickoff, match_row, expected_home in pending_results:
        if available_at <= cutoff:
            _apply_result(match_row, expected_home, team_states, ratings, league)
            latest_included_kickoff_at = result_kickoff
        else:
            still_pending.append((available_at, result_kickoff, match_row, expected_home))
    if still_pending:
        raise ValueError("data_cutoff precedes the availability of a historical result")
    if current_season is None or latest_included_kickoff_at is None:
        raise ValueError("Completed historical state is required")

    prospective_season = str(future.iloc[0]["season_key"])
    continuing_current_season = prospective_season == current_season
    if prospective_season in seasons and not (
        allow_same_season and continuing_current_season
    ):
        raise ValueError("Prospective season must not overlap the historical sample")
    if continuing_current_season:
        current_index = seasons.index(current_season)
        previous_season_teams = (
            season_teams[seasons[current_index - 1]]
            if current_index > 0
            else season_teams[current_season]
        )
    else:
        for team_key in list(ratings):
            ratings[team_key] = INITIAL_ELO + (
                ratings[team_key] - INITIAL_ELO
            ) * (1.0 - ELO_SEASON_REVERSION)
        for state in team_states.values():
            state.season_matches = 0
        previous_season_teams = season_teams[current_season]

    rows: list[dict[str, object]] = []
    for _, fixture in future.iterrows():
        home_key = str(fixture["home_team_key"])
        away_key = str(fixture["away_team_key"])
        home_state = team_states[home_key]
        away_state = team_states[away_key]
        empirical = empirical_probabilities(league)
        elo = elo_probabilities(ratings[home_key], ratings[away_key], empirical[1])
        expected_home, expected_away = poisson_rates(league, home_state, away_state)
        poisson, scoreline = poisson_outcome_probabilities(expected_home, expected_away)
        row: dict[str, object] = {
            "baseline_version": BASELINE_VERSION,
            "fixture_id": str(fixture["fixture_id"]),
            "source_record_key": str(fixture["source_record_key"]),
            "season_key": prospective_season,
            "research_split": "prospective",
            "kickoff_at": fixture["kickoff_at"].isoformat(),
            "data_cutoff": cutoff.isoformat(),
            "historical_result_cutoff": latest_included_kickoff_at.isoformat(),
            "home_team_key": home_key,
            "home_team_name": str(fixture["home_team_name"]),
            "away_team_key": away_key,
            "away_team_name": str(fixture["away_team_name"]),
            "home_is_promoted_or_returning": home_key not in previous_season_teams,
            "away_is_promoted_or_returning": away_key not in previous_season_teams,
            "league_matches_before": league.matches,
            "league_home_goals_per_match_before": league.home_goals / league.matches,
            "league_away_goals_per_match_before": league.away_goals / league.matches,
            "home_elo_before": ratings[home_key],
            "away_elo_before": ratings[away_key],
            "elo_home_advantage": ELO_HOME_ADVANTAGE,
            "expected_home_goals": expected_home,
            "expected_away_goals": expected_away,
            "predicted_home_score": scoreline[0],
            "predicted_away_score": scoreline[1],
        }
        row.update(_feature_values(home_state, "home"))
        row.update(_feature_values(away_state, "away"))
        for model_name, probabilities in (
            ("empirical", empirical),
            ("elo", elo),
            ("poisson", poisson),
        ):
            for column, value in zip(PROBABILITY_COLUMNS[model_name], probabilities):
                row[column] = value
        rows.append(row)

    return pd.DataFrame(rows)


def multiclass_metrics(
    rows: pd.DataFrame,
    model_name: str,
) -> dict[str, float | int | None]:
    if rows.empty:
        return {
            "matches": 0,
            "accuracy": None,
            "log_loss": None,
            "brier": None,
            "mean_home_probability": None,
            "mean_draw_probability": None,
            "mean_away_probability": None,
        }
    columns = PROBABILITY_COLUMNS[model_name]
    probabilities = rows[list(columns)].to_numpy(dtype=float)
    targets = np.asarray([OUTCOMES.index(value) for value in rows["outcome_ft"]], dtype=int)
    actual = np.eye(3)[targets]
    chosen = probabilities.argmax(axis=1)
    selected = np.clip(probabilities[np.arange(len(rows)), targets], 1e-15, 1.0)
    return {
        "matches": int(len(rows)),
        "accuracy": float(np.mean(chosen == targets)),
        "log_loss": float(-np.mean(np.log(selected))),
        "brier": float(np.mean(np.sum((probabilities - actual) ** 2, axis=1))),
        "mean_home_probability": float(probabilities[:, 0].mean()),
        "mean_draw_probability": float(probabilities[:, 1].mean()),
        "mean_away_probability": float(probabilities[:, 2].mean()),
    }


def frame_fingerprint(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> str:
    selected = frame[list(columns)].copy() if columns is not None else frame.copy()
    payload = selected.to_csv(index=False, float_format="%.12g", lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_research_report(predictions: pd.DataFrame) -> dict[str, object]:
    visible = predictions[predictions["research_split"] != "sealed_test"].copy()
    sealed = predictions[predictions["research_split"] == "sealed_test"].copy()
    sealed_seasons = sorted(sealed["season_key"].astype(str).unique())
    if len(sealed_seasons) != 1:
        raise ValueError("The sealed-test split must contain exactly one season")
    metrics: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for split in ("development", "validation"):
        split_rows = visible[visible["research_split"] == split]
        metrics[split] = {
            model: multiclass_metrics(split_rows, model) for model in PROBABILITY_COLUMNS
        }
    safe_sealed_columns = [
        "source_record_key",
        "kickoff_at",
        *[column for columns in PROBABILITY_COLUMNS.values() for column in columns],
        "expected_home_goals",
        "expected_away_goals",
    ]
    return {
        "schema_version": "1.0",
        "baseline_version": BASELINE_VERSION,
        "status": "BASELINES_COMPLETE_NOT_PUBLICATION_APPROVED",
        "model_selection_policy": (
            f"Development and validation metrics are visible. The {sealed_seasons[0]} "
            "sealed-test metrics remain withheld until the Week 4 model is frozen."
        ),
        "parameters": {
            "initial_elo": INITIAL_ELO,
            "elo_home_advantage": ELO_HOME_ADVANTAGE,
            "elo_k_factor": ELO_K_FACTOR,
            "elo_season_reversion": ELO_SEASON_REVERSION,
            "result_availability_lag_hours": RESULT_AVAILABILITY_LAG_HOURS,
            "empirical_dirichlet_prior_per_outcome": EMPIRICAL_PRIOR,
            "poisson_prior_matches": POISSON_PRIOR_MATCHES,
            "initial_home_goals": INITIAL_HOME_GOALS,
            "initial_away_goals": INITIAL_AWAY_GOALS,
        },
        "metrics": metrics,
        "sealed_test": {
            "season": sealed_seasons[0],
            "matches": int(len(sealed)),
            "metrics": "WITHHELD",
            "prediction_fingerprint": frame_fingerprint(sealed, safe_sealed_columns),
        },
        "all_prediction_fingerprint": frame_fingerprint(
            predictions,
            ["source_record_key", *[c for cols in PROBABILITY_COLUMNS.values() for c in cols]],
        ),
    }


def build_ios_backtest_seed(predictions: pd.DataFrame, limit: int = 10) -> dict[str, object]:
    """Build an internal-only UI seed that cannot be mistaken for a live publication."""

    validation = predictions[predictions["research_split"] == "validation"].tail(limit)
    fixtures: list[dict[str, object]] = []
    for _, row in validation.iterrows():
        probabilities = {
            "home": round(float(row["poisson_home"]), 6),
            "draw": round(float(row["poisson_draw"]), 6),
            "away": round(float(row["poisson_away"]), 6),
        }
        fixtures.append(
            {
                "fixture_id": row["fixture_id"],
                "kickoff_at": row["kickoff_at"],
                "status": "final",
                "home_team": {"id": row["home_team_key"], "name": row["home_team_name"]},
                "away_team": {"id": row["away_team_key"], "name": row["away_team_name"]},
                "forecast": {
                    "model_version": "laliga-poisson-baseline-research-v1",
                    "probabilities": probabilities,
                    "predicted_scoreline": {
                        "home": int(row["predicted_home_score"]),
                        "away": int(row["predicted_away_score"]),
                    },
                },
                "result": {
                    "home": int(row["home_score_ft"]),
                    "away": int(row["away_score_ft"]),
                    "outcome": row["outcome_ft"],
                },
            }
        )
    return {
        "schema_version": "fixture-backtest-seed/1.0",
        "demo_only": True,
        "publication_eligible": False,
        "source_use": "internal model research and private UI development only",
        "warning": "Historical backtest forecasts; never display as live predictions.",
        "fixtures": fixtures,
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
