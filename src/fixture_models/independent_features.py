"""Feature-state excerpt from Fixture independent club v4; see docs/provenance.md.

Input must already be ordered by kickoff with unique match IDs, and each result's
available_at must follow its kickoff. Raw labels remain in the returned research
table; do not train on all columns without an explicit feature allowlist.
"""
from __future__ import annotations
import heapq
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd

STAT_FIELDS = (
    "goals_for",
    "goals_against",
    "goal_diff",
    "points",
    "shots_for",
    "shots_against",
    "shots_on_target_for",
    "shots_on_target_against",
    "corners_for",
    "corners_against",
    "shot_diff",
    "shots_on_target_diff",
    "elo_surprise",
)
WINDOWS = (5, 10, 20)


@dataclass
class TeamState:
    elo: float = 1500.0
    matches: int = 0
    last_kickoff: pd.Timestamp | None = None
    last_season: str | None = None
    results: deque[dict[str, float]] = field(
        default_factory=lambda: deque(maxlen=max(WINDOWS))
    )
    fast: dict[str, float] = field(default_factory=dict)
    slow: dict[str, float] = field(default_factory=dict)

    def enter_season(self, season: str) -> None:
        if self.last_season is None:
            self.last_season = season
            return
        if self.last_season != season:
            self.elo = 1500.0 + 0.72 * (self.elo - 1500.0)
            self.fast = {
                key: 0.65 * value for key, value in self.fast.items()
            }
            self.slow = {
                key: 0.82 * value for key, value in self.slow.items()
            }
            self.last_season = season



@dataclass
class LeagueState:
    matches: int = 0
    home_goals: float = 0.0
    away_goals: float = 0.0
    draws: float = 0.0

    def snapshot(self) -> dict[str, float]:
        denominator = max(self.matches, 1)
        return {
            "league_matches_before": float(self.matches),
            "league_home_goals_mean": self.home_goals / denominator,
            "league_away_goals_mean": self.away_goals / denominator,
            "league_draw_rate": self.draws / denominator,
        }



def _finite(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if np.isfinite(parsed) else float("nan")



def _mean(records: list[dict[str, float]], key: str) -> float:
    values = np.asarray([record.get(key, np.nan) for record in records], float)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if len(finite) else float("nan")



def _snapshot(state: TeamState, kickoff: pd.Timestamp) -> dict[str, float]:
    result: dict[str, float] = {
        "matches_before": float(state.matches),
        "elo": float(state.elo),
        "days_rest": (
            -1.0
            if state.last_kickoff is None
            else max(
                0.0,
                float((kickoff - state.last_kickoff).total_seconds() / 86400.0),
            )
        ),
    }
    all_records = list(state.results)
    for window in WINDOWS:
        records = all_records[-window:]
        for key in STAT_FIELDS:
            result[f"last_{window}_{key}"] = _mean(records, key)
    for speed, values in (("fast", state.fast), ("slow", state.slow)):
        for key in STAT_FIELDS:
            result[f"{speed}_{key}"] = values.get(key, float("nan"))
    home_records = [item for item in all_records[-10:] if item["was_home"] == 1.0]
    away_records = [item for item in all_records[-10:] if item["was_home"] == 0.0]
    for venue, records in (("home", home_records), ("away", away_records)):
        for key in ("goals_for", "goals_against", "points", "shots_on_target_diff"):
            result[f"venue_{venue}_{key}"] = _mean(records, key)
    return result



def _expected_home(home_elo: float, away_elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-((home_elo + 62.0) - away_elo) / 400.0))



def _release(match: dict[str, Any], teams: defaultdict[str, TeamState], league: LeagueState) -> None:
    home_key = f"{match['competition_key']}:{match['home_team_id']}"
    away_key = f"{match['competition_key']}:{match['away_team_id']}"
    home = teams[home_key]
    away = teams[away_key]
    expected_home = _expected_home(home.elo, away.elo)
    if match["home_goals"] > match["away_goals"]:
        actual_home, home_points, away_points = 1.0, 3.0, 0.0
    elif match["home_goals"] < match["away_goals"]:
        actual_home, home_points, away_points = 0.0, 0.0, 3.0
    else:
        actual_home, home_points, away_points = 0.5, 1.0, 1.0
    margin = max(abs(int(match["home_goals"]) - int(match["away_goals"])), 1)
    elo_delta = 28.0 * (1.0 + 0.30 * math.log(margin)) * (actual_home - expected_home)
    home.elo += elo_delta
    away.elo -= elo_delta

    hs = _finite(match.get("home_shots"))
    aws = _finite(match.get("away_shots"))
    hst = _finite(match.get("home_shots_on_target"))
    ast = _finite(match.get("away_shots_on_target"))
    hc = _finite(match.get("home_corners"))
    ac = _finite(match.get("away_corners"))

    def record(
        *,
        goals_for: int,
        goals_against: int,
        points: float,
        shots_for: float,
        shots_against: float,
        sot_for: float,
        sot_against: float,
        corners_for: float,
        corners_against: float,
        was_home: float,
        surprise: float,
    ) -> dict[str, float]:
        return {
            "goals_for": float(goals_for),
            "goals_against": float(goals_against),
            "goal_diff": float(goals_for - goals_against),
            "points": points,
            "shots_for": shots_for,
            "shots_against": shots_against,
            "shots_on_target_for": sot_for,
            "shots_on_target_against": sot_against,
            "corners_for": corners_for,
            "corners_against": corners_against,
            "shot_diff": shots_for - shots_against,
            "shots_on_target_diff": sot_for - sot_against,
            "elo_surprise": surprise,
            "was_home": was_home,
        }

    home_record = record(
        goals_for=int(match["home_goals"]), goals_against=int(match["away_goals"]),
        points=home_points, shots_for=hs, shots_against=aws, sot_for=hst,
        sot_against=ast, corners_for=hc, corners_against=ac, was_home=1.0,
        surprise=actual_home - expected_home,
    )
    away_record = record(
        goals_for=int(match["away_goals"]), goals_against=int(match["home_goals"]),
        points=away_points, shots_for=aws, shots_against=hs, sot_for=ast,
        sot_against=hst, corners_for=ac, corners_against=hc, was_home=0.0,
        surprise=(1.0 - actual_home) - (1.0 - expected_home),
    )
    for state, item in ((home, home_record), (away, away_record)):
        state.results.append(item)
        state.matches += 1
        state.last_kickoff = match["kickoff_at"]
        for key in STAT_FIELDS:
            value = item[key]
            if not np.isfinite(value):
                continue
            state.fast[key] = (
                value if key not in state.fast else 0.32 * value + 0.68 * state.fast[key]
            )
            state.slow[key] = (
                value if key not in state.slow else 0.14 * value + 0.86 * state.slow[key]
            )
    league.matches += 1
    league.home_goals += float(match["home_goals"])
    league.away_goals += float(match["away_goals"])
    league.draws += float(match["home_goals"] == match["away_goals"])



def build_feature_table(matches: list[dict[str, Any]]) -> pd.DataFrame:
    teams: defaultdict[str, TeamState] = defaultdict(TeamState)
    leagues: defaultdict[str, LeagueState] = defaultdict(LeagueState)
    pending: list[tuple[pd.Timestamp, str, dict[str, Any]]] = []
    rows: list[dict[str, Any]] = []
    for match in matches:
        kickoff = pd.Timestamp(match["kickoff_at"])
        while pending and pending[0][0] <= kickoff:
            _, _, released = heapq.heappop(pending)
            _release(released, teams, leagues[released["competition_key"]])
        home_uid = f"{match['competition_key']}:{match['home_team_id']}"
        away_uid = f"{match['competition_key']}:{match['away_team_id']}"
        home = teams[home_uid]
        away = teams[away_uid]
        home.enter_season(str(match["season_code"]))
        away.enter_season(str(match["season_code"]))
        home_snapshot = _snapshot(home, kickoff)
        away_snapshot = _snapshot(away, kickoff)
        row = dict(match)
        row["date"] = kickoff
        row["year"] = kickoff.year
        row["home_team_uid"] = home_uid
        row["away_team_uid"] = away_uid
        row["elo_expected_home"] = _expected_home(home.elo, away.elo)
        row["elo_diff"] = home.elo - away.elo
        row.update(leagues[match["competition_key"]].snapshot())
        for prefix, snapshot in (("home", home_snapshot), ("away", away_snapshot)):
            row.update({f"{prefix}_{key}": value for key, value in snapshot.items()})
        comparable = sorted(set(home_snapshot) & set(away_snapshot))
        for key in comparable:
            row[f"diff_{key}"] = home_snapshot[key] - away_snapshot[key]
        row["attack_defence_fast_home"] = (
            home_snapshot["fast_goals_for"] - away_snapshot["fast_goals_against"]
        )
        row["attack_defence_fast_away"] = (
            away_snapshot["fast_goals_for"] - home_snapshot["fast_goals_against"]
        )
        row["sot_matchup_home"] = (
            home_snapshot["fast_shots_on_target_for"]
            - away_snapshot["fast_shots_on_target_against"]
        )
        row["sot_matchup_away"] = (
            away_snapshot["fast_shots_on_target_for"]
            - home_snapshot["fast_shots_on_target_against"]
        )
        rows.append(row)
        heapq.heappush(
            pending,
            (pd.Timestamp(match["available_at"]), str(match["match_id"]), match),
        )
    return pd.DataFrame(rows).sort_values(["date", "match_id"]).reset_index(drop=True)
