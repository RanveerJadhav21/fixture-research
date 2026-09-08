# Research notes

## Start with a baseline

The empirical model uses past home/draw/away frequencies. Elo represents relative
team strength and includes home advantage. The Poisson baseline estimates home
and away scoring rates, then sums a score grid into three outcome probabilities.
These give a more useful comparison than judging a complicated model in isolation.

## Only use information available at prediction time

The baseline delays result updates by three hours after kickoff and forecasts
simultaneous matches together. The independent feature excerpt uses each record's
explicit `available_at` timestamp. It updates team state only after that time.
The time delay is a modeling assumption, not proof that every provider field was
actually published at that moment.

The feature excerpt expects matches already ordered by kickoff, with unique IDs
and result availability later than kickoff. Its returned table includes raw
labels alongside derived features for research; never feed the entire table to
a classifier. The full project selects feature columns separately.

## Independent club v4

The original research uses lagged goals, shots, shots on target, corners, rest,
Elo, venue and league state. Its classifier and goal-model blends were selected
on the 2024/25 validation season, with 2025/26 reserved for the recorded evaluation.
The full CatBoost training procedure and fitted weights are not included here.

The public feature excerpt is not a standalone v4 predictor. Its purpose is to
make the feature timing and state updates inspectable without licensed data.

## Why accuracy is not enough

Accuracy counts the most likely outcome. Log loss measures the probability given
to the actual result and penalizes confident mistakes. The multiclass Brier score
sums squared errors over all three outcomes (the convention used here ranges
from 0 to 2). A model should be compared against simple baselines and market
probabilities on the same evaluation matches.

## Limits of the recorded result

The scorecard is a retained research artifact, not an independently reproduced
backtest in this public repository. It does not establish profitability, future
performance, or superiority over bookmaker probabilities. Feature selection,
repeated experiments and data timing all matter when interpreting an apparent
improvement. The synthetic demo cannot validate those real-world claims.
