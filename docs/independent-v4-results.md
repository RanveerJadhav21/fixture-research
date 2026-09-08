# Independent club v4 report

This challenger uses no bookmaker odds, market prices, Sportmonks predictions,
or post-kickoff information. It combines lagged score, shot, shot-on-target,
corner, venue, rest, dynamic Elo, attack/defence and league-state features.

- Training rows through validation: **81,163**
- Sealed matches: **1,446**

## Validation-locked league configurations

| League | Classifier | Classifier weight | Temperature | Validation log loss |
|---|---|---:|---:|---:|
| La Liga | global-d5-all | 80% | 0.85 | 0.9595 |
| Premier League | global-d6-long | 40% | 1.30 | 0.9951 |
| Serie A | global-d6-long | 100% | 1.05 | 0.9608 |
| Bundesliga | global-d5-all | 100% | 1.15 | 0.9982 |

## Sealed 2025/26 results

| Forecast | Accuracy | Log loss | Brier | ECE |
|---|---:|---:|---:|---:|
| Independent club v4 | 52.14% | 0.9908 | 0.5906 | 0.0148 |
| Fixture league-v1 | 51.11% | 0.9978 | 0.5952 | 0.0238 |
| Pre-close market benchmark | 53.18% | 0.9795 | 0.5830 | — |
| Closing market benchmark | 53.46% | 0.9779 | 0.5820 | — |

## Independent v4 by league

| League | Result accuracy | Exact-score accuracy | Log loss | Brier |
|---|---:|---:|---:|---:|
| La Liga | 51.84% | 14.21% | 0.9777 | 0.5795 |
| Premier League | 48.68% | 11.05% | 1.0286 | 0.6188 |
| Serie A | 53.16% | 12.63% | 0.9841 | 0.5866 |
| Bundesliga | 55.56% | 11.11% | 0.9685 | 0.5743 |
| **Overall** | **52.14%** | **12.31%** | **0.9908** | **0.5906** |

The paired bootstrap gives v4 a 91.78% probability of lower log loss than
Fixture v1, but the 95% interval still crosses zero (-0.0165 to +0.0029).
That is promising shadow evidence, not a valid production-superiority claim.

**Production decision: DO NOT PROMOTE — improvement is not statistically decisive**

The 2025/26 season was not used to choose features, hyperparameters,
calibration or league blends. Existing published forecasts remain immutable.
