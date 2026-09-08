# Fixture: football forecasting research

I started Fixture to explore a question I kept coming back to while watching
football: how do you turn what you know about two teams into probabilities you
can actually test?

My focus has been researching and building the sports model. The broader Fixture
project also has a frontend prototyped with AI assistance. This repository is a
small, runnable selection of the modeling work, with a fictional dataset so
someone else can try it without a paid data subscription.

For the frontend side of my work, see my [TypeScript interface projects](https://github.com/RanveerJadhav21/typescript-projects).

## Try it

Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m unittest discover -s tests -v
python demo.py
```

On Windows, activate with `.venv\Scripts\activate` instead. The demo writes
match probabilities and a summary to `demo-output/`. Its 24 matches are entirely
fictional. The numbers demonstrate the pipeline, not forecasting skill.

## What to look at

- [Chronological baselines](src/fixture_models/laliga_baselines.py): empirical
  frequencies, Elo and Poisson score probabilities from the original project.
- [Independent-model feature state](src/fixture_models/independent_features.py):
  rolling form, lagged shots/corners, team Elo and league context, extracted from
  the independent club v4 research script.
- [Tests](tests/): normalization, reproducibility and checks that later or
  overlapping match results cannot change an earlier prediction.
- [Research notes](docs/research-notes.md): model choices, evaluation and limitations.
- [Recorded v4 results](docs/independent-v4-results.md): the original aggregate
  scorecard. This public demo does not retrain v4 or reproduce that scorecard.

## What the research found

The recorded independent v4 evaluation covers 1,446 matches from 2025/26:

| Forecast | Outcome accuracy | Log loss | Brier score |
| --- | ---: | ---: | ---: |
| Independent club v4 | 52.14% | 0.9908 | 0.5906 |
| Earlier Fixture v1 | 51.11% | 0.9978 | 0.5952 |
| Pre-close market benchmark | 53.18% | 0.9795 | 0.5830 |

Lower log loss and Brier score are better. The independent model improved on v1
in this evaluation but did not beat the market benchmark. The reported 95%
interval for its log-loss improvement over v1 crosses zero, so the research
decision was to keep it as a challenger rather than claim a decisive improvement.

Fixture's configured operating forecast is a separate calibrated market-based
system. It should not be confused with the independent model discussed here.

## Scope and provenance

The research modules come from my existing Fixture repository; this is not the
full production app. The public packaging, demo and additional checks were
prepared with AI assistance. See [provenance](docs/provenance.md) for exactly what
was copied and what was added. No model-training data, provider credentials,
user records or production deployment configuration are included.

I have also learned through research and resources on GitHub and Reddit. This
repository makes no claim that community members wrote particular components.
