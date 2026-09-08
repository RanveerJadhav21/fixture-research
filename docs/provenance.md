# Source and preparation

The source snapshot is Fixture commit
`dfb78e2bb2a30e1bdcdbd1a772567b093f9b2852` from the owner's private production
repository. That history has not been rewritten.

- `src/fixture_models/laliga_baselines.py` and its existing test module are copied
  unchanged, including the original research-only safeguards.
- `independent_features.py` copies eight state/feature definitions unchanged from
  `scripts/122_independent_club_v4.py`. Only the surrounding imports, constants and
  module explanation were assembled for this standalone excerpt.
- The v4 results page is the existing aggregate report, not a new backtest.
- The README, explanatory notes, fictional dataset, demo runner, additional
  independent-feature tests and public CI configuration were prepared with AI
  assistance for this showcase. They are not presented as earlier independent work.

The owner identifies the sports-model research/development as their contribution
and the product frontend as AI-assisted. Repository inspection cannot independently
attribute every original line to a person. No external contributor or endorsement
is invented. A machine-readable source manifest records paths and hashes for the
selected code.

Only newly selected files are published here. The private repository's Git history,
licensed match records, trained model files, infrastructure, account data and
credentials are not part of this repository.
