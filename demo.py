"""Run the original research baselines on explicitly fictional matches."""
import json
from pathlib import Path
import pandas as pd
from src.fixture_models.laliga_baselines import build_chronological_baselines, multiclass_metrics

ROOT = Path(__file__).resolve().parent

def main():
    matches = pd.read_csv(ROOT / 'data/synthetic_matches.csv')
    predictions = build_chronological_baselines(matches)
    output = ROOT / 'demo-output'
    output.mkdir(exist_ok=True)
    predictions.to_csv(output / 'predictions.csv', index=False)
    metrics = {name: multiclass_metrics(predictions, name) for name in ('empirical', 'elo', 'poisson')}
    summary = {'data': 'FICTIONAL demonstration only; not evidence of model performance', 'metrics': metrics}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print('Saved predictions to demo-output/predictions.csv')

if __name__ == '__main__':
    main()
