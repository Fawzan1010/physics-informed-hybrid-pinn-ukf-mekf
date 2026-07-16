from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.experiments import run_all_experiments
from src.evaluation.pipeline import prepare_project, train_models, generate_synthetic_dataset
from src.visualization.plots import make_all_plots


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Spacecraft hybrid PINN + UKF + MEKF benchmark')
    p.add_argument('--config', type=str, default='configs/default.yaml')
    p.add_argument('--mode', type=str, default='all',
                   choices=['synth', 'train', 'evaluate', 'plot', 'ablate', 'theory', 'all'])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    project = prepare_project(Path(args.config))
    if args.mode in {'synth', 'all'}:
        generate_synthetic_dataset(project)
    if args.mode in {'train', 'all'}:
        train_models(project)
    if args.mode in {'evaluate', 'all'}:
        results = run_all_experiments(project)
        (project.output_dir / 'results.json').write_text(json.dumps(results, indent=2))
    if args.mode in {'plot', 'all'}:
        make_all_plots(project)
    if args.mode == 'ablate':
        from src.evaluation.ablation import run_ablation
        run_ablation(project)
    if args.mode in {'theory', 'all'}:
        from src.evaluation.theory import run_theory_analysis
        run_theory_analysis(project)


if __name__ == '__main__':
    main()
