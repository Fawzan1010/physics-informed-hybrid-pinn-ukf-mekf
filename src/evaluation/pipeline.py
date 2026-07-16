from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import yaml
import numpy as np
import pandas as pd

from src.dynamics.simulator import generate_dataset, Trajectory
from src.dynamics.spacecraft import SpacecraftParams
from src.pinn.train import train_pinn, load_pinn
from src.models.train import train_transformer, load_transformer
from src.utils.reproducibility import set_seed, ensure_dir


@dataclass
class Project:
    root: Path
    config: dict
    output_dir: Path
    data_dir: Path
    model_dir: Path
    figure_dir: Path
    table_dir: Path


def prepare_project(config_path: Path) -> Project:
    raw = yaml.safe_load(config_path.read_text())
    cfg = dict(raw)
    proj = cfg.pop('project', {})
    cfg.update(proj)
    root = config_path.parent.parent
    output_dir = ensure_dir(Path(cfg['output_dir']))
    data_dir = ensure_dir(output_dir / 'data')
    model_dir = ensure_dir(output_dir / 'models')
    figure_dir = ensure_dir(output_dir / 'figures')
    table_dir = ensure_dir(output_dir / 'tables')
    set_seed(int(cfg['seed']))
    return Project(root, cfg, output_dir, data_dir, model_dir, figure_dir, table_dir)


def _load_split(path: Path) -> list[Trajectory]:
    data = np.load(path, allow_pickle=True)
    return list(data['trajectories'])


def generate_synthetic_dataset(project: Project) -> None:
    cfg = project.config
    synth = cfg['synthetic']
    generate_dataset(project.data_dir, cfg, 'train', int(synth['train_trajectories']), int(cfg['seed']))
    generate_dataset(project.data_dir, cfg, 'val', int(synth['val_trajectories']), int(cfg['seed']) + 1000)
    generate_dataset(project.data_dir, cfg, 'test', int(synth['test_trajectories']), int(cfg['seed']) + 2000)
    meta = {'config': cfg}
    (project.output_dir / 'metadata.json').write_text(json.dumps(meta, indent=2, default=str))


def train_models(project: Project):
    train = _load_split(project.data_dir / 'train.npz')
    val = _load_split(project.data_dir / 'val.npz')
    pinn = train_pinn(train, val, project.config, project.model_dir, project.config['device'])
    transformer = train_transformer(train, val, project.config, project.model_dir, project.config['device'])
    return {'pinn': pinn, 'transformer': transformer}
