"""Shared settings and console progress."""
from pathlib import Path
import logging
import yaml

METRICS = ['MCC', 'DA', 'Precision', 'Recall', 'F1', 'AUC']
MODEL_NAMES = ['LogisticRegression', 'MLP', 'RandomForest', 'SVC', 'XGBoost']


def load_config(path):
    path = Path(path).expanduser().resolve()
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    for key, value in config['paths'].items():
        target = Path(value).expanduser()
        config['paths'][key] = str(target if target.is_absolute() else path.parent / target)
    checkpoint = config['text'].get('checkpoint')
    if checkpoint:
        target = Path(checkpoint).expanduser()
        config['text']['checkpoint'] = str(target if target.is_absolute() else path.parent / target)
    return config


def progress_logger():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    return logging.getLogger('systems')
