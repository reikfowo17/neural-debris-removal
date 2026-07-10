"""Read notebook defaults without importing GPU dependencies."""
import ast
import json
import math
from pathlib import Path

STAGES = Path(__file__).parent / 'stages'


def defaults():
    tree = ast.parse((STAGES / 'configuration.py').read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CFG' for t in node.targets):
            return {keyword.arg: ast.literal_eval(keyword.value) for keyword in node.value.keywords}
    raise RuntimeError('Notebook CFG not found')


def load_overrides(path=None):
    values = {} if path is None else json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(values, dict):
        raise ValueError('Configuration must be a JSON object')
    base = defaults()
    for key, value in values.items():
        if key not in base:
            raise ValueError(f'Unknown configuration key: {key}')
        expected = type(base[key])
        valid = type(value) is expected or (expected is float and type(value) is int)
        if not valid or (type(value) in (int, float) and not math.isfinite(value)):
            raise ValueError(f'{key} requires a finite {expected.__name__}')
        if key != 'SEED' and type(value) in (int, float) and value <= 0:
            raise ValueError(f'{key} must be positive')
        if key == 'N_FOLDS' and value < 2:
            raise ValueError('N_FOLDS must be at least 2')
        if key == 'SEED' and not 0 <= value < 2 ** 32:
            raise ValueError('SEED must be in [0, 2**32)')
    return values
