"""Execute faithful Python stages in the original notebook dependency order.

The shared namespace is intentional: notebook functions resolve globals populated
by later stages. Independent module imports would silently change those bindings.
Only path assignments and configuration overrides are adapted at execution time.
"""
import ast
import json
import platform
import sys
import time
from pathlib import Path

from .config import STAGES, defaults, load_overrides


def prepare_source(source, stage):
    tree = ast.parse(source)
    body = []
    for node in tree.body:
        name = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if stage == 'configuration' and name == 'WORK':
            node.value = ast.parse('Path(_output_dir)', mode='eval').body
        if stage == 'detector_data' and name == 'CANDIDATE_ROOTS':
            node.value = ast.parse('[_data_root]', mode='eval').body
        body.append(node)
        if stage == 'configuration' and name == 'CFG':
            body.extend(ast.parse('CFG.update(_overrides)').body)
    tree.body = body
    return ast.fix_missing_locations(tree)


def execute_stage(source, stage, namespace):
    code = prepare_source(source, stage)
    exec(compile(code, f'<neural_debris/{stage}>', 'exec'), namespace)


def manifest():
    return json.loads((STAGES / 'manifest.json').read_text(encoding='utf-8'))


def validate_inputs(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f'Data directory not found: {root}')
    for pattern in ('*.pth', 'annotations_coco.json', 'sample_submission.csv', '*.png'):
        if not next(root.rglob(pattern), None):
            raise ValueError(f'Missing {pattern} beneath {root}; extract competition data first')
    return root


def run(data_root, output_dir, config=None):
    root = validate_inputs(data_root)
    overrides = load_overrides(config)
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Use an empty output directory: {output}')
    # Dependency check occurs before creating outputs or running any training.
    import torch
    import detectron2
    if not torch.cuda.is_available():
        raise RuntimeError('Full reproduction requires CUDA; use inspect for CPU-only inspection')
    torch.ones(1, device='cuda').sum().item()
    output.mkdir(parents=True, exist_ok=True)
    namespace = {'__name__': 'neural_debris_v126', '_data_root': str(root),
                 '_output_dir': str(output), '_overrides': overrides}
    provenance = {'kernel': manifest()['kernel'], 'version': 126,
                  'source_sha256': manifest()['sha256'], 'data_root': str(root),
                  'config': defaults() | overrides, 'python': sys.version,
                  'platform': platform.platform(), 'torch': torch.__version__,
                  'detectron2': getattr(detectron2, '__version__', 'unknown'),
                  'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
                  'completed_stages': [], 'status': 'running'}
    record = output / 'reproduction.json'

    def save():
        record.write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')

    save()
    try:
        for item in manifest()['stages']:
            path = STAGES / item['file']
            print(f'\n=== {path.stem} ===', flush=True)
            start = time.monotonic()
            execute_stage(path.read_text(encoding='utf-8'), path.stem, namespace)
            provenance['completed_stages'].append({'name': path.stem, 'seconds': time.monotonic() - start})
            save()
        provenance['status'] = 'complete'
    except BaseException as error:
        provenance['status'] = 'failed'
        provenance['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        save()
    return output
