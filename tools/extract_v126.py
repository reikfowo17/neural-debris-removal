"""Archive and losslessly extract a downloaded v126 notebook into Python stages."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def extract(source, root):
    notebook = json.loads(source.read_text(encoding='utf-8'))
    archive = root / 'notebooks/final'
    stages = root / 'src/neural_debris/stages'
    archive.mkdir(parents=True, exist_ok=True)
    stages.mkdir(parents=True, exist_ok=True)
    destination = archive / 'neural-debris-v126.ipynb'
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)
    metadata = source.with_name('kernel-metadata.json')
    if metadata.exists() and metadata.resolve() != (archive / 'kernel-metadata.json').resolve():
        shutil.copyfile(metadata, archive / 'kernel-metadata.json')
    manifest = {'kernel': 'heon29/neural-debris', 'version': 126,
                'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'environment_cells': [], 'stages': []}
    names = ['configuration', 'detector_data', 'poison_sources', 'synthetic_streaks',
             'legacy_ranker', 'real_anchors', 'contrastive_masks', 'localized_kd',
             'scrub_training', 'crossfit_training', 'hard_gates', 'checkpoint_audit',
             'oof_scoring', 'candidate_audit', 'test_scoring', 'submission', 'export']
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        code = cell['source']
        code = ''.join(code) if isinstance(code, list) else code
        if 'pip install' in code:
            manifest['environment_cells'].append(index)
            (archive / 'environment_setup.py').write_text(code, encoding='utf-8')
            continue
        compile(code, f'cell_{index}', 'exec')
        name = names[len(manifest['stages'])] + '.py'
        (stages / name).write_text(code, encoding='utf-8')
        manifest['stages'].append({'cell_index': index, 'file': name})
    (stages / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook', type=Path)
    args = parser.parse_args()
    extract(args.notebook, Path(__file__).resolve().parents[1])
