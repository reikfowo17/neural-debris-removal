import importlib.util
import hashlib
import json
from pathlib import Path
import unittest
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


class ExtractionTests(unittest.TestCase):
    def test_executable_cells_preserved(self):
        notebook_path = ROOT / 'notebooks/final/neural-debris-v126.ipynb'
        self.assertTrue(notebook_path.exists(), 'Version 126 source must be archived')
        notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
        manifest = json.loads((ROOT / 'src/neural_debris/stages/manifest.json').read_text())
        self.assertEqual(manifest['version'], 126)
        self.assertEqual(manifest['sha256'], hashlib.sha256(notebook_path.read_bytes()).hexdigest())
        for stage in manifest['stages']:
            cell = notebook['cells'][stage['cell_index']]
            source = cell['source']
            if isinstance(source, list):
                source = ''.join(source)
            extracted = (ROOT / 'src/neural_debris/stages' / stage['file']).read_text(encoding='utf-8')
            self.assertEqual(source.rstrip(), extracted.rstrip())
            compile(extracted, stage['file'], 'exec')
        code_indices = [i for i, c in enumerate(notebook['cells']) if c['cell_type'] == 'code']
        self.assertEqual(code_indices, sorted([s['cell_index'] for s in manifest['stages']] + manifest['environment_cells']))

class RunnerTests(unittest.TestCase):
    def test_configuration_injected_before_seed(self):
        from neural_debris.pipeline import prepare_source
        code = 'CFG = dict(SEED=42)\nSEED = CFG["SEED"]\nWORK = Path("/kaggle/working")\n'
        namespace = {'Path': Path, '_overrides': {'SEED': 7}, '_output_dir': '/tmp/result'}
        exec(compile(prepare_source(code, 'configuration'), '<test>', 'exec'), namespace)
        self.assertEqual(namespace['SEED'], 7)
        self.assertEqual(namespace['WORK'], Path('/tmp/result'))

    def test_data_root_is_explicit(self):
        from neural_debris.pipeline import prepare_source
        code = 'CANDIDATE_ROOTS = ["/kaggle/input/a"]\nDATA_ROOT = CANDIDATE_ROOTS[0]\n'
        namespace = {'_data_root': '/tmp/data'}
        exec(compile(prepare_source(code, 'detector_data'), '<test>', 'exec'), namespace)
        self.assertEqual(namespace['DATA_ROOT'], '/tmp/data')

    def test_stage_functions_share_state(self):
        from neural_debris.pipeline import execute_stage
        namespace = {}
        execute_stage('x = 3\ndef value():\n    return x\n', 'first', namespace)
        execute_stage('x = 8\nresult = value()\n', 'second', namespace)
        self.assertEqual(namespace['result'], 8)

    def test_invalid_config_is_rejected(self):
        from neural_debris.config import load_overrides
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text('{"SEED": "bad"}')
            with self.assertRaises(ValueError):
                load_overrides(path)

class SubmissionTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from neural_debris.config import defaults
        from neural_debris.pipeline import execute_stage
        self.np = np
        self.namespace = {'np': np, 'CFG': defaults(), 'test_files': [Path('a.png'), Path('b.png')]}
        code = (ROOT / 'src/neural_debris/stages/submission.py').read_text(encoding='utf-8')
        execute_stage(code, 'submission', self.namespace)

    def test_exact_k_with_per_image_cap_and_stable_ties(self):
        np = self.np
        cache = {name: {'scores': np.array([.8, .7, .6])} for name in ('a', 'b')}
        scores = {name: np.array([.1, .1, .1]) for name in cache}
        selected, count, boundary = self.namespace['calibrate_exact_keep_set'](cache, scores, 3)
        self.assertEqual(selected, {'a': {0, 1}, 'b': {0}})
        self.assertEqual(count, 3)
        self.assertAlmostEqual(boundary, .1)

    def test_export_preserves_boxes_and_confidence_floor(self):
        import pandas as pd
        np = self.np
        with tempfile.TemporaryDirectory() as directory:
            sample = Path(directory) / 'sample.csv'
            sample.write_text('image_id,prediction_string\n1,\n2,\n')
            self.namespace.update({'pd': pd, 'SAMPLE_SUB': sample})
            cache = {'1': {'scores': np.array([.8, .4, .05]),
                           'boxes': np.array([[10, 20, 30, 50], [1, 2, 4, 6], [0, 0, 1, 1]])}}
            output = Path(directory) / 'submission.csv'
            count, mass = self.namespace['compose_exact_submission'](cache, {'1': {0}}, output)
            result = pd.read_csv(output, dtype={'image_id': str}, keep_default_na=False)
            self.assertEqual(count, 1)
            self.assertAlmostEqual(mass, .95, places=6)
            self.assertEqual(result.iloc[0]['prediction_string'],
                             '0.950000 10.00 20.00 20.00 30.00 0.020000 1.00 2.00 3.00 4.00')
            self.assertEqual(result.iloc[1]['prediction_string'], ' ')

    def test_unknown_config_is_rejected(self):
        from neural_debris.config import load_overrides
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text('{"TYPO": 1}')
            with self.assertRaises(ValueError):
                load_overrides(path)


if __name__ == '__main__':
    unittest.main()
