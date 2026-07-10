"""Command-line entry points; help and inspection need only Python."""
import argparse
import json

from .config import defaults, load_overrides
from .pipeline import manifest, run, validate_inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description='Neural Debris: reproduce Kaggle v126 (NB51)')
    commands = parser.add_subparsers(dest='command', required=True)
    inspect = commands.add_parser('inspect', help='Display defaults and execution stages without GPU imports')
    inspect.add_argument('--config')
    check = commands.add_parser('check-data', help='Check extracted competition data layout')
    check.add_argument('--data-root', required=True)
    train = commands.add_parser('run', help='Train, audit, score and export all four NB51 candidates')
    train.add_argument('--data-root', required=True)
    train.add_argument('--output-dir', required=True)
    train.add_argument('--config', help='JSON overrides; omit to preserve v126 defaults')
    args = parser.parse_args(argv)
    try:
        if args.command == 'inspect':
            print(json.dumps({'source': manifest(), 'config': defaults() | load_overrides(args.config)}, indent=2))
        elif args.command == 'check-data':
            print(f'Data layout OK: {validate_inputs(args.data_root)}')
        else:
            print(f'Outputs: {run(args.data_root, args.output_dir, args.config)}')
    except (ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, f'Error: {error}\n')
