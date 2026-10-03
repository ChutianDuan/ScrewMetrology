#!/usr/bin/env python3
"""Reproduce synthetic validation, frozen representatives, and full MVTec evaluation."""
import argparse
import csv
import json
from pathlib import Path
import subprocess


def run(*args):
    print('+', *map(str, args), flush=True)
    subprocess.run(list(map(str, args)), check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, default=Path('build'))
    parser.add_argument('--dataset', type=Path, default=Path('data/mvtec_screws'))
    parser.add_argument('--output', type=Path, default=Path('result'))
    parser.add_argument('--skip-full', action='store_true')
    args = parser.parse_args()
    exe = args.build_dir / 'screw_metrology'
    output, dataset = args.output, args.dataset
    run('ctest', '--test-dir', args.build_dir, '--output-on-failure')
    run(args.build_dir / 'generate_samples', 'data/synthetic')
    run(exe, 'measure', '--input', 'data/synthetic/images', '--config', 'config/synthetic.yaml',
        '--mode', 'single', '--recipe', 'config/example_recipe_px.yaml', '--output', output / 'synthetic')
    run(exe, 'evaluate', '--results', output / 'synthetic', '--truth', 'data/synthetic/truth.csv',
        '--output', output / 'synthetic_evaluation')
    run(exe, 'calibrate-scale', '--input', 'data/synthetic/images/rotation_0.png',
        '--p1', '100,100', '--p2', '300,100', '--mm', '20', '--output', output / 'synthetic_scale.yaml')
    run(exe, 'measure', '--input', 'data/synthetic/images', '--config', 'config/synthetic.yaml',
        '--scale', output / 'synthetic_scale.yaml', '--mode', 'single',
        '--recipe', 'config/example_recipe_mm.yaml', '--output', output / 'synthetic_mm')
    mm_truth = output / 'synthetic_truth_mm.csv'
    with open('data/synthetic/truth.csv', newline='') as source, mm_truth.open('w', newline='') as dest:
        reader = csv.DictReader(source); writer = csv.DictWriter(dest, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            row['truth'] = str(float(row['truth']) * 0.1); row['unit'] = 'mm'; writer.writerow(row)
    run(exe, 'evaluate', '--results', output / 'synthetic_mm', '--truth', mm_truth,
        '--output', output / 'synthetic_mm_evaluation')
    representative = output / 'representative_input'
    representative.mkdir(parents=True, exist_ok=True)
    names = Path('config/representative.txt').read_text().splitlines()
    for name in names:
        path = representative / name
        source = (dataset / 'images' / name).resolve()
        if not source.is_file():
            raise RuntimeError('Missing real dataset: ' + str(source))
        if path.exists() and path.resolve() != source:
            raise RuntimeError('Representative path already points to other data: ' + str(path))
        if not path.exists():
            path.symlink_to(source)
    actual = {p.name for p in representative.iterdir() if p.is_file()}
    if actual != set(names):
        raise RuntimeError('Representative directory differs from frozen manifest')
    comparison = {}
    for method in ['otsu', 'adaptive', 'lab']:
        folder = output / 'comparison_frozen' / method
        run(exe, 'measure', '--input', representative, '--config', 'config/measurement.yaml',
            '--segmentation', method, '--output', folder)
        report_folder = folder.parent / (method + '_evaluation')
        run(exe, 'evaluate', '--results', folder, '--annotations', dataset / 'mvtec_screws.json',
            '--output', report_folder)
        report = json.loads((report_folder / 'report.json').read_text())
        comparison[method] = {key: report[key] for key in ['tp', 'fp', 'fn', 'recall', 'precision']}
    (output / 'comparison_frozen' / 'summary.json').write_text(json.dumps(comparison, indent=2) + '\n')
    if not args.skip_full:
        run(exe, 'measure', '--input', dataset / 'images', '--config', 'config/measurement.yaml',
            '--output', output / 'mvtec_full')
        run(exe, 'evaluate', '--results', output / 'mvtec_full', '--annotations', dataset / 'mvtec_screws.json',
            '--output', output / 'mvtec_evaluation')


if __name__ == '__main__':
    main()
