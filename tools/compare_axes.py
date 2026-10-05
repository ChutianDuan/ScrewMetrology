#!/usr/bin/env python3
"""Frozen 39-image synthetic + 16-image real axis comparison (Python standard library only)."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('pca', 'fitline', 'rect')
DIMENSIONS = ('length', 'width', 'head_length', 'head_width', 'shaft_width')
SLOTS = ('pixel', 'subpixel', 'preferred_px')


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def angle_error(first, second):
    difference = abs((first - second) % 180)
    return min(difference, 180 - difference)


def stats(values):
    values = list(values)
    if not values:
        return dict(count=0, mean=None, mae=None, max_abs=None, std=None)
    return dict(count=len(values), mean=statistics.mean(values),
                mae=statistics.mean(abs(value) for value in values),
                max_abs=max(abs(value) for value in values),
                std=statistics.stdev(values) if len(values) > 1 else None)


def paired_stats(samples):
    """Own-valid and intersection cohorts; missing samples never become zero errors."""
    common = set.intersection(*(set(samples[method]) for method in METHODS))
    return {method: dict(own_valid=stats(samples[method].values()),
                         common_valid=stats(samples[method][key] for key in sorted(common)))
            for method in METHODS}


def load_cohort(folder):
    manifest = read_jsonl(folder / 'manifest.jsonl')
    expected = {row['image'] for row in manifest}
    if len(expected) != len(manifest):
        raise RuntimeError('Duplicate comparison image')
    results = {}
    for method in METHODS:
        records = [read_json(path) for path in (folder / 'methods' / method).glob('*.json')]
        results[method] = {Path(record['image']).name: record for record in records}
        if len(results[method]) != len(records) or set(results[method]) != expected:
            raise RuntimeError('Result inventory differs from manifest: ' + method)
    # Assert fixed candidates also give identical IDs/centers/boxes; errors remain explicit.
    for row in manifest:
        records = [results[method][row['image']] for method in METHODS]
        if any(record['error'] for record in records):
            continue
        reference = records[0]['objects']
        if len(reference) != row['candidate_count']:
            raise RuntimeError('Candidate count changed: ' + row['image'])
        for record in records[1:]:
            signatures = lambda objects: [
                [obj[field] for field in ('id', 'center', 'rotated_box',
                                         'comparison_angles_deg', 'eigen_ratio', 'solidity')]
                for obj in objects]
            if signatures(reference) != signatures(record['objects']):
                raise RuntimeError('Geometry/IDs changed between methods: ' + row['image'])
        configs = [{key: value for key, value in record['config'].items() if key != 'axis'}
                   for record in records]
        if any(config != configs[0] for config in configs):
            raise RuntimeError('More than axis changed: ' + row['image'])
    return results


def coverage(records):
    objects = [obj for record in records.values() for obj in record['objects']]
    reliable = [obj for obj in objects if obj['reliable']]
    return dict(images=len(records), failed_images=sum(bool(r['error']) for r in records.values()),
                objects=len(objects), reliable=len(reliable), unreliable=len(objects) - len(reliable),
                reliable_fraction=len(reliable) / len(objects) if objects else None,
                subpixel_status=dict(Counter(obj['subpixel_status'] for obj in objects)),
                issues=dict(Counter(issue for obj in objects for issue in obj['quality_issues'])),
                missing_preferred={name: sum(obj['preferred_px'][name] is None for obj in objects)
                                   for name in DIMENSIONS},
                subpixel_available={name: sum(obj['subpixel'][name] is not None for obj in objects)
                                    for name in DIMENSIONS},
                reliable_subpixel_available={name: sum(obj['subpixel'][name] is not None
                                                       for obj in reliable) for name in DIMENSIONS})


def get_object(records, image, target_id):
    record = records.get(image)
    if record is None or record['error']:
        return None
    return next((obj for obj in record['objects'] if obj['id'] == target_id), None)


def dimension_samples(records, truth_rows, slot, name):
    samples, excluded = {}, Counter()
    for row in truth_rows:
        if row['measurement'] != name:
            continue
        image, target = row['image'], int(row['target_id'])
        obj = get_object(records, image, target)
        if image not in records:
            excluded['missing_result'] += 1
        elif records[image]['error']:
            excluded['processing_error'] += 1
        elif obj is None:
            excluded['missing_target'] += 1
        elif not obj['reliable']:
            excluded['unreliable_geometry'] += 1
        elif obj[slot][name] is None:
            excluded['unavailable_dimension'] += 1
        else:
            samples[(image, target)] = obj[slot][name] - float(row['truth'])
    return samples, dict(excluded)


def synthetic_summary(results, truth_file, axes_file):
    with truth_file.open(newline='') as file:
        truth = list(csv.DictReader(file))
    with axes_file.open(newline='') as file:
        axes = list(csv.DictReader(file))
    if {row['image'] for row in axes} != set(results['pca']):
        raise RuntimeError('Synthetic axis truth differs from image inventory')
    if len({(row['image'], row['target_id']) for row in axes}) != len(axes):
        raise RuntimeError('Duplicate axis truth')
    expected = {(row['image'], row['target_id'], name) for row in axes for name in DIMENSIONS}
    if {(row['image'], row['target_id'], row['measurement']) for row in truth} != expected \
            or len(truth) != len(expected) or any(row['unit'] != 'px' for row in truth):
        raise RuntimeError('Synthetic dimension truth differs from axis truth')
    angle_samples = {method: {} for method in METHODS}
    reliable_angles = {method: {} for method in METHODS}
    details = []
    for row in axes:
        key = (row['image'], int(row['target_id']))
        detail = dict(image=key[0], target_id=key[1], truth_angle_deg=float(row['angle_deg']))
        for method in METHODS:
            obj = get_object(results[method], *key)
            error = angle_error(obj['angle_deg'], detail['truth_angle_deg']) if obj else None
            detail[method] = error
            if obj:
                angle_samples[method][key] = error
                if obj['reliable']:
                    reliable_angles[method][key] = error
        details.append(detail)
    dimensions = {}
    for name in DIMENSIONS:
        dimensions[name] = {}
        for slot in SLOTS:
            samples, excluded = {}, {}
            for method in METHODS:
                samples[method], excluded[method] = dimension_samples(results[method], truth, slot, name)
            dimensions[name][slot] = dict(statistics=paired_stats(samples), excluded=excluded,
                                         truth_count=sum(row['measurement'] == name for row in truth))
    # Repeats are simulated image noise, not repeated physical acquisition.
    repeatability = {}
    for method in METHODS:
        repeatability[method] = {}
        for name in DIMENSIONS:
            values = [obj['preferred_px'][name] for image in sorted(results[method])
                      if image.startswith('repeat_')
                      for obj in results[method][image]['objects']
                      if obj['id'] == 1 and obj['reliable'] and obj['preferred_px'][name] is not None]
            repeatability[method][name] = stats(values)
    # Separate construction conditions so the 30 noise repeats do not hide the nine distinct settings.
    groups = {row['image']: row['part_id'] for row in truth}
    grouped_angles = {}
    for group in sorted(set(groups.values())):
        grouped_angles[group] = paired_stats({method: {key: value for key, value in samples.items()
                                                       if groups[key[0]] == group}
                                              for method, samples in angle_samples.items()})
    return dict(direction_all_detected=paired_stats(angle_samples),
                direction_reliable=paired_stats(reliable_angles), angle_truth_count=len(axes),
                direction_by_construction=grouped_angles,
                dimensions=dimensions, simulated_repeatability=repeatability), details


def real_summary(results, folder):
    evaluations = {method: read_jsonl(folder / 'evaluation' / method / 'images.jsonl')
                   for method in METHODS}
    pairs, angle_samples, reliable_angles = {}, {}, {}
    near_round, missed = {}, {}
    for method in METHODS:
        pairs[method], angle_samples[method], reliable_angles[method] = {}, {}, {}
        near_round[method] = 0
        for row in evaluations[method]:
            for match in row['matches']:
                key = (row['image'], match['target_id'], match['annotation_id'])
                pairs[method][key] = match['iou']
                if match['angle_error_deg'] is None:
                    near_round[method] += 1
                    continue
                angle_samples[method][key] = match['angle_error_deg']
                obj = get_object(results[method], row['image'], match['target_id'])
                if obj and obj['reliable']:
                    reliable_angles[method][key] = match['angle_error_deg']
        missed[method] = sum(len(row['missed_annotation_ids']) for row in evaluations[method])
    # Known processing failures reduce own coverage; they are not a geometry mismatch.
    successful_images = {image for image in results['pca']
                         if all(not results[method][image]['error'] for method in METHODS)}
    comparable_pairs = {method: {key: value for key, value in pairs[method].items()
                                 if key[0] in successful_images} for method in METHODS}
    if any(comparable_pairs[method] != comparable_pairs['pca'] for method in METHODS):
        raise RuntimeError('IoU match identities changed between methods')
    all_angle_keys = set.union(*(set(samples) for samples in angle_samples.values()))
    details = [dict(image=key[0], target_id=key[1], annotation_id=key[2],
                    **{method: angle_samples[method].get(key) for method in METHODS})
               for key in sorted(all_angle_keys)]
    return dict(direction_all_matched=paired_stats(angle_samples),
                direction_reliable=paired_stats(reliable_angles), near_round_excluded=near_round,
                missed_targets=missed, detection={method: read_json(folder / 'evaluation' / method / 'report.json')
                                                  for method in METHODS}), details


def display(value):
    return '—' if value is None else f'{value:.4f}'


def write_report(output, summary, angle_details):
    lines = ['# 三种主轴的量化对比', '',
             f"实验时间：{summary['created_at_utc']}；环境：{summary['environment']}。", '',
             '每图仅一次分割和候选提取；只切换 axis，重新测量投影、头杆剖面和亚像素。',
             '角度为 180° 周期无向误差；真实图没有外形尺寸真值。', '',
             '## 方向误差（度）', '',
             '| 数据 / 样本口径 | 方法 | 自身有效 N | 公共有效 N | 公共均值 | 公共最大 |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for cohort, key, label in [('synthetic', 'direction_all_detected', '合成 / 检出'),
                               ('synthetic', 'direction_reliable', '合成 / 可靠'),
                               ('representative', 'direction_all_matched', '真实 / 匹配非近圆'),
                               ('representative', 'direction_reliable', '真实 / 可靠匹配非近圆')]:
        for method in METHODS:
            metric = summary[cohort][key][method]
            common = metric['common_valid']
            lines.append(f"| {label} | {method} | {metric['own_valid']['count']} | {common['count']} | "
                         f"{display(common['mean'])} | {display(common['max_abs'])} |")
    lines += ['', '### 合成方向按生成条件拆分', '',
              '| 条件 | 公共 N | PCA 均值 | fitLine 均值 | rect 均值 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for group, metrics in summary['synthetic']['direction_by_construction'].items():
        values = ' | '.join(display(metrics[method]['common_valid']['mean']) for method in METHODS)
        lines.append(f"| {group} | {metrics['pca']['common_valid']['count']} | {values} |")
    lines += ['', '## 合成尺寸误差（px，可靠且三方法均可测的交集）', '',
              '| 尺寸 | 结果层 | 公共 N | PCA MAE | fitLine MAE | rect MAE |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    for name in DIMENSIONS:
        for slot in SLOTS:
            metrics = summary['synthetic']['dimensions'][name][slot]['statistics']
            count = metrics['pca']['common_valid']['count']
            values = ' | '.join(display(metrics[method]['common_valid']['mae']) for method in METHODS)
            lines.append(f'| {name} | {slot} | {count} | {values} |')
    lines += ['', '## 覆盖率与失败数量', '',
              '分母为所有检出候选，含未匹配和不可靠候选；细化比例不能当作精度。', '',
              '| 数据 | 方法 | 图像失败 | 候选 | 可靠 / 不可靠 | refined / fallback / disabled | 头长缺失 | 总长细化可用 |',
              '| --- | --- | ---: | ---: | --- | --- | ---: | --- |']
    for cohort in ('synthetic', 'representative', 'stress'):
        for method in METHODS:
            c = summary[cohort]['coverage'][method]
            status = c['subpixel_status']
            lines.append(f"| {cohort} | {method} | {c['failed_images']} | {c['objects']} | "
                         f"{c['reliable']} / {c['unreliable']} | "
                         f"{status.get('refined', 0)} / {status.get('fallback', 0)} / {status.get('disabled', 0)} | "
                         f"{c['missing_preferred']['head_length']} | {c['subpixel_available']['length']} / {c['objects']} |")
    lines += ['', '各尺寸每种方法的有效 N、排除原因、单项细化覆盖及模拟重复标准差见 summary.json；',
              '原评价报告保留在各数据目录 evaluation/。refined 是整体细化状态，单项仍可能缺失。', '',
              '## 真实检出（固定候选时不随主轴变化）', '']
    for method in METHODS:
        r = summary['representative']['detection'][method]
        lines.append(f"- {method}: TP={r['tp']}，FP={r['fp']}，FN={r['fn']}，"
                     f"recall={display(r['recall'])}，precision={display(r['precision'])}；"
                     f"匹配近圆排除 {summary['representative']['near_round_excluded'][method]}。")
    lines += ['', '## 方向误差最大的真实匹配样本', '',
              '按三方法误差的最大值排序；编号是图内候选 ID。', '',
              '| 图片 / ID | PCA | fitLine | rect | 面板 |',
              '| --- | ---: | ---: | ---: | --- |']
    worst = sorted(angle_details['representative'],
                   key=lambda row: max(row[m] for m in METHODS if row[m] is not None), reverse=True)[:10]
    for row in worst:
        values = ' | '.join(display(row[m]) for m in METHODS)
        panel = f"representative/panels/{Path(row['image']).stem}.png"
        lines.append(f"| {row['image']} / {row['target_id']} | {values} | [查看]({panel}) |")
    lines += ['', '## 控制形状案例（不加入正常精度统计）', '',
              'near_round 没有唯一长轴真值；huge_head 的构造杆轴为 17°；touching/occluded 没有合并或片段尺寸真值。', '',
              '| 案例 / ID | 方法 | 角度 | 可靠 | 原因 | 头长 |',
              '| --- | --- | ---: | --- | --- | ---: |']
    for image, methods in summary['stress']['cases'].items():
        for method, objects in methods.items():
            for obj in objects:
                issues = ', '.join(obj['quality_issues']) or '—'
                lines.append(f"| {image} / {obj['id']} | {method} | {display(obj['angle_deg'])} | "
                             f"{obj['reliable']} | {issues} | {display(obj['preferred_px']['head_length'])} |")
    lines += ['', '限制：代表图参加过 Lab 参数选择，不是留出测试；匹配框主轴与杆轴定义可能不同。',
              '30 张重复是模拟噪声，占合成集的大部分；不代表 39 个独立形状或实物精度。',
              '当前 locate 每次计算全部三种方向，实验还包含存盘；本报告不比较单算法速度。', '']
    (output / 'report.md').write_text('\n'.join(lines))
    for cohort, rows in angle_details.items():
        with (output / (cohort + '_angles.csv')).open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else ['image'])
            writer.writeheader()
            writer.writerows(rows)


def run(*args, allow_failure=False):
    print('+', *map(str, args), flush=True)
    result = subprocess.run(list(map(str, args)), check=False)
    if result.returncode and not (allow_failure and result.returncode == 1):
        raise RuntimeError('Command failed: ' + str(args[0]))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, default=ROOT / 'build')
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/mvtec_screws')
    parser.add_argument('--synthetic', type=Path, default=ROOT / 'data/synthetic')
    parser.add_argument('--output', type=Path, default=ROOT / 'result/axis_comparison')
    args = parser.parse_args()
    args.build_dir = args.build_dir.resolve()
    args.dataset = args.dataset.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Output must be empty; choose a new directory to preserve previous reports')
    output.mkdir(parents=True, exist_ok=True)
    exe, compare = args.build_dir / 'screw_metrology', args.build_dir / 'compare_axes'
    synthetic_config, real_config = ROOT / 'config/synthetic.yaml', ROOT / 'config/measurement.yaml'
    axes_file, list_file = ROOT / 'config/synthetic_axes.csv', ROOT / 'config/representative.txt'
    # Existing images are read-only; unavailable synthetic fixtures are generated into experiment output.
    synthetic = args.synthetic.resolve()
    if not (synthetic / 'truth.csv').is_file():
        synthetic = output / 'inputs/synthetic'
        run(args.build_dir / 'generate_samples', synthetic)
    run(compare, synthetic / 'images', synthetic_config, output / 'synthetic', allow_failure=True)
    run(compare, args.dataset / 'images', real_config, output / 'representative', list_file, allow_failure=True)
    run(args.build_dir / 'generate_axis_cases', output / 'inputs/stress')
    run(compare, output / 'inputs/stress', synthetic_config, output / 'stress', allow_failure=True)
    results = {cohort: load_cohort(output / cohort) for cohort in ('synthetic', 'representative', 'stress')}
    for method in METHODS:
        run(exe, 'evaluate', '--results', output / 'synthetic/methods' / method,
            '--truth', synthetic / 'truth.csv', '--output', output / 'synthetic/evaluation' / method)
        run(exe, 'evaluate', '--results', output / 'representative/methods' / method,
            '--annotations', args.dataset / 'mvtec_screws.json',
            '--output', output / 'representative/evaluation' / method)
    summary = dict(schema='screw_axis_comparison_summary_v1',
                   created_at_utc=datetime.now(timezone.utc).isoformat(),
                   environment=f'{platform.system()} {platform.machine()}, OpenCV '
                               + read_json(output / 'synthetic/metadata.json')['opencv_version'])
    summary['synthetic'], synthetic_details = synthetic_summary(results['synthetic'], synthetic / 'truth.csv', axes_file)
    summary['representative'], real_details = real_summary(results['representative'], output / 'representative')
    summary['stress'] = dict(cases={image: {method: results['stress'][method][image]['objects']
                                          for method in METHODS} for image in sorted(results['stress']['pca'])})
    for cohort in results:
        summary[cohort]['coverage'] = {method: coverage(results[cohort][method]) for method in METHODS}
    # Reproduction metadata: exact source/config/truth/input bytes, including uncommitted source edits.
    source_files = sorted(path for directory in ('src', 'include', 'tests', 'tools')
                          for path in (ROOT / directory).rglob('*')
                          if path.is_file() and path.suffix in ('.cpp', '.hpp', '.py'))
    frozen = [ROOT / 'CMakeLists.txt', synthetic_config, real_config, axes_file, list_file,
              synthetic / 'truth.csv', args.dataset / 'mvtec_screws.json',
              exe, compare, args.build_dir / 'generate_axis_cases'] + source_files
    frozen += [Path(record['image']) for cohort in results for record in results[cohort]['pca'].values()]
    write_json(output / 'provenance.json', dict(created_at_utc=summary['created_at_utc'],
                                              files={str(path.resolve()): digest(path) for path in frozen}))
    write_json(output / 'summary.json', summary)
    write_report(output, summary, dict(synthetic=synthetic_details, representative=real_details))
    print('Saved:', output / 'report.md')
    if any(c['failed_images'] for cohort in results for c in summary[cohort]['coverage'].values()):
        raise SystemExit('Comparison contains processing failures; see manifest and coverage')


if __name__ == '__main__':
    main()
