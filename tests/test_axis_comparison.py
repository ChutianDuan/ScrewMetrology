"""Statistics and comparison failure paths; no imaging package required."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import compare_axes as comparison


class StatisticsTests(unittest.TestCase):
    def test_periodic_angles(self):
        self.assertEqual(comparison.angle_error(179, 1), 2)
        self.assertEqual(comparison.angle_error(-1, 181), 2)
        self.assertEqual(comparison.angle_error(180, 0), 0)

    def test_intersection_does_not_hide_own_coverage(self):
        samples = dict(pca={'a': 1, 'b': 9}, fitline={'a': 2}, rect={'a': 3, 'c': 20})
        report = comparison.paired_stats(samples)
        self.assertEqual(report['pca']['own_valid']['count'], 2)
        self.assertEqual(report['fitline']['own_valid']['count'], 1)
        self.assertEqual(report['pca']['common_valid']['mae'], 1)
        self.assertEqual(report['rect']['common_valid']['count'], 1)
        samples['fitline'] = {}
        self.assertIsNone(comparison.paired_stats(samples)['pca']['common_valid']['mae'])

    def test_missing_and_rejected_dimensions_are_not_zero(self):
        def obj(reliable=True, length=201):
            return dict(id=1, reliable=reliable, preferred_px=dict(length=length))
        records = dict(ok=dict(error=None, objects=[obj()]),
                       rejected=dict(error=None, objects=[obj(False)]),
                       absent=dict(error=None, objects=[]),
                       null=dict(error=None, objects=[obj(length=None)]),
                       failed=dict(error='decode', objects=[]))
        truth = [dict(image=image, target_id='1', measurement='length', truth='200')
                 for image in ('ok', 'rejected', 'absent', 'null', 'failed', 'missing')]
        samples, excluded = comparison.dimension_samples(records, truth, 'preferred_px', 'length')
        self.assertEqual(samples, {('ok', 1): 1})
        self.assertEqual(excluded, dict(unreliable_geometry=1, missing_target=1,
                                       unavailable_dimension=1, processing_error=1, missing_result=1))

    def test_real_near_round_and_unreliable_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = {}
            for method in comparison.METHODS:
                folder = root / 'evaluation' / method
                folder.mkdir(parents=True)
                row = dict(image='scene.png', missed_annotation_ids=[4], matches=[
                    dict(target_id=1, annotation_id=1, iou=0.8, angle_error_deg=None),
                    dict(target_id=2, annotation_id=2, iou=0.9, angle_error_deg=3),
                    dict(target_id=3, annotation_id=3, iou=0.7, angle_error_deg=20)])
                (folder / 'images.jsonl').write_text(json.dumps(row) + '\n')
                (folder / 'report.json').write_text('{}')
                results[method] = {'scene.png': dict(error=None, objects=[
                    dict(id=1, reliable=False), dict(id=2, reliable=True), dict(id=3, reliable=False)])}
            report, details = comparison.real_summary(results, root)
            self.assertEqual(report['direction_all_matched']['pca']['common_valid']['count'], 2)
            self.assertEqual(report['direction_reliable']['pca']['common_valid']['count'], 1)
            self.assertEqual(report['near_round_excluded']['pca'], 1)
            self.assertEqual(report['missed_targets']['pca'], 1)
            self.assertEqual(len(details), 2)
            # A failed method must keep other methods' own samples, while the intersection is empty.
            results['fitline']['scene.png']['error'] = 'measurement failed'
            failed = dict(image='scene.png', matches=[], missed_annotation_ids=[1, 2, 3, 4])
            fit_details = root / 'evaluation/fitline/images.jsonl'
            fit_details.write_text(json.dumps(failed) + '\n')
            report, details = comparison.real_summary(results, root)
            self.assertEqual(report['direction_all_matched']['pca']['own_valid']['count'], 2)
            self.assertEqual(report['direction_all_matched']['pca']['common_valid']['count'], 0)
            self.assertTrue(all(detail['fitline'] is None for detail in details))
            # For successfully processed images, altered matching identities must still be rejected.
            results['fitline']['scene.png']['error'] = None
            row['matches'][1]['annotation_id'] = 99
            fit_details.write_text(json.dumps(row) + '\n')
            with self.assertRaisesRegex(RuntimeError, 'match identities'):
                comparison.real_summary(results, root)


@unittest.skipUnless(os.environ.get('SCREW_COMPARE_EXE'), 'CLI integration requires built executable')
class CliTests(unittest.TestCase):
    def test_decode_failure_is_saved_for_every_method(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'bad.png'
            image.write_text('not an image')
            config = Path(__file__).resolve().parents[1] / 'config/synthetic.yaml'
            output = root / 'output'
            command = [os.environ['SCREW_COMPARE_EXE'], str(image), str(config), str(output)]
            run = subprocess.run(command, capture_output=True)
            self.assertEqual(run.returncode, 1)
            results = comparison.load_cohort(output)
            for method in comparison.METHODS:
                self.assertIn('Cannot decode', results[method]['bad.png']['error'])
                self.assertEqual(results[method]['bad.png']['objects'], [])
            sentinel = output / 'keep.txt'
            sentinel.write_text('preserve')
            rerun = subprocess.run(command, capture_output=True)
            self.assertEqual(rerun.returncode, 1)
            self.assertIn(b'must be empty', rerun.stderr)
            self.assertEqual(sentinel.read_text(), 'preserve')

    def test_duplicate_stems_fail_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / 'inputs'
            inputs.mkdir()
            for name in ('same.png', 'same.jpg'):
                (inputs / name).write_text('unused')
            output = root / 'output'
            config = Path(__file__).resolve().parents[1] / 'config/synthetic.yaml'
            run = subprocess.run([os.environ['SCREW_COMPARE_EXE'], str(inputs), str(config), str(output)],
                                 capture_output=True)
            self.assertEqual(run.returncode, 1)
            self.assertIn(b'Duplicate output stem', run.stderr)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
