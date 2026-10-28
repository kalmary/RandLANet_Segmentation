"""Benchmark-only tests: no raw clouds or checkpoints are read."""
import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    'benchmark_inference',
    Path(__file__).resolve().parents[1] / 'tools' / 'benchmark_inference.py',
)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


@pytest.mark.parametrize('legacy', [False, True])
def test_wrappers_preserve_results_restore_methods_and_capture_nested_stages(tmp_path, legacy):
    class Segmenter:
        def _model_predict(self, points):
            return points[:, 0].astype(np.int8)

        def segment_pcd(self, points, intensity):
            return (self._segment_small_voxel(points, intensity) if legacy
                    else self._segment_part(points, intensity))

    method = '_segment_small_voxel' if legacy else '_segment_part'
    setattr(Segmenter, method, lambda self, points, intensity: self._model_predict(points))
    segmenter = Segmenter()
    original = segmenter.segment_pcd
    points = np.arange(12).reshape(4, 3)
    intensity = np.ones(4)
    expected = original(points, intensity)
    directory = tmp_path / 'run'
    with benchmark.Recorder(directory, interval=0.01) as recorder:
        with recorder.stage('full_task'):
            with benchmark.instrument(recorder, segmenter, SimpleNamespace()):
                np.testing.assert_array_equal(segmenter.segment_pcd(points, intensity), expected)
                time.sleep(0.03)
    assert segmenter.segment_pcd == original
    assert not segmenter.__dict__  # No wrappers left behind.
    stages = {row['stage']: row for row in read_csv(directory / 'stages.csv')}
    assert int(stages['model_prediction']['points']) == 4
    assert int(stages['inference']['calls']) == 1
    assert float(stages['inference']['inclusive_s']) >= float(stages['model_prediction']['inclusive_s'])
    assert float(stages['inference']['self_s']) >= 0
    samples = read_csv(directory / 'resources.csv')
    assert len(samples) >= 2
    assert int(samples[0]['rss_delta_bytes']) == 0
    assert all(row['gpu_allocated_bytes'] == '' for row in samples)
    assert recorder.peak_rss >= recorder.baseline_rss > 0


def test_failure_records_partial_events_and_restores_wrappers(tmp_path):
    def fail(*args):
        raise ValueError('synthetic failure')

    segmenter = SimpleNamespace(segment_pcd=fail)
    directory = tmp_path / 'failed'
    with pytest.raises(ValueError, match='synthetic failure'):
        with benchmark.Recorder(directory) as recorder:
            with benchmark.instrument(recorder, segmenter, SimpleNamespace()):
                segmenter.segment_pcd(np.zeros((3, 3)), np.ones(3))
    assert segmenter.segment_pcd is fail
    assert read_csv(directory / 'events.csv')[0]['status'] == 'error'
    assert (directory / 'stages.csv').exists()


def test_report_directory_cannot_be_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        with benchmark.Recorder(tmp_path):
            pass


def test_compare_preserves_branch_status_and_metrics(tmp_path):
    runs = [tmp_path / 'prob', tmp_path / 'dales']
    for directory, branch, status in zip(runs, ['prob-sampler-DALES', 'DALES'], ['ok', 'error']):
        directory.mkdir()
        benchmark.write_csv(directory / 'summary.csv', [
            dict(branch=branch, status=status, full_task_s=2.5),
        ])
    output = tmp_path / 'comparison.csv'
    benchmark.compare(SimpleNamespace(runs=runs, output=output))
    rows = read_csv(output)
    assert [r['branch'] for r in rows] == ['prob-sampler-DALES', 'DALES']
    assert rows[1]['status'] == 'error'
    assert float(rows[0]['full_task_s']) == 2.5
    with pytest.raises(FileExistsError):
        benchmark.compare(SimpleNamespace(runs=runs, output=output))


@pytest.mark.parametrize('legacy,fail', [(False, False), (True, False), (True, True)])
def test_full_runner_with_fake_reader_and_model(tmp_path, monkeypatch, legacy, fail):
    """Exercise summaries and both constructor APIs without reading a raw file."""
    class Common:
        config = {'num_classes': 2, 'model_config': {}}

        def _model_predict(self, points):
            if fail:
                raise RuntimeError('fake model failed')
            return np.zeros(len(points), dtype=np.uint8)

        def segment_pcd(self, points, intensity):
            return self._model_predict(points)

    class Probability(Common):
        def __init__(self, model_name, device, config_dir, pbar_bool):
            assert pbar_bool is False
            self._rng = np.random.default_rng()

    class Legacy(Common):
        def __init__(self, model_name, device, config_dir, verbose):
            assert verbose is False

    cloud = SimpleNamespace(x=np.arange(4), y=np.zeros(4), z=np.zeros(4),
                            intensity=np.ones(4))
    fake_laspy = SimpleNamespace(read=lambda _: cloud, __version__='fake')
    fake_torch = SimpleNamespace(
        device=lambda name: SimpleNamespace(type=name),
        manual_seed=lambda _: None, __version__='fake',
        version=SimpleNamespace(cuda=None),
    )
    fake_array = SimpleNamespace(SegmentClass=Legacy if legacy else Probability)
    monkeypatch.setitem(sys.modules, 'laspy', fake_laspy)
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    monkeypatch.setitem(sys.modules, 'src', SimpleNamespace(array_processing=fake_array))
    source = tmp_path / 'placeholder.las'
    source.write_bytes(b'placeholder, never passed to a LAS decoder')
    directory = tmp_path / 'report'
    args = SimpleNamespace(input_path=source, output_path=None, device='cpu',
                           model_name='fake', config_dir='fake', model_dir=None,
                           seed=42, sample_interval=0.01, report_dir=directory)
    if fail:
        with pytest.raises(RuntimeError, match='fake model failed'):
            benchmark.run(args)
    else:
        benchmark.run(args)
    summary = read_csv(directory / 'summary.csv')[0]
    assert summary['status'] == ('error' if fail else 'ok')
    assert int(summary['points']) == 4
    assert float(summary['full_task_s']) >= float(summary['inference_s']) > 0
    assert (directory / 'report.txt').exists()
    assert summary['gpu_peak_allocated_bytes'] == ''
    if fail:
        assert 'fake model failed' in summary['error']


def test_final_sampling_failure_preserves_timing_reports(tmp_path, monkeypatch):
    directory = tmp_path / 'monitor_failure'
    with benchmark.Recorder(directory) as recorder:
        with recorder.stage('full_task'):
            pass
        def fail():
            raise RuntimeError('monitor failed')
        monkeypatch.setattr(recorder, 'sample', fail)
    assert 'monitor failed' in recorder.sample_error
    assert read_csv(directory / 'stages.csv')[0]['stage'] == 'full_task'
