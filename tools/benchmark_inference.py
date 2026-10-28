"""Opt-in single-cloud inference benchmark; no changes to production inference.

Run from any directory. See BENCHMARKING.md for units and measurement limits.
"""
import argparse
import csv
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
from functools import wraps
import inspect
import json
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
from unittest.mock import patch

import psutil

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Recorder:
    """Stream samples/events to disk, keeping only stage aggregates in memory."""

    def __init__(self, directory, interval=0.2, gpu=None):
        self.directory = Path(directory)
        self.interval = interval
        self.gpu = gpu
        self.process = psutil.Process()
        self.stack = []
        self.totals = {}
        self.stop = threading.Event()
        self.lock = threading.RLock()
        self.counters = {}
        self.previous_cpu = 0.0
        self.previous_time = None
        self.baseline_rss = None
        self.peak_rss = 0
        self.peak_pss = None
        self.cpu_seconds = 0.0
        self.read_bytes = 0
        self.write_bytes = 0
        self.sample_error = ''

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=False)
        self.started = time.perf_counter()
        self.sample_stream = (self.directory / 'resources.csv').open('w', newline='')
        self.event_stream = (self.directory / 'events.csv').open('w', newline='')
        self.samples = csv.DictWriter(self.sample_stream, fieldnames=[
            'elapsed_s', 'stage', 'processes', 'rss_bytes', 'pss_bytes',
            'rss_delta_bytes', 'cpu_percent', 'cpu_seconds',
            'read_bytes', 'write_bytes', 'gpu_allocated_bytes', 'gpu_reserved_bytes',
        ])
        self.events = csv.DictWriter(self.event_stream, fieldnames=[
            'stage', 'parent', 'start_s', 'duration_s', 'self_s', 'points', 'status',
        ])
        self.samples.writeheader()
        self.events.writeheader()
        self.sample()
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        return self

    def _monitor(self):
        while not self.stop.wait(self.interval):
            try:
                self.sample()
            except Exception as error:
                self.sample_error = repr(error)
                return

    def sample(self):
        rss, pss, count = 0, 0, 0
        pss_available = True
        for process in [self.process, *self.process.children(recursive=True)]:
            try:
                identity = (process.pid, process.create_time())
                memory = process.memory_info()
                cpu = process.cpu_times()
                try:
                    io = process.io_counters()
                    reads, writes = io.read_bytes, io.write_bytes
                except (AttributeError, psutil.AccessDenied):
                    reads, writes = 0, 0
                try:
                    full = process.memory_full_info()
                    value = getattr(full, 'pss', None)
                except psutil.AccessDenied:
                    value = None
                if value is None:
                    pss_available = False
                else:
                    pss += value
                values = (cpu.user + cpu.system, reads, writes)
                # Root process's pre-benchmark work is excluded. New children
                # contribute from birth; retain their counters after they exit.
                if identity not in self.counters:
                    baseline = values if process.pid == self.process.pid else (0, 0, 0)
                    self.counters[identity] = (baseline, values)
                baseline, _ = self.counters[identity]
                self.counters[identity] = (baseline, values)
                rss += memory.rss
                count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        totals = [sum(current[i] - base[i] for base, current in self.counters.values())
                  for i in range(3)]
        now = time.perf_counter()
        cpu_percent = '' if self.previous_time is None else (
            100 * (totals[0] - self.previous_cpu) / (now - self.previous_time)
        )
        self.previous_time, self.previous_cpu = now, totals[0]
        self.cpu_seconds, self.read_bytes, self.write_bytes = totals
        if self.baseline_rss is None:
            self.baseline_rss = rss
        self.peak_rss = max(self.peak_rss, rss)
        if pss_available:
            self.peak_pss = max(self.peak_pss or 0, pss)
        allocated, reserved = self.gpu() if self.gpu else ('', '')
        with self.lock:
            stage = self.stack[-1]['name'] if self.stack else 'idle'
        self.samples.writerow(dict(zip(self.samples.fieldnames, [
            now - self.started, stage, count, rss, pss if pss_available else '',
            rss - self.baseline_rss, cpu_percent, *totals, allocated, reserved,
        ])))
        self.sample_stream.flush()

    @contextmanager
    def stage(self, name, points=0):
        frame = {'name': name, 'start': time.perf_counter(), 'children': 0.0}
        with self.lock:
            parent = self.stack[-1]['name'] if self.stack else ''
            self.stack.append(frame)
        status = 'ok'
        try:
            yield
        except BaseException:
            status = 'error'
            raise
        finally:
            duration = time.perf_counter() - frame['start']
            own = duration - frame['children']
            with self.lock:
                self.stack.pop()
                if self.stack:
                    self.stack[-1]['children'] += duration
            self.events.writerow(dict(zip(self.events.fieldnames, [
                name, parent, frame['start'] - self.started, duration, own, points, status,
            ])))
            self.event_stream.flush()
            total = self.totals.setdefault(name, [0, 0.0, 0.0, 0])
            for i, value in enumerate((1, duration, own, points)):
                total[i] += value

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        try:
            self.sample()
        finally:
            self.sample_stream.close()
            self.event_stream.close()
            write_csv(self.directory / 'stages.csv', [
                dict(stage=name, calls=v[0], inclusive_s=v[1], self_s=v[2],
                     points=v[3], points_per_s=v[3] / v[1] if v[1] else '')
                for name, v in self.totals.items()
            ])


def timed(recorder, name, function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        # Count point occurrences, including repeated crops/votes, not unique points.
        data = next((a for a in (*args, *kwargs.values())
                     if hasattr(a, 'shape') and len(a.shape) >= 2), None)
        points = 0 if data is None else int(
            data.shape[0] * (data.shape[1] if len(data.shape) == 3 else 1)
        )
        with recorder.stage(name, points):
            return function(*args, **kwargs)
    return wrapped


@contextmanager
def instrument(recorder, segmenter, module):
    """Wrap only the benchmark instance; restore everything even on failure."""
    methods = {
        'segment_pcd': 'inference',
        '_segment_subsampled': 'tile_segmentation',
        '_segment_part': 'part_segmentation',
        '_query_neighbors': 'neighbor_query',
        '_update_possibilities': 'possibility_update',
        '_add_predictions': 'prediction_accumulation',
        '_hard_vote': 'tile_voting',
        '_upsample_labels': 'upsampling',
        '_model_predict': 'model_prediction',
        '_segment_big_voxel': 'big_voxel_segmentation',
        '_segment_small_voxel': 'small_voxel_segmentation',
        '_segment_voxel_base': 'voxel_segmentation',
        '_upsample_labeled_chunk_parallel': 'upsampling',
    }
    with ExitStack() as stack:
        for method, name in methods.items():
            if hasattr(segmenter, method):
                stack.enter_context(patch.object(
                    segmenter, method, timed(recorder, name, getattr(segmenter, method))
                ))
        if hasattr(module, 'voxel_subsample_vectorized'):
            stack.enter_context(patch.object(
                module, 'voxel_subsample_vectorized', timed(
                    recorder, 'voxel_subsampling', module.voxel_subsample_vectorized
                )
            ))
        yield


def git_value(*args):
    result = subprocess.run(['git', '-C', str(ROOT), *args],
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else 'unknown'


def run(args):
    # Import heavy runtime dependencies before the task baseline and timer.
    import laspy
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT))
    from src import array_processing

    source = args.input_path.expanduser().resolve()
    if not source.is_file() or source.suffix.lower() not in {'.las', '.laz'}:
        raise ValueError('input_path must be a single existing LAS/LAZ file')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    device = torch.device(args.device)
    gpu = None
    if device.type == 'cuda':
        torch.cuda.init()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        gpu = lambda: (torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device))
    metadata = {
        'branch': git_value('branch', '--show-current'),
        'commit': git_value('rev-parse', 'HEAD'),
        'dirty': bool(git_value('status', '--porcelain')),
        'input_path': str(source), 'input_bytes': source.stat().st_size,
        'input_mtime_ns': source.stat().st_mtime_ns,
        'model_name': args.model_name, 'device': args.device,
        'python': platform.python_version(), 'platform': platform.platform(),
        'cpu': platform.processor(), 'logical_cpus': psutil.cpu_count(),
        'torch': torch.__version__, 'numpy': np.__version__, 'laspy': laspy.__version__,
        'cuda_runtime': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(device) if gpu else '',
        'sample_interval_s': args.sample_interval, 'seed': args.seed,
        'output_path': str(args.output_path or ''),
    }
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    report = args.report_dir
    recorder = Recorder(report, args.sample_interval, gpu)
    status, error, point_count = 'ok', '', 0
    try:
        with recorder:
            with recorder.stage('full_task'):
                with recorder.stage('model_setup'):
                    settings = dict(model_name=args.model_name, device=device,
                                    config_dir=args.config_dir)
                    parameters = inspect.signature(array_processing.SegmentClass).parameters
                    settings['pbar_bool' if 'pbar_bool' in parameters else 'verbose'] = False
                    if args.model_dir is not None:
                        settings['model_dir'] = args.model_dir
                    segmenter = array_processing.SegmentClass(**settings)
                    if hasattr(segmenter, '_rng'):
                        segmenter._rng = np.random.default_rng(args.seed)
                    metadata['config'] = segmenter.config
                    metadata['inference_settings'] = {
                        key: getattr(segmenter, key) for key in (
                            'voxel_size', 'tile_size', 'overlap', 'num_votes',
                            'query_workers', 'voxel_size_big', 'voxel_size_small', 'scaled',
                        ) if hasattr(segmenter, key)
                    }
                    if gpu:
                        torch.cuda.synchronize(device)
                with recorder.stage('read_cloud'):
                    cloud = laspy.read(source)
                with recorder.stage('extract_arrays'):
                    points = np.column_stack((cloud.x, cloud.y, cloud.z))
                    intensity = np.asarray(cloud.intensity)
                    point_count = len(points)
                    metadata['array_bytes'] = points.nbytes + intensity.nbytes
                with instrument(recorder, segmenter, array_processing):
                    labels = segmenter.segment_pcd(points, intensity)
                if gpu:
                    torch.cuda.synchronize(device)
                if args.output_path is not None:
                    with recorder.stage('write_cloud', point_count):
                        output = args.output_path.expanduser().resolve()
                        if output == source or output.exists():
                            raise FileExistsError(f'Refusing to overwrite {output}')
                        if output.suffix.lower() not in {'.las', '.laz'}:
                            raise ValueError('output_path must be a LAS/LAZ filename')
                        # Save the branch-native labels; class mapping is deliberately
                        # left to the caller, since these branches use different labels.
                        cloud.classification = labels
                        output.parent.mkdir(parents=True, exist_ok=True)
                        cloud.write(output)
    except Exception as exception:
        status, error = 'error', repr(exception)
        raise
    finally:
        if report.exists() and 'full_task' in recorder.totals:
            full = recorder.totals['full_task'][1]
            inference = recorder.totals.get('inference', [0, 0])[1]
            summary = dict(
                branch=metadata['branch'], commit=metadata['commit'], status=status,
                input_path=str(source), input_bytes=metadata['input_bytes'],
                model_name=args.model_name, device=args.device, points=point_count,
                full_task_s=full, inference_s=inference,
                full_points_per_s=point_count / full if full else '',
                inference_points_per_s=point_count / inference if inference else '',
                baseline_rss_bytes=recorder.baseline_rss,
                peak_rss_bytes=recorder.peak_rss,
                peak_rss_delta_bytes=recorder.peak_rss - (recorder.baseline_rss or 0),
                peak_pss_bytes=recorder.peak_pss,
                cpu_seconds=recorder.cpu_seconds,
                read_bytes=recorder.read_bytes, write_bytes=recorder.write_bytes,
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if gpu else '',
                gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if gpu else '',
                sample_error=recorder.sample_error, error=error,
            )
            write_csv(report / 'summary.csv', [summary])
            (report / 'report.txt').write_text(
                '\n'.join(f'{k}: {v}' for k, v in summary.items())
                + '\n\nMetadata:\n' + json.dumps(metadata, indent=2, default=str)
                + '\n\nStages (inclusive timings overlap; self timings exclude wrapped children):\n'
                + (report / 'stages.csv').read_text(), encoding='utf-8',
            )


def compare(args):
    rows = []
    for directory in args.runs:
        with (directory / 'summary.csv').open(newline='') as stream:
            for row in csv.DictReader(stream):
                rows.append(dict(run=str(directory), **row))
    if args.output.exists():
        raise FileExistsError(args.output)
    write_csv(args.output, rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    bench = commands.add_parser('run')
    bench.add_argument('--input-path', required=True, type=Path)
    bench.add_argument('--model-name', required=True)
    bench.add_argument('--config-dir', default='final_files')
    bench.add_argument('--model-dir', type=Path)
    bench.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    bench.add_argument('--sample-interval', type=float, default=0.2)
    bench.add_argument('--seed', type=int, default=42)
    bench.add_argument('--report-dir', type=Path, default=Path('benchmark_results') /
                       datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    bench.add_argument('--output-path', type=Path,
                       help='Optional output LAS/LAZ filename; includes write timing.')
    bench.set_defaults(action=run)
    comparison = commands.add_parser('compare')
    comparison.add_argument('runs', nargs='+', type=Path)
    comparison.add_argument('--output', required=True, type=Path)
    comparison.set_defaults(action=compare)
    args = parser.parse_args(argv)
    if args.command == 'run' and not (0 < args.sample_interval < float('inf')):
        parser.error('sample-interval must be finite and positive')
    args.action(args)


if __name__ == '__main__':
    main()
