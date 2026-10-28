# Single-file inference benchmarks

This optional runner is identical on `prob-sampler-DALES` and `DALES`.
It wraps the existing `SegmentClass` instance at runtime; production source,
model math, sampling, normalization, and upsampling are unchanged. Normal
`src/main.py` runs incur no benchmark overhead.

Install the optional monitor dependency in your existing inference environment:

```bash
.venv/bin/python -m pip install -r requirements-benchmark.txt
```

## Plan and measurements

| Measurement | Purpose | Artifact |
| --- | --- | --- |
| Full task and inference seconds, raw points/second | Compare end-to-end and inference throughput | `summary.csv` |
| Model/config loading, LAS/LAZ reading, array extraction, optional output writing | Separate setup, I/O, and processing costs | `stages.csv`, `events.csv` |
| Subsampling, tile/part processing, neighbor queries, possibility updates, model calls, accumulation, voting, upsampling | Locate costs in probability sampling | `stages.csv`, `events.csv` |
| Big/small voxel processing, voxel inference, model calls, multiprocessing upsampling | Locate costs in the legacy DALES path | `stages.csv`, `events.csv` |
| RAM baseline, sampled peak, growth above baseline, time series | Compare memory consumption and fluctuations | `resources.csv`, `summary.csv` |
| Process-tree RSS and PSS, worker count | Account for DALES multiprocessing and shared memory | `resources.csv` |
| CPU seconds, CPU utilization, physical read/write bytes | Explain compute and storage pressure | `resources.csv`, `summary.csv` |
| CUDA allocated/reserved memory over time and allocator peaks | Track tensor memory including transient CUDA peaks | `resources.csv`, `summary.csv` |
| Branch, revision, dirty state, input identity, model/config, seed, runtime/hardware | Make comparisons traceable | `report.txt` |

## Run later, once per branch and file

Use the **same absolute raw input file**, hardware, device, sampling interval,
and seed on both branches. Use each branch's compatible checkpoint and config.
Relative model/config directories resolve against `src`, matching `SegmentClass`.
Absolute directories are recommended. Model names exclude `.pt`.

```bash
# On prob-sampler-DALES:
.venv/bin/python tools/benchmark_inference.py run \
  --input-path /data/shared_scan.laz --model-name PROB_MODEL \
  --config-dir /models/prob/dict_files --model-dir /models/prob \
  --device cuda --report-dir benchmark_results/prob_dales_01

# On DALES (same script and arguments, different checkpoint/report directory):
.venv/bin/python tools/benchmark_inference.py run \
  --input-path /data/shared_scan.laz --model-name DALES_MODEL \
  --config-dir /models/dales/dict_files --model-dir /models/dales \
  --device cuda --report-dir benchmark_results/dales_01
```

Each run requires a new report directory and never modifies the raw input.
By default, the full task includes model setup, file reading, array extraction,
and inference. Interpreter startup/imports, CUDA context creation, and report
finalization are excluded. No tracking files from `src/main.py` are read or
written, so a previously processed input is still benchmarked.

To include file writing, add `--output-path /results/new_prediction.laz`.
The runner refuses existing outputs. Predictions are saved as branch-native
zero-based labels; this is a benchmark output, with no extra class remapping.
Compare runs with the same output policy and compression format.

Failures propagate with a nonzero exit status and write partial reports when
the timed task has started. Check `status`, `error`, and `sample_error` before
including a run in comparisons. A monitor failure makes resource data incomplete.

## CSV semantics and plotting

- `resources.csv`: elapsed seconds on the X axis; RSS/PSS/RSS growth or CUDA
  allocated/reserved bytes on the Y axis (divide by 2**20 for MiB). `stage`
  identifies the innermost timed stage at that sample. The default interval is
  0.2 seconds; override with `--sample-interval 0.1`.
- `events.csv`: one row per call, including start, duration, parent, self time,
  and status. Plot per-part or per-model-call durations across the run.
- `stages.csv`: aggregated call counts, inclusive seconds, self seconds, input
  point occurrences, and occurrences/second. Inclusive parent and child timings
  overlap: **do not sum them**. Self time excludes wrapped children and includes
  other work, such as normalization, tree construction, fragmentation, and batch
  preparation. Generator iteration remains in its calling stage.
- Stage point counts describe the first array argument with at least two
  dimensions. Model batches count batch size times crop size. Repeated
  neighborhoods/votes count repeatedly; only summary throughput uses the number
  of unique raw input points. Do not compare model occurrences/second directly
  with end-to-end raw points/second.
- CUDA predictions already copy results to CPU in both implementations, so
  prediction timings include transfer and completion. Setup and task completion
  synchronize CUDA. This is wall time, not kernel-only time. GPU memory comes
  from this process's PyTorch allocator; it excludes driver/context memory,
  non-PyTorch allocations, and other applications. CPU runs leave GPU cells empty.
- CPU utilization is summed over the task's process tree: 100% is one busy core,
  so values can exceed 100%. It includes monitor/report overhead. Counters for
  observed workers are retained after exit; workers that start and finish between
  samples can be missed. I/O counters measure physical storage traffic where
  supported, rather than LAS decompressed bytes or page-cache reads. Report I/O
  itself contributes a small amount. Unsupported I/O counters are zero.
- RSS includes shared pages in each process; summed RSS can overcount. PSS
  apportions shared pages and is preferable for multiprocessing comparisons.
  PSS is blank when unavailable. Baseline RSS includes imported libraries and
  the initialized CUDA context, before checkpoint loading. Growth above baseline
  is an observed process-tree change, not an exact count of task allocations.
  Sampling can miss brief RAM peaks. Existing unrelated processes are excluded.

Create a comparison table (no inference runs):

```bash
.venv/bin/python tools/benchmark_inference.py compare \
  benchmark_results/prob_dales_01 benchmark_results/dales_01 \
  --output benchmark_results/comparison.csv
```

This concatenates summaries with a run identifier and preserves failed runs for
inspection. Use pandas to filter `status == 'ok'`, then pivot branch versus full
seconds, inference seconds, peak PSS/RSS growth, and peak GPU allocation. Stage
CSVs share their schema; implementation-specific stages should be compared
within their own pipeline. `inference`, `model_prediction`, and `upsampling`
are common stage names with different internal workloads.

Repeat runs in fresh processes and report median/spread. Distinguish first/cold
and warm filesystem-cache runs; this runner does not flush OS caches. A fixed
seed does not guarantee deterministic CUDA results or equivalent model work.
The checkpoints, number of classes, voting, and processing settings affect
speed, so retain `report.txt` with every table. These are resource/performance
benchmarks, not accuracy measurements. Monitoring and wrappers add overhead;
use the same interval on both branches.

Synthetic verification only (does not open any LAS/LAZ files):

```bash
.venv/bin/python -m pytest -q tests/test_benchmark_inference.py
```
