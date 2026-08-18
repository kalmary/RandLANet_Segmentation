![BANNER](https://github.com/kalmary/RandLANet_Segmentation/blob/readme-preparation/img/Banner.png)

# Table of contents
1. [Overview](#overview)
2. [Repository Structure](#fstructure)
3. [Installation](#installation)
4. [Usage](#usage)
    1. [Preprocessing](#preprocessing)
    2. [Training](#training)
    3. [Evaluation](#evaluation)
5. [Final data processing](#pipelines)
6. [Testing](#testing)
7. [Citation](#citation)

---
# 1. Overview <a name="overview"></a>

**RandLANet_Segmentation** is a set of tools for point cloud semantic segmentation using the RandLANet architecture. RandLANet is a deep learning model designed to process large-scale point clouds. The repository includes tools for defining the model, training it, and performing segmentation on new files. Key Features:
- Data preprocessing: cut, decimate and distribute data for training model,
- Model definition: necessary code to define and build the RandLANet model architecture, allows for scalability and adjustment for hardware-specific needs,
- Training & Evaluation: Tools for training the model on custom datasets and evaluating its performance,
- Inference & Segmentation: Utility to perform semantic segmentation on new point cloud files using pre-trained models/ segmentation on preloaded arrays.


You can see the RandLANet scheme from original paper [[1]](#cite1) on the picture below:
![IMG](https://github.com/kalmary/RandLANet_Segmentation/blob/readme-preparation/img/RandLANet_scheme.png)


Key modifications we added:
- memory efficient, gpu - based knn search,
- model configurability from .json file,
- high scalability of model, training and processing pipelines,
- better decoder upsampling.

---
# 2. Repository structure: <a name="fstructure"></a>

```
.
├── src
│   ├── data_processing
│   │   └── downsample_LAZ.py       # Creates paired NPY point-cloud tiles and PKL KD-trees
│   ├── final_files                 # Models and configs used for inference
│   ├── main.py                     # LAS/LAZ inference entry point
│   ├── model_pipeline
│   │   ├── TrainSegmAutomated.py   # CUDA training and Optuna optimization
│   │   ├── EvalSegm_RandLANet.py   # CUDA evaluation
│   │   ├── RandLANet_CB.py         # Configurable RandLANet model
│   │   ├── model_configs           # Model architecture configs
│   │   ├── training_configs        # Training and optimization configs
│   │   └── training_results        # Trained models, configs, reports, and plots
│   └── utils
│       ├── nn_utils                 # Git submodule with losses and metrics
│       └── pcd_manipulation.py
├── tests                            # Pytest suite
├── processed_files.txt              # Successful inference mappings
└── error_files.txt                  # Failed inference source paths
```
---

# 3. Instalation: <a name="installation"></a>

The `nn_utils` submodule uses SSH, so configure a GitHub SSH key before cloning.

```bash
git clone --recurse-submodules git@github.com:kalmary/RandLANet_Segmentation.git
cd RandLANet_Segmentation

python -m venv .venv
source .venv/bin/activate

# Install project requirements without PyTorch/CUDA.
pip install -r requirements.txt

# CUDA build used by this project.
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

For an existing checkout, synchronize and update the submodule after pulling:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

---

# 4. Usage <a name="usage"></a>
## 1. Preprocessing <a name="preprocessing"></a>

Training and evaluation consume paired files in each dataset split:

- `*.npy`: arrays with shape `(N, 5)` containing `x, y, z, intensity, label`,
- `*.pkl`: a `scipy.spatial.cKDTree` with the same stem and point count.

Configure the source, output, voxel-size, and tile-size constants in the
`if __name__ == "__main__"` block of `src/data_processing/downsample_LAZ.py`,
then run it from the repository root:

```bash
python src/data_processing/downsample_LAZ.py
```

The resulting `train/`, `val/`, and `test/` directories are referenced by the
JSON files in `src/model_pipeline/training_configs/`.

## 2. Training <a name="training"></a>
Examine contents of:
- ``src/model_pipeline/model_configs`` - .json files with model architectures,
- ``src/model_pipeline/training_configs`` - .json files with training configs.
Pay attention to above files and adjust them to ensure the fit with your available resources.

Files with `_single` suffix are meant for single training without any optimizations. Others are for multi-hyperparameter optimization.

Training is CUDA-only. Run the command from the repository root:

```bash
python src/model_pipeline/TrainSegmAutomated.py --model_name MODEL_NAME --mode 1
```

Available flags:

- `--model_name` — required model name. Results are stored under
  `src/model_pipeline/training_results/`.
- `--mode`:
  - `0` — short test training with reduced parameters,
  - `1` — single training using `config_train_single.json`,
  - `2` — Optuna optimization using `config_train.json`,
  - `3` — compile model configs and estimate their resource requirements.

Training uses `FocalLoss` with class weights
computed by `compute_pos_weights_prob()` from the labels in column `4` of every
NPY tile. Sampling itself is class-independent: crop centers follow a spatial
possibility score and each tile point must occur in at least one crop. Missing
directories, directories without NPY tiles, malformed tiles, empty tiles, and
labels outside the configured classes stop training with a descriptive error.

Before training, both iterable loaders are traversed once to measure progress
bar and `OneCycleLR` step counts. Every `train_repeat` is an independent run with
a fresh model, optimizer, scheduler, loss instances, and metric histories. The
outer checkpoint logic selects the best epoch across all repetitions using
ordinary validation accuracy and validation loss.

Mode `2` currently runs 80 Optuna trials. Change `n_trials` in `main()` if a
different optimization budget is required. The best model, its config, and
metric-history plots are saved during training.

Use `--help` to display the current CLI options.

## 3. Evaluation <a name="evaluation"></a>

Evaluation is CUDA-only and uses the test dataset configured in the saved model
config. Run it from the repository root:

```bash
python src/model_pipeline/EvalSegm_RandLANet.py --model_name MODEL_NAME --mode 1
```

`MODEL_NAME` is the trained filename without `.pt`, for example
`RandLANetTest_123`. Available modes are:

- `0` — verify that the trained model compiles and can run,
- `1` — evaluate the model and generate outputs.

Evaluation uses the same spatial possibility sampler and `FocalLoss` class
weights as training, while reporting ordinary accuracy. Dataset validation
rejects missing directories, missing NPY tiles, malformed or empty tiles,
missing matching PKL trees, and invalid labels before metrics are generated.

Evaluation outputs include precision-recall and ROC curves, a confusion matrix,
and a text classification report. Copy selected trained models and configs to
`src/final_files/` for inference.

# 5. Final data processing <a name="pipelines"></a>

To perform .LAZ files semantic segmentation, based on pretrained model run:
```bash
python src/main.py --model_name MODEL_NAME --device cuda --input_path path/to/raw/data --output_path path/with/processed/files --verbose
```
Available flags:

- `--model_name` — model filename without `.pt`,
- `--device` — `cpu` or `cuda`; CUDA availability is checked when requested,
- `--input_path` — one LAS/LAZ file or a directory searched recursively,
- `--output_path` — optional output directory; otherwise outputs are written beside their sources,
- `--verbose` — display file and inference progress.

Each successful file is appended to `processed_files.txt` as
`original/path -> processed/path`, while failed source paths are appended to
`error_files.txt`. Paths are stored with POSIX separators. Files already listed
in `processed_files.txt` are skipped on later runs, regardless of whether they
also appear in `error_files.txt`. A per-file processing error is logged and does
not stop the remaining files in the batch.

For preloaded point clouds, use the segmentation class directly:

```python
import numpy as np
import torch

from src.array_processing import SegmentClass

segmenter = SegmentClass(
    model_name="MODEL_NAME",
    config_dir="src/final_files",
    device=torch.device("cuda"),
    voxel_size=0.10,
    tile_size=40.0,
    overlap=5.0,
    pbar_bool=True,
)

points = np.random.random((1_000_000, 3)).astype(np.float32)
intensity = np.random.randint(0, 65536, len(points), dtype=np.uint16)
labels = segmenter.segment_pcd(points, intensity)
```

# 6. Testing <a name="testing"></a>

Tests use pytest:

```bash
python -m pytest -q
```

The DataLoader uses one non-persistent spawned worker per traversal, so tests
that iterate it require an environment that permits process synchronization
primitives.

# 7. Citation <a name="citation"></a>

Our model is based on:
- https://github.com/QingyongHu/RandLA-Net
- https://github.com/aRI0U/RandLA-Net-pytorch

To cite the original paper about RandLANet use:

[1] - RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds  <a name="cite1"></a>
```bibtex
@article{RandLA-Net,
  arxivId = {1911.11236},
  author = {Hu, Qingyong and Yang, Bo and Xie, Linhai and Rosa, Stefano and Guo, Yulan and Wang, Zhihua and Trigoni, Niki and Markham, Andrew},
  eprint = {1911.11236},
  title = {{RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds}},
  url = {http://arxiv.org/abs/1911.11236},
  year = {2019}
}
```

### **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.





















