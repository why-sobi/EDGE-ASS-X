# Edge AI Assignment 3: Quantization & Pruning (ResNet-18, CIFAR-10)

This notebook optimizes ResNet-18 on CIFAR-10 in two ways:

- **Q1:** INT8 quantization with Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT).
- **Q2:** Unstructured and structured pruning with fine-tuning.

All five models are exported to TorchScript and ONNX, run through an inference pipeline and benchmarked on CPU.

The notebook has **two code cells per question**. Inside each cell, the modules are marked with `[M0]`…`[M15]` comment headers.

## Folder contents

| File | Description |
|---|---|
| `Assignment3.ipynb` | Full pipeline (4 code cells) |
| `Technical_Report.md` | Technical report with results and evidence figures |
| `screenshots/` | Evidence captures rendered from the notebook's saved outputs and result CSVs |
| `requirements.txt` / `environment.yml` | Python dependencies (pip / conda) |
| `pipeline_support.py` | Shared data loading, model, evaluation, and training helpers for the standalone scripts |
| `saved_models/` | Model checkpoints (`.pth`) and ONNX exports (`.onnx`); running the notebook also creates TorchScript (`.ts`) exports |
| `results/` | Benchmark and prediction CSV files |
| `run_baseline.py` | Train or load the FP32 baseline and export it |
| `run_ptq.py` | PTQ calibration, conversion, and exports |
| `run_qat.py` | QAT fine-tuning, conversion, and exports |
| `run_pruning.py` | Unstructured and structured pruning, fine-tuning, and exports |
| `benchmark.py` | CPU TorchScript and ONNX Runtime benchmark for all five models |
| `run_inference.py` | Standalone ONNX Runtime CPU inference script |

The evidence images document the outputs already saved in the notebook and CSV files; they are not screenshots of a newly executed training run. The CIFAR-10 dataset is not included in the submission and must be placed as described below to rerun the pipeline.

## Installation

Requires Python 3.10 or 3.11. Tested with PyTorch 2.5.1.

```bash
python -m venv venv
venv\Scripts\activate                     # Linux/macOS: source venv/bin/activate

# GPU (CUDA 12.1) build of PyTorch
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
#   CPU-only alternative: pip install torch==2.5.1 torchvision==0.20.1

pip install -r requirements.txt
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate edgeai_assignment3
```

## Dataset

Put the extracted CIFAR-10 Python version (`cifar-10-batches-py/`) in **either** location:

- `../data/cifar-10-batches-py/` (one folder above this one), **or**
- `./data/cifar-10-batches-py/`

The notebook reads the raw batch files directly. Download them from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz if needed.

## How to run

1. Open a terminal in this folder and run `jupyter notebook Assignment3.ipynb`. You can also open the notebook in VS Code.
2. *(Optional)* To test that everything works in about 2 minutes, set `QUICK = True` at the top of **Q1 · Cell 1**. Set it back to `False` for the real run.
3. Run the four code cells in order:

| Cell | Modules | Main outputs |
|---|---|---|
| **Q1 · Cell 1** | M0 settings, M1 data, M2 model, M3 helpers, M4 FP32 baseline (25 epochs), M5 PTQ, M6 QAT (3 epochs) | `fp32_weights.pth`, `ptq_int8_weights.pth`, `qat_int8_weights.pth` |
| **Q1 · Cell 2** | M7 TorchScript export + inference, M8 CPU benchmark, M9 ONNX export + ONNX Runtime check | `fp32.ts`, `ptq_int8.ts`, `qat_int8.ts`, `*.onnx`, `results/q1_benchmark.csv`, `results/q1_predictions.csv` |
| **Q2 · Cell 1** | M10 pretrained model, M11 unstructured pruning (3 rounds, about 75% sparsity), M12 structured channel pruning (40% of inner channels) + fine-tuning | – |
| **Q2 · Cell 2** | M13 export, M14 inference for all 5 models + `run_inference.py`, M15 benchmark | `unstructured_weights.pth`, `structured_full_model.pth`, `unstructured.ts`, `structured.ts`, `results/q2_benchmark.csv`, `results/q2_predictions.csv`, `results/onnx_results.csv`, `results/all_models.csv` |

**Runtime:** about 20–30 minutes on a mid-range GPU (RTX 3060 class). CPU only is much slower. If `saved_models/fp32_weights.pth` already exists, the baseline is reloaded instead of retrained.

### Run the standalone scripts

After installing dependencies and placing CIFAR-10 as described above, run these from the project folder:

```bash
python run_baseline.py
python run_ptq.py
python run_qat.py
python run_pruning.py
python benchmark.py
```

PTQ, QAT, pruning, and benchmarking scripts use the existing baseline checkpoint. `run_baseline.py --force` retrains it.

## Standalone inference (CPU, ONNX Runtime)

```bash
python run_inference.py ptq_int8 1000     # accuracy of the PTQ model on the first 1000 test images
python run_inference.py structured 10000
```

Model names: `fp32`, `ptq_int8`, `qat_int8`, `unstructured`, `structured`.

The TorchScript models load without any model code:

```python
model = torch.jit.load("saved_models/qat_int8.ts")
```

## Techniques used

| Part | Method |
|---|---|
| Model | torchvision quantizable ResNet-18 with a 3×3 stem and no max-pool (32×32 input) |
| Training | SGD with Nesterov momentum, OneCycle LR, label smoothing 0.1, random flip and shift augmentation run on the GPU |
| PTQ | `fuse_model()`, `fbgemm` backend, calibration on 1000 random training images, `convert()`. A calibration-size sweep (100 / 1000 / 5000 images) is run first |
| QAT | Fake-quant via `prepare_qat`, 3 epochs at lr 1e-3. BN frozen after epoch 1, observers frozen after epoch 2 |
| Unstructured | Layer-wise L1 magnitude pruning, 3 rounds × 37% of remaining weights. Fine-tuned after each round, then `prune.remove` |
| Structured | Keeps the 60% of each block's inner channels with the largest BatchNorm \|γ\|. `conv1`, `bn1` and the inputs of `conv2` are rebuilt, then fine-tuned for 6 epochs |
| Benchmark | CPU with 4 threads. Measures top-1, latency (`torch.utils.benchmark`), images/s, file size, non-zero parameters and MACs (analytic `conv_cost` / `dense_cost` model) |
| Final ranking | MnasNet-style `reward()`, with model size as the cost and a 12 MB budget → `results/all_models.csv` |

## Troubleshooting

- **`StopIteration` in M1:** The CIFAR-10 folder was not found. Check the dataset location.
- **CUDA out of memory:** Lower `BS` in M0.
- **Retraining the baseline:** Delete `saved_models/fp32_weights.pth` before running.
