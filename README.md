# SCAO — Sparse Curvature-Aware Optimizer

[![CI](https://github.com/whispering3/scao/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/whispering3/scao/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/scao.svg)](https://pypi.org/project/scao/)
[![Python](https://img.shields.io/pypi/pyversions/scao.svg)](https://pypi.org/project/scao/)
[![License](https://img.shields.io/pypi/l/scao.svg)](LICENSE)

Sparse Curvature-Aware Optimizer (SCAO) is a PyTorch optimizer for dense
tensors. It combines Adam-style moments with a blockwise,
Kronecker-factored curvature preconditioner.

SCAO is not a claim of universal improvement over AdamW. Results depend on the
model, data, precision, schedule, and training duration. Use matched,
multi-seed experiments before changing a production training configuration.

## Installation

```bash
pip install scao
```

Hugging Face and QLoRA dependencies are optional:

```bash
pip install "scao[hf]"
```

For development:

```bash
git clone https://github.com/whispering3/scao.git
cd scao
pip install -e ".[dev,hf]"
```

## Basic use

```python
from scao import SCAO

optimizer = SCAO(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.01,
    warmup_steps=100,
    precond_freq=10,
    min_precond_updates=2,
)

for batch in dataloader:
    optimizer.zero_grad()
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
```

The initial warmup uses Adam-style moments. SCAO can use its curvature
preconditioner only after both conditions are met:

```text
step > warmup_steps
preconditioner updates >= min_precond_updates
```

For example, `warmup_steps=100`, `precond_freq=20`, and
`min_precond_updates=10` cannot activate the preconditioner before step 201.
Account for this when choosing a benchmark length.

## Defaults and optional controls

The base `SCAO` defaults keep the following optional controls disabled:

| Setting | Default | Reason |
| --- | ---: | --- |
| `sparsity` | `0.0` | A gradient mask changes the update and needs an ablation. |
| `noise_std_init` | `0.0` | Absolute noise is unsafe when gradient scales vary, especially with LoRA. |
| `lars_coeff` | `0.0` | Trust-ratio scaling changes the effective learning rate. |
| `lookahead_k` | `0` | Slow weights add state and alter optimizer dynamics. |

Enable one control at a time and compare it with a matched AdamW run.

If noise is required, use gradient-relative noise instead of a fixed absolute
standard deviation:

```python
optimizer = SCAO(
    model.parameters(),
    noise_std_init=0.01,  # 1% of each effective gradient's RMS
    noise_relative=True,
)
```

Keep it disabled for the initial LoRA comparison. Relative scaling prevents a
small-adapter gradient from being dominated by a value such as `0.01`; it does
not establish that injected noise improves convergence.

## QLoRA and LoRA

For rank-16 QLoRA of a 3B model, use the memory-bounded preset:

```python
from scao import scao_3b

trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = scao_3b(trainable_params, lr=2e-4, weight_decay=0.01)
```

`scao_3b()` uses int8 EMA storage, 256-wide blocks, and
`factor_mode="wide"`. Large LoRA matrix sides are partitioned into blocks and
only the wide-side curvature factor is retained; the rank-`r` side remains an
identity transform. This avoids an unnecessary second dense factor and its
eigendecomposition for rectangular adapter matrices.

When optimizer state is the limiting resource, use the more aggressive profile
and compare it with the same AdamW run:

```python
from scao import scao_3b_low_memory

optimizer = scao_3b_low_memory(trainable_params, lr=2e-4, weight_decay=0.01)
```

It uses 128-wide curvature blocks and less frequent updates. This is a
memory/performance trade-off, not a convergence claim.

Use `factor_mode="full"` with `SCAO(...)` to retain the two-sided Kronecker
preconditioner. `factor_mode="wide"` is designed for rectangular LoRA matrices
and should be compared with both AdamW and the full SCAO profile.

### Optional CUDA path

The package intentionally does not compile CUDA code during `pip install`,
because the installed PyTorch wheel, CUDA toolkit, and GPU architecture must
match. In the target GPU environment, build the optional extension after
installing SCAO:

```bash
scao-build-cuda
```

SCAO emits a warning and uses the PyTorch fallback if the extension is absent.
The supported optimizer path uses the same PyTorch Kronecker application on
every environment; the optional extension accelerates the int8 EMA update.

For Hugging Face `Trainer`, construct the optimizer explicitly:

```python
from scao.integrations.huggingface import get_scao_optimizer
from transformers import Trainer

optimizer, scheduler = get_scao_optimizer(
    model,
    training_args,
    scao_kwargs={
        "max_precond_dim": 256,
        "use_int8_ema": True,
        "min_precond_updates": 2,
        "factor_mode": "wide",
    },
    preconditioner_policy="adapters_only",
    num_training_steps=training_args.max_steps,
)
trainer = Trainer(model=model, args=training_args, optimizers=(optimizer, scheduler))
```

`preconditioner_policy="adapters_only"` leaves non-adapter trainable
parameters on the Adam-style path. Use `"all"`, `"auto"`, or `"none"` only
when that choice is part of the experiment.

## Axolotl

Use the Axolotl integration only when the installed Axolotl version includes
the SCAO optimizer integration. Keep settings explicit:

```yaml
optimizer: scao
optim_args:
  precond_freq: 10
  min_precond_updates: 2
  max_precond_dim: 256
  use_int8_ema: true
  factor_mode: wide
  noise_std_init: 0.0
  sparsity: 0.0
  lars_coeff: 0.0
  lookahead_k: 0
```

## Benchmarking

The repository includes a QLoRA comparison for Qwen2.5-3B:

```bash
python benchmark/scao_vs_adamw_bench/benchmark_4b.py \
  --optimizer both \
  --steps 300 \
  --lr 2e-4 \
  --seed 42 \
  --model-revision main
```

To assess the memory/throughput trade-off, keep every other setting fixed and
run SCAO alone with smaller blocks:

```bash
python benchmark/scao_vs_adamw_bench/benchmark_4b.py \
  --optimizer scao --steps 300 --lr 2e-4 --seed 42 \
  --scao-max-precond-dim 128 --scao-precond-freq 20
```

Repeat with `--scao-max-precond-dim 64` before selecting a profile.

The default benchmark uses int8 curvature EMAs to bound optimizer state. On a
GPU with sufficient memory, measure the FP32 EMA variant separately before
choosing a throughput-oriented profile:

```bash
python benchmark/scao_vs_adamw_bench/benchmark_4b.py \
  --optimizer scao --steps 300 --lr 2e-4 --seed 42 \
  --scao-factor-mode wide --scao-max-precond-dim 256 \
  --scao-precond-freq 40 --no-scao-use-int8-ema
```

This is an ablation, not a recommended default: it trades additional optimizer
state for less quantization work. Compare loss, wall-clock time, and state
against the int8 run using the same seed and complete both seeds before
adopting it.

The script rebuilds the model for each optimizer and keeps the learning rate,
weight decay, scheduler, seed, data split, batch size, accumulation, and update
count aligned. It reports evaluation loss, peak allocated VRAM, optimizer state
size, and the earliest possible SCAO activation step. It also writes one JSON
record per run with the effective optimizer configuration, dataset fingerprints,
package versions, GPU, CUDA/PyTorch versions, measured metrics, requested and
resolved model revisions, and a SHA-256 fingerprint of the installed SCAO code.

Run at least two seeds before comparing losses. Record package versions, GPU,
CUDA and PyTorch versions, model and data revisions, LoRA configuration,
effective batch size, schedule, clipping, weight decay, total updates, and
measurement method.

Historic benchmark artefacts in this repository are retained for reference.
They are not current performance guarantees and should not be used to rank
optimizers without reproducing the protocol above.

### Reference result: QLoRA 3B on A100

The table below is a reproducible two-seed result with
`Qwen/Qwen2.5-3B@main`, rank-16 QLoRA on `q_proj`, `k_proj`, `v_proj`, and
`o_proj`, `Salesforce/wikitext` (`wikitext-2-raw-v1`), 300 optimizer updates,
learning rate `2e-4`, effective batch size 8, linear decay, and an
NVIDIA A100-SXM4-40GB. SCAO used the supported dense-curvature path with
`factor_mode="wide"`, int8 EMA storage, 256-wide blocks, and a curvature
update interval of 40 steps. The optional CUDA extension was unavailable, so
the PyTorch fallback was used.

| Metric (mean of seeds 42 and 123) | AdamW | SCAO wide-256 / freq-40 |
| --- | ---: | ---: |
| Evaluation loss | 2.2383 | 2.2371 |
| Optimizer state | — | 184.1 MiB |
| Peak allocated VRAM | — | 3.79 GiB |
| SCAO training time | — | 836.1 s |

The observed loss difference (`-0.0012`) is encouraging but is not a general
quality claim: it covers two seeds and 300 updates on one workload. The state
and time values describe SCAO only; compare its JSON record with the paired
AdamW JSON on the same GPU before making a memory or throughput claim. Full
records, including software versions, dataset fingerprints, model revision,
and source fingerprint, are generated by the benchmark command and should
accompany external comparisons.

For a multi-scale H100 evidence set and paired-report procedure, follow
[the Hugging Face readiness protocol](benchmark/scao_vs_adamw_bench/HF_READINESS.md).

## Configuration reference

| Parameter | Default | Description |
| --- | --- | --- |
| `lr` | `1e-3` | Optimizer learning rate. Match AdamW first. |
| `betas` | `(0.9, 0.999)` | First and second moment coefficients. |
| `weight_decay` | `0.01` | Decoupled weight decay. |
| `warmup_steps` | `100` | Adam-style steps before a preconditioned update is eligible. |
| `min_precond_updates` | `2` | Curvature updates required before phase two. |
| `precond_freq` | `20` | Curvature update interval in optimizer steps. |
| `k_min`, `k_max` | `8`, `128` | Bounds for adaptive low-rank factors. |
| `max_precond_dim` | `4096` | Maximum dimension of each curvature block. |
| `use_int8_ema` | `False` | Stores EMA factors as int8 with scale metadata. |
| `factor_mode` | `"full"` | `"wide"` tracks only the wide side of a rectangular matrix. |
| `tau` | `1.0` | Natural-gradient norm cap; set `None` to disable. |
| `sparsity` | `0.0` | Fraction of low-magnitude entries filtered. |
| `noise_std_init` | `0.0` | Absolute gradient-noise standard deviation. |
| `noise_relative` | `False` | Scales `noise_std_init` by the current effective-gradient RMS. |
| `lars_coeff` | `0.0` | Enables trust-ratio scaling when positive. |
| `lookahead_k` | `0` | Enables lookahead synchronization when positive. |

Scale presets are available as `scao_sub1b`, `scao_1b`, `scao_3b`,
`scao_3b_low_memory`, `scao_7b`, `scao_40b`, and `scao_125b`. Treat them as
starting configurations, not tuned recommendations for every workload.

## Diagnostics

```python
from scao.logging import ConsoleLogger

optimizer.add_callback(ConsoleLogger(log_every=100))
```

Metrics include active rank, preconditioned layer count, curvature norms where
available, and warmup status. Call `optimizer.synchronize_precond()` before
checkpointing or evaluating when asynchronous preconditioning is enabled.

## Development

Run the local quality gates before opening a pull request:

```bash
python -m ruff check .
python -m mypy scao/ --ignore-missing-imports --exclude 'scao/(benchmarks|tests|cuda)'
python -m pytest scao/tests/ -v --tb=short --ignore=scao/tests/test_profiling.py
python -m build
python -m twine check dist/*
```

The CI matrix also runs the DDP tests and a CPU benchmark smoke test. GPU and
QLoRA benchmarks require a separately provisioned CUDA environment.

## Project layout

```text
scao/
  optimizer.py       optimizer step and state management
  preconditioner.py  low-rank and block-diagonal curvature factors
  filters.py         optional gradient filters and warmup controls
  presets.py         scale-oriented starting configurations
  integrations/      Hugging Face integration
  tests/             unit, DDP, profiling, and integration tests
benchmark/           runnable training and comparison scripts
scripts/             utilities and notebook-oriented helpers
```

## Citation

If you use SCAO in research, cite the repository release and the associated
Zenodo record: <https://doi.org/10.5281/zenodo.19839495>.

## License

Apache License 2.0. See [LICENSE](LICENSE).
