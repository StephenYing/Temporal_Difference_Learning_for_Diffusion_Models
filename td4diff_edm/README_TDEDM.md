# TD4Diffusion EDM Code

This directory contains the EDM-side implementation for **Temporal Difference
Learning for Diffusion Models**.  The code is based on NVIDIA's EDM
implementation and keeps the original EDM training interface where possible.

Upstream code: [`NVlabs/edm`](https://github.com/NVlabs/edm), the official
PyTorch implementation of **Elucidating the Design Space of Diffusion-Based
Generative Models** by Karras, Aittala, Aila, and Laine.

The `src/` directory includes the EDM training entry points together with the
upstream `training`, `torch_utils`, and `dnnlib` modules needed to run them.

## What Is Added

`src/train.py` extends the original EDM training CLI with a TD mode:

- `--loss-type edm|td`
- `--td-loss-class TDLossEDMIndexed`
- `--lambda-edm`, `--td-coupling`, and `--td-weight`
- EDM index-space schedule options: `--steps`, `--rho`, `--k-idx`,
  `--one-idx`, `--sigma-min`, and `--sigma-max`
- target-network options: `--target-update-type`, `--target-update-freq`, and
  `--target-ema-tau`
- optional FID controls: `--fid-every`, `--fid-num-images`,
  `--fid-ref-path`, `--fid-nfe-grid`, and `--early-stop-fid`

## Layout

```text
td4diff_edm/
└── src/
    ├── dnnlib/
    ├── docs/
    ├── torch_utils/
    ├── training/
    ├── train.py
    ├── generate.py
    ├── fid.py
    ├── dataset_tool.py
    ├── download_ffhq.py
    └── env.yml
```

## CIFAR-10 Commands

Prepare the dataset and FID reference statistics using the standard EDM tools:

```bash
cd td4diff_edm/src
DATA_ROOT=$PWD/data/cifar10
python dataset_tool.py --source="$DATA_ROOT" --dest=datasets/cifar10-32x32.zip
python fid.py ref --data=datasets/cifar10-32x32.zip --dest=fid-refs/cifar10-32x32.npz
```

EDM baseline:

```bash
torchrun --standalone --nproc_per_node=4 train.py \
    --outdir=training-runs \
    --data=datasets/cifar10-32x32.zip \
    --cond=1 \
    --arch=ddpmpp \
    --precond=edm \
    --duration=200 \
    --batch=512 \
    --batch-gpu=128 \
    --lr=2e-4 \
    --ema=0.5 \
    --augment=0.12 \
    --loss-type=edm \
    --fid-ref-path=fid-refs/cifar10-32x32.npz
```

TD training:

```bash
torchrun --standalone --nproc_per_node=4 train.py \
    --outdir=training-runs \
    --data=datasets/cifar10-32x32.zip \
    --cond=1 \
    --arch=ddpmpp \
    --precond=edm \
    --duration=200 \
    --batch=512 \
    --batch-gpu=128 \
    --lr=2e-4 \
    --ema=0.5 \
    --augment=0.12 \
    --loss-type=td \
    --td-loss-class=TDLossEDMIndexed \
    --lambda-edm=0.5 \
    --td-coupling=markov \
    --td-weight \
    --steps=18 \
    --rho=7.0 \
    --k-idx=3.0 \
    --one-idx=0.2 \
    --sigma-min=0.002 \
    --sigma-max=80.0 \
    --target-update-type=ema \
    --target-ema-tau=0.999 \
    --fid-ref-path=fid-refs/cifar10-32x32.npz
```

Resume from a saved training state:

```bash
torchrun --standalone --nproc_per_node=4 train.py \
    --resume=training-runs/<run>/training-state-*.pt \
    --outdir=training-runs \
    --data=datasets/cifar10-32x32.zip \
    --cond=1 \
    --arch=ddpmpp \
    --precond=edm \
    --loss-type=td
```

Use the same architecture, loss, batch, and TD hyperparameters when resuming.

## Evaluation

Use the standard EDM generation and FID workflow:

```bash
cd td4diff_edm/src
torchrun --standalone --nproc_per_node=1 generate.py \
    --outdir=fid-tmp \
    --seeds=0-49999 \
    --subdirs \
    --network="$NETWORK_PKL" \
    --batch=64 \
    --steps=18

torchrun --standalone --nproc_per_node=1 fid.py calc \
    --images=fid-tmp \
    --ref=fid-refs/cifar10-32x32.npz \
    --num=50000 \
    --batch=64
```

## License

The EDM-derived files retain the upstream EDM license.  See
`src/LICENSE.txt`.
