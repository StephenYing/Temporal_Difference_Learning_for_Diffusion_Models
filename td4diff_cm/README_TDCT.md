# TD4Diffusion Consistency-Model Code

This directory contains the consistency-model implementation for **Temporal
Difference Learning for Diffusion Models**.  The code follows the original
Consistency Models layout and keeps the CIFAR-10 and AFHQ/FFHQ implementations
separate.

Upstream code:

- `cifar10/`: based on OpenAI's
  [`openai/consistency_models_cifar10`](https://github.com/openai/consistency_models_cifar10)
  implementation for CIFAR-10 consistency-model experiments.
- `afhq_ffhq/`: based on OpenAI's
  [`openai/consistency_models`](https://github.com/openai/consistency_models)
  implementation used for 64x64 consistency-model experiments.

## Layout

```text
td4diff_cm/
├── cifar10/       # JAX/Haiku CIFAR-10 experiments
└── afhq_ffhq/     # PyTorch AFHQ/FFHQ experiments
```

## CIFAR-10

The CIFAR-10 implementation adds the TD consistency-training objective:

- `configs/cifar10_ve_consistency_td.py`: TD training config
- `jcm/losses.py`: `get_consistency_td_loss_fn(...)`
- TD hyperparameters: `lambda_td`, `k_idx`, `one_idx`, and `weight_td`

Before running, set `training.ref_model_path` in the config to the reference EDM
checkpoint.

```bash
cd td4diff_cm/cifar10
python -m pip install -e .
python -m jcm.main \
    --config=configs/cifar10_ve_consistency_td.py \
    --mode=train
```

The adaptive CT baseline is available through:

```bash
python -m jcm.main \
    --config=configs/cifar10_ve_ct_adaptive.py \
    --mode=train
```

## AFHQ / FFHQ

The AFHQ/FFHQ implementation adds `consistency_td` to the PyTorch
consistency-model training code:

- `scripts/cm_train.py`: supports `--training_mode consistency_td`
- `cm/karras_diffusion.py`: `consistency_td_losses(...)`
- `cm/train_util.py`: target-model update and checkpoint logic for TD training
- `cm/script_util.py`: TD defaults for `lambda_td`, `one_idx`, and `weight_td`

Example:

```bash
cd td4diff_cm/afhq_ffhq
python -m pip install -e .
export OPENAI_LOGDIR=$PWD/runs
DATA_DIR=$PWD/data
python scripts/cm_train.py \
    --training_mode consistency_td \
    --data_dir "$DATA_DIR" \
    --lambda_td 0.5 \
    --one_idx 0.25 \
    --weight_td True
```

## Licenses

The consistency-model-derived files retain the corresponding upstream license
terms from
[`openai/consistency_models_cifar10`](https://github.com/openai/consistency_models_cifar10)
and [`openai/consistency_models`](https://github.com/openai/consistency_models).
See the repository-level `LICENSE.md` for the top-level license notice.
