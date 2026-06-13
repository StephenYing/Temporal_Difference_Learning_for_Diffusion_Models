# Temporal Difference Learning for Diffusion Models

This repository provides the official implementation for **Temporal Difference
Learning for Diffusion Models**, accepted to **ICML 2026**.

The code is organized into two main components:

- `td4diff_edm/`: EDM-based experiments.  See
  `td4diff_edm/README_TDEDM.md`.
- `td4diff_cm/`: consistency-model experiments for CIFAR-10 and 64x64
  datasets.  See `td4diff_cm/README_TDCT.md`.

## Layout

```text
.
├── td4diff_edm/
│   ├── README_TDEDM.md
│   └── src/
└── td4diff_cm/
    ├── README_TDCT.md
    ├── cifar10/
    └── afhq_ffhq/
```

## Upstream Code

This implementation builds on the following public codebases:

- `td4diff_edm/`: based on NVIDIA's
  [`NVlabs/edm`](https://github.com/NVlabs/edm) implementation of
  **Elucidating the Design Space of Diffusion-Based Generative Models**
  by Karras, Aittala, Aila, and Laine.
- `td4diff_cm/cifar10/`: based on OpenAI's
  [`openai/consistency_models_cifar10`](https://github.com/openai/consistency_models_cifar10)
  implementation for CIFAR-10 consistency-model experiments.
- `td4diff_cm/afhq_ffhq/`: based on OpenAI's
  [`openai/consistency_models`](https://github.com/openai/consistency_models)
  implementation used for 64x64 consistency-model experiments.

## Licenses

This is a multi-license repository.  See `LICENSE_note.md` for the top-level license
notice and the corresponding upstream license terms.

- EDM: `td4diff_edm/src/LICENSE.txt`
- CIFAR-10 Consistency Models code: see the upstream
  [`openai/consistency_models_cifar10`](https://github.com/openai/consistency_models_cifar10)
  license notice.
- 64x64 Consistency Models code: see the upstream
  [`openai/consistency_models`](https://github.com/openai/consistency_models)
  license notice.
