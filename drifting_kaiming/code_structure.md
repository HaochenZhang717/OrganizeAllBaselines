# Code Structure: Generative Modeling via Drifting

Official JAX codebase for the paper *Generative Modeling via Drifting* (Deng et al., arXiv 2602.04770).
Target task: class-conditional ImageNet 256×256 generation in **one forward pass** (1 NFE).

---

## High-Level Overview

The framework has two stages:

1. **MAE Pretraining** (`train_mae.py`): Train a ResNet-based Masked Autoencoder as a fixed feature extractor.
2. **Generator Training** (`train.py`): Train a one-step DiT generator using the **drift loss**, which pushes generated features toward a memory bank of real images.

At inference, the generator maps (Gaussian noise + class label + CFG scale) directly to an image in a single forward pass.

---

## Entry Points

### `main.py`
Top-level CLI dispatcher. Routes to one of two training paths:
```
python main.py --config <yaml> --workdir <dir>          # MAE training
python main.py --gen --config <yaml> --workdir <dir>    # Generator training
```
Imports `train.main` or `train_mae.main` lazily to avoid initializing both distributed backends.

### `inference.py`
Standalone FID/IS/precision-recall evaluation. Loads a pretrained generator (from `hf://<name>` or a local workdir), generates 50k samples, and reports metrics.
```
python inference.py --init-from "hf://latent_L_sota" --cfg-scale 1.0 --num-samples 50000
```

---

## Core Algorithm Files

### `drift_loss.py` — The Drift Loss
The central loss function of the paper.

**Signature**: `drift_loss(gen, fixed_pos, fixed_neg, weight_gen, weight_pos, weight_neg, R_list)`

**Inputs** (all in feature space, shape `[B, N, D]`):
- `gen`: generated sample features (gradients flow through these)
- `fixed_pos`: real/positive sample features from the memory bank (stop-gradient)
- `fixed_neg`: unconditional/negative sample features (stop-gradient, optional)

**Algorithm**:
1. Computes pairwise distances between generated and all target features.
2. For each temperature `R` in `R_list`, computes a softmax-based affinity that acts as a force field: positive samples attract, negative samples repel.
3. Aggregates forces across all `R` values (each normalized by its own scale).
4. The loss is the MSE between the current generated features and the "goal" position implied by the total force.

This loss encourages generated samples to distribute themselves to match the real data distribution, without any explicit density estimation.

### `memory_bank.py` — ArrayMemoryBank
A CPU-side per-class ring buffer that stores real training images for use as positive/negative anchors during generator training.

**Two banks are maintained**:
- **Positive bank**: stores images indexed by class label (1000 classes).
- **Negative bank**: stores images under a dummy class 0 (unconditional/class-agnostic negatives).

Key methods:
- `add(samples, labels)`: insert new images into the ring buffer.
- `sample(labels, n_samples)`: retrieve `n_samples` stored images per requested label.

---

## Training Loops

### `train.py` — Generator Training
Main function: `train_gen(model, optimizer, logger, ..., activation_fn, feature_params, ...)`

**Per-step logic**:
1. **Memory bank fill**: push a batch of real images (`push_per_step` images) into both the positive (class-conditioned) and negative (unconditional) memory banks.
2. **Sample from banks**: retrieve positive samples (same class) and negative samples (any class) for the current batch.
3. **Generate**: call the generator `gen_per_label` times per label to produce candidate images.
4. **Feature extraction**: run the frozen MAE on both the memory bank images and the generated images to get multi-scale feature representations.
5. **Drift loss**: compute `drift_loss` in feature space, treating memory bank positives/negatives as anchors.
6. **Update**: backprop through the generator only; MAE and memory bank are frozen.
7. **EMA update**: maintain an EMA copy of generator weights.
8. **Periodic evaluation**: compute FID on the EMA model at `eval_per_step` intervals.

Key hyperparameters (from config `train` section):
| Parameter | Description |
|---|---|
| `gen_per_label` | Generator forward passes per label per step |
| `pos_per_sample` / `neg_per_sample` | Samples drawn from each memory bank |
| `cfg_min` / `cfg_max` | CFG scale range sampled each step |
| `push_per_step` | Images pushed to memory bank per step |
| `R_list` | Temperature list for drift loss kernel |

### `train_mae.py` — MAE Training
Main function: `train_mae(model, optimizer, ...)`

Trains `MAEResNetJAX` with masked reconstruction loss. Optionally fine-tunes a classification head in the last `finetune_last_steps` steps.

**Per-step logic**: random mask → ResNet encoder → U-Net decoder → pixel-space MSE on masked patches + optional cross-entropy classification loss.

---

## Models

### `models/generator.py` — DitGen (One-Step Generator)

**Class hierarchy**:
```
DitGen
└── LightningDiT          # DiT-style transformer backbone
    ├── LightningDiTBlock  # Transformer block with AdaLN modulation
    │   ├── Attention      # Multi-head attention (optional RoPE, QK-norm)
    │   ├── SwiGLUFFN / StandardMLP
    │   └── AdaLN          # Adaptive LayerNorm conditioning
    └── FinalLayer         # Output projection with AdaLN
```

**`DitGen.__call__(c, cfg_scale, temp)`**:
1. Samples Gaussian noise `x ~ N(0, I)` shaped `[B, H, W, C]`.
2. Optionally samples discrete noise labels for extra diversity.
3. Combines class embedding + CFG-scale embedding into condition vector.
4. Passes `(x, cond)` through `LightningDiT` in a single forward pass.
5. Returns the output image directly (no iterative denoising).

**Key supporting components**:
- `TimestepEmbedder`: embeds the CFG scale as a sinusoidal timestep-style embedding.
- `RMSNorm`, `modulate()`: AdaLN modulation utilities.
- `apply_rope()`: Rotary positional embeddings.
- `sincos_init()`: 2D sinusoidal positional embedding initializer.

**`build_generator_from_config(model_config)`**: reconstructs `DitGen` from a metadata dict (used when loading HF checkpoints).

### `models/mae_model.py` — MAEResNetJAX (Feature Extractor)

A ResNet encoder + U-Net decoder MAE used as a **frozen** feature extractor during generator training.

**Architecture**:
```
MAEResNetJAX
├── _ResNetEncoder     # 4-stage ResNet (conv1, layer1–4), outputs multi-scale feature maps
└── _UNetDecoder       # U-Net decoder with skip connections for pixel reconstruction
    └── _UpBlock (×4)  # Bilinear upsample + concat skip + conv-GN-ReLU
```

**`get_activations(x, patch_mean_size, patch_std_size, every_k_block)`**: the key method called during generator training. Returns a `dict` of named feature tensors (shape `[B, T, D]`) from multiple encoder stages, including:
- Per-pixel spatial tokens (`layerN`)
- Global mean/std pooling (`layerN_mean`, `layerN_std`)
- Spatial patch-pooled variants (`layerN_mean_{size}`, `layerN_std_{size}`)
- Intermediate ResNet block outputs (`layerN_blkK`)

This rich multi-scale feature representation is what the drift loss operates on.

**`build_activation_function(mae_path, use_convnext, ...)`**: factory that builds the combined feature extraction callable (`activation_fn`) and loads its parameters. Supports MAE-only, ConvNeXt-only, or both in combination.

### `models/convnext.py` — ConvNextV2 (Optional Feature Extractor)

JAX reimplementation of ConvNeXtV2, loadable from HuggingFace PyTorch weights. Can be used as an additional (or alternative) feature extractor alongside MAE.

Key method: `get_activations(x)` — returns normalized multi-scale feature maps from all 4 stages, compatible with the drift loss input format.

`convert_weights_to_jax()`: converts PyTorch weight tensors to JAX/Flax parameter layout.

### `models/hf.py` — HuggingFace Artifact Helpers

Low-level helpers for downloading and loading model artifacts from HuggingFace Hub (`Goodeat/drifting`).

- `load_mae_jax(name, ...)`: downloads MAE artifact, reconstructs `MAEResNetJAX` from metadata, loads EMA params.
- `load_generator_jax(name, ...)`: downloads generator artifact, reconstructs `DitGen` from `model_config` in metadata.
- `_download_artifact(...)`: internal helper using `huggingface_hub.snapshot_download`.

---

## Dataset

### `dataset/dataset.py` — ImageNet Data Pipeline

Three operating modes controlled by config flags:

| Flag | Mode | Description |
|---|---|---|
| `use_cache=True` | Cached latent | Load precomputed `.pt` VAE latents from disk (fastest for latent training) |
| `use_latent=True` | Online latent | Encode RGB images through VAE at training time |
| neither | Pixel | Load and normalize raw RGB images |

**Main function**: `create_imagenet_split(resolution, batch_size, split, ...)` returns `(loader, preprocess_fn, postprocess_fn)`.

Shape convention: raw DataLoader outputs `BCHW`; JAX model tensors are `BHWC`.

Helper generators:
- `infinite_sampler(loader, start_step)`: infinite iterator that supports resuming from a given step.
- `epoch0_sampler(loader)`: one deterministic epoch for FID evaluation.

### `dataset/latent.py` — Latent Cache Builder

`create_cached_dataset(local_batch_size, target_path, data_path, ...)`: encodes the full ImageNet train/val splits through the VAE (both original and horizontally flipped) and writes per-image `.pt` files. Run once before latent-space generator training.

`LatentDataset`: `ImageFolder`-style dataset that reads the precomputed `.pt` files and randomly selects between the original and flipped latent at load time.

### `dataset/vae.py`
Wraps a Stable Diffusion VAE (loaded from Flax/HuggingFace). Provides `vae_enc_decode()` returning `(encode_fn, decode_fn)` for use in the data pipeline and `postprocess_fn`.

---

## Utilities (`utils/`)

| File | Purpose |
|---|---|
| `env.py` | Global path constants: `IMAGENET_PATH`, `IMAGENET_CACHE_PATH`, `IMAGENET_FID_NPZ`, `IMAGENET_PR_NPZ`, `HF_ROOT`, `HF_REPO_ID`. **Edit this before running.** |
| `model_builder.py` | `build_model_dict(config, model_class)`: one-stop factory that builds model + dataloaders + optimizer + LR schedule + logger from a config dict. |
| `hsdp_util.py` | Distributed training utilities: global mesh setup (`set_global_mesh`), data/DDP sharding specs, `init_state_from_dummy_input`, `merge_data`, `enforce_ddp`. |
| `ckpt_util.py` | Orbax-based checkpoint save/restore (`save_checkpoint`, `restore_checkpoint`) and EMA artifact saving (`save_params_ema_artifact`). |
| `init_util.py` | `maybe_init_state_params`: initialize model parameters from a local workdir or `hf://` artifact at training start. `load_generator_model_and_params`: for inference loading. |
| `fid_util.py` | `evaluate_fid(...)`: generates `num_samples` images and computes FID, IS, precision, and recall using precomputed reference stats. |
| `logging.py` | `WandbLogger`: unified logger that writes to W&B or a local `metrics.jsonl` file. `log_for_0` / `is_rank_zero`: rank-aware logging. |
| `misc.py` | `load_config` (YAML → `EasyDict`), `prepare_rng`, `profile_func`, `run_init`, `EasyDict`. |
| `jax_fid/` | JAX re-implementation of FID pipeline: `inception.py` (Inception-v3 features), `fid.py` (FID computation), `precision_recall.py`, `resize.py`, `cvt.py` (checkpoint conversion). |

---

## Configs (`configs/`)

### Generator configs (`configs/gen/`)

| File | Model | Space | Encoder | Notes |
|---|---|---|---|---|
| `latent_ablation.yaml` | Drift-B (small) | Latent | MAE-256 | Ablation run, 30k steps |
| `latent_sota_B.yaml` | Drift-B | Latent | MAE-640 | SOTA, FID 1.74 |
| `latent_sota_L.yaml` | Drift-L | Latent | MAE-640 | SOTA, FID 1.54 |
| `pixel_sota_B.yaml` | Drift-B | Pixel | MAE-640 | SOTA, FID 1.73 |
| `pixel_sota_L.yaml` | Drift-L | Pixel | MAE-640 | SOTA, FID 1.62 |

Each config has five sections: `dataset`, `model`, `optimizer`, `train`, `feature`.

### MAE configs (`configs/mae/`)

| File | Description |
|---|---|
| `latent_ablation_256.yaml` | Small MAE for latent ablation experiments |
| `latent_640.yaml` | Full MAE-640 for latent-space SOTA |
| `pixel_640.yaml` | Full MAE-640 for pixel-space SOTA |

---

## Checkpoint Layout

Each `--workdir <dir>` produces:
```
<dir>/
├── checkpoints/          # Full Orbax training state (model + optimizer + step)
├── params_ema/
│   ├── ema_params.msgpack  # EMA-only weights, loadable for inference
│   └── metadata.json       # Model config dict for reconstruction
└── log/
    ├── metrics.jsonl       # Training metrics (when W&B disabled)
    └── images/             # Sample preview grids saved during eval
```

`params_ema/` artifacts can be passed directly to `inference.py` via `--init-from /path/to/workdir`.

---

## Data Flow Summary (Generator Training)

```
ImageNet images
      │
      ▼
 DataLoader (dataset/)
      │  BCHW torch → BHWC jax
      ▼
 preprocess_fn ──(use_cache)──► LatentDataset (.pt files)
      │           └─(online)──► VAE encoder
      │
      ├──────────────────────────────────────►  ArrayMemoryBank
      │                                          (positive + negative)
      │                                                │
      │                          sample(labels, n) ◄──┘
      │                                │
      │              ┌─────────────────┴──────────────────┐
      │          positive_samples              negative_samples
      │                                │
      ▼                                ▼
  DitGen (generator)          MAE.get_activations()
  (noise → image)             (multi-scale features)
      │                                │
      ▼                                ▼
  gen_features ──────► drift_loss(gen, fixed_pos, fixed_neg)
                                       │
                                       ▼
                                  loss.backward()
                                       │
                                       ▼
                              update generator params
                              update EMA params
```