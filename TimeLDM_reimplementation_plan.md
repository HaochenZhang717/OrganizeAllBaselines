# TimeLDM Reimplementation Plan

**Paper**: TimeLDM: Latent Diffusion Model for Unconditional Time Series Generation  
**arXiv**: 2407.04211v2

---

## Architecture Summary

TimeLDM has two sequential training stages:
1. **Stage 1**: Train a β-VAE (Transformer encoder + decoder) to map time series → latent space
2. **Stage 2**: Train a latent diffusion model (MLP denoiser) on the frozen latent space

At inference, sample from the LDM and decode with the frozen VAE decoder.

---

## Step 1: Project Structure

```
timeldm/
├── data/
│   ├── __init__.py
│   ├── datasets.py        # dataset loading & normalization
│   └── utils.py           # windowing, train/test split
├── models/
│   ├── __init__.py
│   ├── encoder.py         # Transformer VAE encoder
│   ├── decoder.py         # Transformer VAE decoder
│   ├── vae.py             # full β-VAE wrapper
│   └── ldm.py             # latent diffusion MLP denoiser
├── losses.py              # Lrecon (L1 + L2 + FFT) + KL loss
├── train_vae.py           # Stage 1 training script
├── train_ldm.py           # Stage 2 training script
├── sample.py              # inference / generation script
├── eval/
│   ├── context_fid.py     # Context-FID score
│   ├── discriminative.py  # Discriminative score
│   ├── correlational.py   # Correlational score
│   └── predictive.py      # Predictive score
└── configs/
    ├── sines.yaml
    ├── mujoco.yaml
    ├── stocks.yaml
    ├── etth.yaml
    └── fmri.yaml
```

**Dependencies**: `torch`, `numpy`, `scipy`, `scikit-learn`, `tqdm`, `pyyaml`

---

## Step 2: Data Pipeline

1. **Datasets** (input shape: `(N, τ, d)`):
   | Dataset | Features `d` | Source |
   |---------|-------------|--------|
   | Sines   | 5           | Simulated sinusoids (TimeGAN setup) |
   | MuJoCo  | 14          | dm_control physics simulation |
   | Stocks  | 6           | Google stock prices 2004–2019 |
   | ETTh    | 7           | Electricity transformer (15-min) |
   | fMRI    | 50          | BOLD time series simulation |

2. **Preprocessing**:
   - Normalize each feature to `[0, 1]` (min-max) following TimeGAN convention
   - Slice into non-overlapping windows of length `τ` (default `τ=24`; also test `τ=64, 128` for ETTh)
   - Split into train/test sets

3. **DataLoader**: return batches of shape `(B, τ, d)`; batch sizes per Table I: 1024 for most, 512 for Stocks.

---

## Step 3: VAE Encoder (`encoder.py`)

Input: `x ∈ ℝ^{τ×d}` → Output: `μ ∈ ℝ^{τ×m}`, `σ ∈ ℝ^{τ×m}`

1. **Embedding layer**: `nn.Conv1d(d, m, kernel_size=1)` → shape `(τ, m)`  
   *(1D conv over feature dim acts as a linear projection per timestep)*

2. **Learnable positional encoding**: `pe = nn.Parameter(torch.randn(τ, m))`  
   Add to embedding: `e_pe = emb(x) + pe`

3. **Two parallel Transformer encoder stacks** (independent weights, shared input `e_pe`):
   - Each stack has `N` identical layers:
     ```
     Self-Attention (heads=2, head_dim=16) → Add & Norm → FeedForward → Add & Norm
     ```
   - `N` per dataset (Table I): 1 for Sines/MuJoCo/fMRI, 2 for Stocks/ETTh
   - Stack 1 → linear projection → `μ ∈ ℝ^{τ×m}`
   - Stack 2 → linear projection → `log σ ∈ ℝ^{τ×m}`

4. **Reparameterization**:
   ```python
   def reparameterize(mu, log_sigma):
       eps = torch.randn_like(mu)
       return mu + eps * torch.exp(log_sigma)
   ```

---

## Step 4: VAE Decoder (`decoder.py`)

Input: `z ∈ ℝ^{τ×m}` → Output: `x̂ ∈ ℝ^{τ×d}`

1. **Embedding + positional encoding**: same as encoder (independent weights)

2. **M Transformer decoder layers**, each with:
   ```
   Self-Attention → Add & Norm → Cross-Attention → Add & Norm → FeedForward → Add & Norm
   ```
   - Cross-attention: query from current layer, key/value from `z` (the latent input)
   - `M` per dataset (Table I): 2 for Sines/MuJoCo/fMRI, 3 for Stocks/ETTh

3. **Output projection**: `nn.Conv1d(m, d, kernel_size=1)` → shape `(τ, d)`

---

## Step 5: Loss Functions (`losses.py`)

### Reconstruction Loss
```python
def reconstruction_loss(x, x_hat, lambda1=1.0, lambda2=1.0, lambda3=1.0):
    l2 = F.mse_loss(x, x_hat)                          # L2 norm
    l1 = F.l1_loss(x, x_hat)                           # L1 norm
    fft_loss = torch.mean(torch.abs(torch.fft.fft(x, dim=1) - torch.fft.fft(x_hat, dim=1)))  # FFT
    return lambda1 * l2 + lambda2 * l1 + lambda3 * fft_loss
```

### KL Divergence Loss
```python
def kl_loss(mu, log_sigma):
    # KL(q(z|x) || N(0,I))
    return -0.5 * torch.mean(1 + 2*log_sigma - mu.pow(2) - (2*log_sigma).exp())
```

### Total VAE Loss
```python
total_loss = reconstruction_loss(x, x_hat) + beta * kl_loss(mu, log_sigma)
```

### Adaptive β Schedule
- Initialize `β = β_max = 1e-2`
- Monitor `L_recon` every `S` steps; if it fails to decrease: `β ← β * λ` where `λ = 0.7`
- Minimum value: `β_min = 1e-5`

---

## Step 6: Stage 1 — Train VAE (`train_vae.py`)

```
for each epoch:
    sample x from dataset
    mu, log_sigma = encoder(x)
    z = reparameterize(mu, log_sigma)
    x_hat = decoder(z)
    loss = Lrecon(x, x_hat) + beta * KL(mu, log_sigma)
    loss.backward()
    optimizer.step()
    if Lrecon not decreasing for S steps:
        beta *= 0.7  (clamped to beta_min)
```

- Optimizer: Adam, default betas `(0.9, 0.999)`, lr = `1e-3`
- Train until convergence; save encoder and decoder weights
- **Freeze** encoder and decoder after this stage

---

## Step 7: Latent Diffusion Model (`ldm.py`)

This is a **score-based diffusion** model (EDM/Karras et al. formulation).  
The noise schedule follows `σ(t) = t` (Karras et al. 2022).

**Forward process** (adding noise):
```
z_t = z_0 + σ(t) · ε,   ε ~ N(0, I),   σ(t) = t
```

**Denoising network** `ε_θ(z_t, t)` — MLP architecture (Fig. 2c):

1. **Reshape** `z ∈ ℝ^{τ×m}` → `ℝ^{τ·m}` (flatten to 1D)
2. **Linear layer**: `τ·m → hidden_dim`
3. **Sinusoidal time embedding**: encode scalar `t` as sinusoidal features `t_emb ∈ ℝ^{hidden_dim}`, add to the linear output
4. **4 Linear layers** with activations (e.g., SiLU/GELU) to learn the denoising pattern
5. **Linear output layer**: `hidden_dim → τ·m`
6. **Reshape** back to `ℝ^{τ×m}`

Hidden dimensions per dataset (Table I): 1024 for Sines/Stocks/ETTh, 4096 for MuJoCo/fMRI.

**Training objective**:
```
L_LDM = E_{z0, t, ε} [ || ε_θ(z_t, t) - ε ||²₂ ]
```

---

## Step 8: Stage 2 — Train LDM (`train_ldm.py`)

```
load frozen encoder

for each epoch:
    sample x from dataset
    z0 = encoder(x).sample()          # use reparameterization
    t ~ Uniform(t_min, T)
    eps ~ N(0, I)
    z_t = z0 + sigma(t) * eps
    eps_pred = denoiser(z_t, t)
    loss = MSE(eps_pred, eps)
    loss.backward()
    optimizer.step()
```

- Optimizer: Adam, betas `(0.9, 0.96)`, lr = `1e-4`
- `T` (max noise level): tune per dataset, e.g., `T=80` following EDM convention
- Train until convergence; save denoiser weights

---

## Step 9: Sampling / Inference (`sample.py`)

Follows the reverse SDE (Algorithm 2 in the paper):

```
1. Sample z_T ~ N(0, σ²(T)·I)
2. For t = T, T-1, ..., 1:
     score = -eps_theta(z_t, t) / sigma(t)
     Solve reverse SDE step to get z_{t-1}
        (use Euler-Maruyama or a deterministic ODE solver / DDIM-style)
3. Decode: x_hat = decoder(z_0)
```

Use the stochastic reverse SDE from Song et al. 2020:
```
dz = -2·σ̇(t)·σ(t)·∇log p(z_t) dt + sqrt(2·σ̇(t)·σ(t)) dω_t
```
where `∇log p(z_t) = -ε_θ(z_t, t) / σ(t)`.

---

## Step 10: Evaluation Metrics (`eval/`)

Implement or reuse from existing repos (TimeGAN, Diffusion-TS):

| Metric | Description | Implementation |
|--------|-------------|----------------|
| **Context-FID** | Fréchet distance of local context representations | Use InceptionTime encoder on windows, compute FID |
| **Correlational Score** | Mean absolute error between cross-correlation matrices of real vs. synthetic | `np.corrcoef` on features, average L1 diff |
| **Discriminative Score** | Train/test classifier to distinguish real vs. fake; report `|0.5 - accuracy|` | 2-layer LSTM or GRU classifier |
| **Predictive Score** | Train on synthetic, test on real (TSTR); report MAE | 2-layer LSTM predictor |

For qualitative evaluation:
- **t-SNE**: project real and generated samples into 2D, visualize overlap
- **Kernel Density Estimation**: compare marginal distributions per feature

---

## Step 11: Hyperparameter Configuration

Per Table I:

| Parameter | Sines | MuJoCo | Stocks | ETTh | fMRI |
|-----------|-------|--------|--------|------|------|
| `d` (input dim) | 5 | 14 | 6 | 7 | 50 |
| Attention heads | 2 | 2 | 2 | 2 | 2 |
| Head dim | 16 | 16 | 16 | 16 | 16 |
| Encoder layers | 1 | 1 | 2 | 2 | 1 |
| Decoder layers | 2 | 2 | 3 | 3 | 2 |
| Batch size | 1024 | 1024 | 512 | 1024 | 1024 |
| LDM hidden dim | 1024 | 4096 | 1024 | 1024 | 4096 |
| Latent dim `m` | (tune) | (tune) | (tune) | (tune) | (tune) |

VAE β: `β_max=1e-2`, `β_min=1e-5`, `λ=0.7`  
VAE lr: `1e-3` | LDM lr: `1e-4`  
Hardware: NVIDIA RTX 4080 (or equivalent)

---

## Step 12: Training Pipeline (End-to-End)

```bash
# Stage 1: Train VAE
python train_vae.py --config configs/stocks.yaml --epochs 2000

# Stage 2: Train LDM (encoder weights frozen)
python train_ldm.py --config configs/stocks.yaml --vae_ckpt checkpoints/vae_stocks.pt --epochs 10000

# Generate samples
python sample.py --config configs/stocks.yaml --ldm_ckpt checkpoints/ldm_stocks.pt --n_samples 1000

# Evaluate
python eval/run_all.py --real_data data/stocks_test.npy --fake_data samples/stocks_generated.npy
```

---

## Step 13: Ablation Studies to Validate

1. **Adaptive β vs. fixed β**: compare `β=1e-2` (fixed), `β=1e-5` (fixed), vs. adaptive schedule
2. **Loss term ablation**: train VAE with each of the following removed:
   - w/o FFT loss term
   - w/o L1 norm
   - w/o L2 norm

---

## Key Implementation Notes

- The **latent dimension `m`** is not explicitly stated in the paper — a reasonable starting point is `m = d` (same as input) or `m = 32/64`; tune empirically
- The **cross-attention** in the decoder uses `z` (the latent) as keys/values and the decoder's current hidden state as queries
- The LDM uses a **continuous-time** score-based formulation (Song et al. 2020 + Karras et al. 2022), not the discrete DDPM formulation — use `σ(t) = t` and an ODE/SDE solver for sampling
- The paper trains on **sequence length 24** for main results, and also evaluates on lengths **64 and 128** (ETTh only)
- Reconstruction loss weights `λ1, λ2, λ3` are not specified — start with equal weights `1.0` and tune
