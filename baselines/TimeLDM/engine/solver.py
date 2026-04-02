"""
Training solvers for TimeLDM: VAETrainer and LDMTrainer.

Stage 1 (VAETrainer): Train the β-VAE.
Stage 2 (LDMTrainer): Freeze the VAE encoder, train the MLP denoiser in latent space,
                       evaluate with Context-FID and discriminative score.
"""

import os
import sys
import time
import copy
import torch
import numpy as np
import wandb

from pathlib import Path
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

# Allow running from the baselines/TimeLDM/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from losses import vae_loss, ldm_loss
from models.ldm import sample_time, add_noise
from evaluation_metrics.discriminative_torch import discriminative_score_metrics
from evaluation_metrics.context_fid import Context_FID


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: VAE Trainer
# ─────────────────────────────────────────────────────────────────────────────

class VAETrainer:
    """
    Trains the β-VAE (encoder + decoder).
    Saves checkpoint in `solver.vae.results_folder`.
    """

    def __init__(self, config, args, vae, dataloader):
        self.vae = vae
        self.device = next(vae.parameters()).device
        self.args = args
        self.config = config

        vae_cfg = config['solver']['vae']
        self.max_epochs   = vae_cfg['max_epochs']
        self.eval_every   = vae_cfg.get('eval_every', 100)
        self.lambda1      = vae_cfg.get('lambda1', 1.0)
        self.lambda2      = vae_cfg.get('lambda2', 1.0)
        self.lambda3      = vae_cfg.get('lambda3', 1.0)

        self.results_folder = Path(vae_cfg['results_folder'])
        os.makedirs(self.results_folder, exist_ok=True)

        lr = vae_cfg.get('base_lr', 1e-3)
        self.opt = Adam(
            filter(lambda p: p.requires_grad, vae.parameters()), lr=lr
        )

        self.train_dataloader = dataloader['train_dataloader']
        self.valid_dataloader = dataloader['valid_dataloader']
        self.step = 0

        wandb.init(
            project=os.getenv('WANDB_PROJECT', 'timeldm-vae'),
            name=os.getenv('WANDB_NAME', 'vae'),
            config=config,
        )

    # ── Checkpoint helpers ───────────────────────────────────────────────────

    def save(self, milestone):
        torch.save(
            {
                'step':    self.step,
                'vae':     self.vae.state_dict(),
                'opt':     self.opt.state_dict(),
                'beta':    self.vae.beta,
            },
            str(self.results_folder / f'checkpoint-{milestone}.pt'),
        )

    def load(self, milestone):
        data = torch.load(
            str(self.results_folder / f'checkpoint-{milestone}.pt'),
            map_location=self.device,
        )
        self.vae.load_state_dict(data['vae'])
        self.opt.load_state_dict(data['opt'])
        self.step = data['step']
        self.vae.beta = data.get('beta', self.vae.beta)

    # ── Training loop ────────────────────────────────────────────────────────

    def train(self):
        best_val_loss = float('inf')

        for epoch in range(self.max_epochs):
            self.vae.train()
            train_total_avg = 0.0
            train_recon_avg = 0.0
            train_kl_avg    = 0.0
            tic = time.time()

            for batch in self.train_dataloader:
                batch = batch.to(self.device)
                x_hat, mu, log_sigma, z = self.vae(batch)
                total, recon, kl = vae_loss(
                    batch, x_hat, mu, log_sigma,
                    beta=self.vae.beta,
                    lambda1=self.lambda1,
                    lambda2=self.lambda2,
                    lambda3=self.lambda3,
                )
                total.backward()
                clip_grad_norm_(self.vae.parameters(), 1.0)
                self.opt.step()
                self.opt.zero_grad()
                self.vae.update_beta(recon.item())
                self.step += 1
                train_total_avg += total.item()
                train_recon_avg += recon.item()
                train_kl_avg    += kl.item()

            n = len(self.train_dataloader)
            train_total_avg /= n
            train_recon_avg /= n
            train_kl_avg    /= n
            toc = time.time()

            print(
                f"[VAE] Epoch {epoch:5d}: "
                f"loss={train_total_avg:.5f} "
                f"recon={train_recon_avg:.5f} "
                f"kl={train_kl_avg:.5f} "
                f"beta={self.vae.beta:.2e} "
                f"({toc - tic:.1f}s)"
            )
            wandb.log({
                'train/loss':  train_total_avg,
                'train/recon': train_recon_avg,
                'train/kl':    train_kl_avg,
                'train/beta':  self.vae.beta,
                'train/lr':    self.opt.param_groups[0]['lr'],
                'epoch':       epoch,
            })

            if epoch % self.eval_every == 0:
                val_total_avg = 0.0
                val_recon_avg = 0.0
                val_kl_avg    = 0.0
                self.vae.eval()
                for batch in self.valid_dataloader:
                    batch = batch.to(self.device)
                    with torch.no_grad():
                        x_hat, mu, log_sigma, z = self.vae(batch)
                        total, recon, kl = vae_loss(
                            batch, x_hat, mu, log_sigma,
                            beta=self.vae.beta,
                            lambda1=self.lambda1,
                            lambda2=self.lambda2,
                            lambda3=self.lambda3,
                        )
                    val_total_avg += total.item()
                    val_recon_avg += recon.item()
                    val_kl_avg    += kl.item()

                n_val = len(self.valid_dataloader)
                val_total_avg /= n_val
                val_recon_avg /= n_val
                val_kl_avg    /= n_val

                print(
                    f"[VAE] Epoch {epoch:5d} [val]: "
                    f"loss={val_total_avg:.5f} "
                    f"recon={val_recon_avg:.5f} "
                    f"kl={val_kl_avg:.5f}"
                )
                wandb.log({
                    'valid/loss':  val_total_avg,
                    'valid/recon': val_recon_avg,
                    'valid/kl':    val_kl_avg,
                    'epoch':       epoch,
                })

                if val_total_avg < best_val_loss:
                    best_val_loss = val_total_avg
                    self.save('best')


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: LDM Trainer
# ─────────────────────────────────────────────────────────────────────────────

class LDMTrainer:
    """
    Trains the MLP denoiser in latent space with the VAE encoder frozen.

    Sampling uses Euler denoising (DDIM-like, deterministic) with the EDM
    schedule sigma(t) = t.  Generated latents are decoded by the frozen VAE.

    Evaluates with Context-FID and discriminative score every `eval_every` epochs.
    """

    def __init__(self, config, args, vae, ldm_denoiser, dataloader):
        self.vae = vae
        self.ldm = ldm_denoiser
        self.device = next(ldm_denoiser.parameters()).device
        self.args = args
        self.config = config

        # Freeze VAE completely
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.vae.eval()

        ldm_cfg = config['solver']['ldm']
        self.max_epochs        = ldm_cfg['max_epochs']
        self.eval_every        = ldm_cfg.get('eval_every', 500)
        self.t_min             = ldm_cfg.get('t_min', 0.01)
        self.t_max             = ldm_cfg.get('t_max', 1.0)
        self.num_sampling_steps = ldm_cfg.get('num_sampling_steps', 200)

        self.results_folder = Path(ldm_cfg['results_folder'])
        os.makedirs(self.results_folder, exist_ok=True)

        lr = ldm_cfg.get('base_lr', 1e-4)
        self.opt = Adam(
            filter(lambda p: p.requires_grad, ldm_denoiser.parameters()), lr=lr,
            betas=(0.9, 0.96),
        )

        self.ema_decay = 0.999
        self.ema_ldm = copy.deepcopy(ldm_denoiser).to(self.device)
        self.ema_ldm.eval()

        self.train_dataloader = dataloader['train_dataloader']
        self.valid_dataloader = dataloader['valid_dataloader']
        self.step = 0

        wandb.init(
            project=os.getenv('WANDB_PROJECT', 'timeldm-ldm'),
            name=os.getenv('WANDB_NAME', 'ldm'),
            config=config,
        )

    # ── Checkpoint helpers ───────────────────────────────────────────────────

    def save(self, milestone):
        torch.save(
            {
                'step':    self.step,
                'ldm':     self.ldm.state_dict(),
                'ema_ldm': self.ema_ldm.state_dict(),
                'opt':     self.opt.state_dict(),
            },
            str(self.results_folder / f'checkpoint-{milestone}.pt'),
        )

    def load(self, milestone):
        data = torch.load(
            str(self.results_folder / f'checkpoint-{milestone}.pt'),
            map_location=self.device,
        )
        self.ldm.load_state_dict(data['ldm'])
        self.ema_ldm.load_state_dict(data['ema_ldm'])
        self.opt.load_state_dict(data['opt'])
        self.step = data['step']

    # ── EMA update ───────────────────────────────────────────────────────────

    def _update_ema(self):
        with torch.no_grad():
            for ema_p, p in zip(self.ema_ldm.parameters(), self.ldm.parameters()):
                ema_p.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)

    # ── Sampling ─────────────────────────────────────────────────────────────

    def _denoise_latents(self, batch_size: int) -> torch.Tensor:
        """
        Euler denoising from z_T ~ N(0, t_max^2 * I) to z_{t_min}.

        Update rule: z_{t-dt} = z_t - dt * eps_theta(z_t, t)
        (derived from d z_t / d t = eps, which comes from z_t = z_0 + t * eps)
        """
        seq_len    = self.ldm.seq_len
        latent_dim = self.ldm.latent_dim

        z = torch.randn(batch_size, seq_len, latent_dim, device=self.device) * self.t_max

        ts = torch.linspace(self.t_max, self.t_min, self.num_sampling_steps + 1,
                            device=self.device)

        self.ema_ldm.eval()
        with torch.no_grad():
            for i in range(self.num_sampling_steps):
                t_curr = ts[i]
                t_next = ts[i + 1]
                dt = t_next - t_curr                     # negative (denoising direction)
                t_batch = t_curr.expand(batch_size)
                eps_pred = self.ema_ldm(z, t_batch)
                z = z + dt * eps_pred                    # Euler step

        return z

    def sample(self, num: int, size_every: int, shape=None) -> np.ndarray:
        """
        Generate `num` samples.  `shape = [seq_len, feature_size]`.
        Returns (num, seq_len, feature_size) numpy array.
        """
        seq_len, feature_size = shape[0], shape[1]
        samples = np.empty([0, seq_len, feature_size])
        num_cycle = int(num // size_every) + 1
        tic = time.time()

        for _ in range(num_cycle):
            z0 = self._denoise_latents(size_every)
            with torch.no_grad():
                x_hat = self.vae.decode(z0)              # (B, seq_len, feature_size)
            samples = np.row_stack([samples, x_hat.detach().cpu().numpy()])
            torch.cuda.empty_cache()

        print(f'Sampling done, time: {time.time() - tic:.2f}s')
        return samples

    # ── Evaluation ───────────────────────────────────────────────────────────

    def _collect_eval_real(self, max_samples: int = 512) -> np.ndarray:
        batches, collected = [], 0
        for batch in self.valid_dataloader:
            batches.append(batch)
            collected += batch.shape[0]
            if collected >= max_samples:
                break
        return torch.cat(batches, dim=0)[:max_samples].numpy()

    def _eval_metrics(self, epoch: int, real_np: np.ndarray):
        window, var_num = real_np.shape[1], real_np.shape[2]
        n_eval = len(real_np)

        fake_np = self.sample(n_eval, size_every=128, shape=[window, var_num])[:n_eval]

        real_t = torch.tensor(real_np.astype(np.float32))
        fake_t = torch.tensor(fake_np.astype(np.float32))
        fid_val  = Context_FID(real_t, fake_t)
        disc_val = discriminative_score_metrics(
            real_np, fake_np, input_size=var_num, device=self.device,
        )

        print(f"[LDM] Epoch {epoch}: Context-FID={fid_val:.4f}  Disc={disc_val:.4f}")
        wandb.log({
            'eval/context_fid':      fid_val,
            'eval/discriminative':   disc_val,
            'epoch':                 epoch,
        })

    # ── Training loop ────────────────────────────────────────────────────────

    def train(self):
        eval_real = self._collect_eval_real(max_samples=512)
        best_val_loss = float('inf')

        for epoch in range(self.max_epochs):
            self.ldm.train()
            train_loss_avg = 0.0
            tic = time.time()

            for batch in self.train_dataloader:
                batch = batch.to(self.device)

                # Encode with frozen VAE (use sampled z, not mu, for diversity)
                with torch.no_grad():
                    z, mu, log_sigma = self.vae.encode(batch)  # (B, T, m)

                t      = sample_time(batch.shape[0], self.t_min, self.t_max, self.device)
                z_t, eps = add_noise(z, t)
                eps_pred = self.ldm(z_t, t)
                loss     = ldm_loss(eps_pred, eps)

                loss.backward()
                clip_grad_norm_(self.ldm.parameters(), 1.0)
                self.opt.step()
                self.opt.zero_grad()
                self._update_ema()
                self.step += 1
                train_loss_avg += loss.item()

            train_loss_avg /= len(self.train_dataloader)
            toc = time.time()

            print(
                f"[LDM] Epoch {epoch:5d}: "
                f"loss={train_loss_avg:.6f} "
                f"({toc - tic:.1f}s)"
            )
            wandb.log({
                'train/loss': train_loss_avg,
                'train/lr':   self.opt.param_groups[0]['lr'],
                'epoch':      epoch,
            })

            if epoch % self.eval_every == 0:
                # ── Validation loss ───────────────────────────────────────────
                val_loss_avg = 0.0
                self.ema_ldm.eval()
                for batch in self.valid_dataloader:
                    batch = batch.to(self.device)
                    with torch.no_grad():
                        z, mu, log_sigma = self.vae.encode(batch)
                        t        = sample_time(batch.shape[0], self.t_min, self.t_max, self.device)
                        z_t, eps = add_noise(z, t)
                        eps_pred = self.ema_ldm(z_t, t)
                        loss     = ldm_loss(eps_pred, eps)
                    val_loss_avg += loss.item()
                val_loss_avg /= len(self.valid_dataloader)

                print(f"[LDM] Epoch {epoch:5d} [val]: loss={val_loss_avg:.6f}")
                wandb.log({'valid/loss': val_loss_avg, 'epoch': epoch})

                if val_loss_avg < best_val_loss:
                    best_val_loss = val_loss_avg
                    self.save('best')

                # ── FID + discriminative score ────────────────────────────────
                self._eval_metrics(epoch, eval_real)
