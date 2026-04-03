import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.optim import Adam
import types
# Use shared evaluation_metrics from baselines/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# from evaluation_metrics.fid import compute_fid
from evaluation_metrics.discriminative_torch import discriminative_score_metrics
from evaluation_metrics.ts2vec.context_fid import Context_FID


class Trainer:
    def __init__(self, config, args, nets, train_loader, valid_loader):
        """
        nets: dict with keys:
            encoder, recovery, generator, supervisor, discriminator
        """
        self.config = config
        self.args = args
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = nets["encoder"].to(self.device)
        self.recovery = nets["recovery"].to(self.device)
        self.generator = nets["generator"].to(self.device)
        self.supervisor = nets["supervisor"].to(self.device)
        self.discriminator = nets["discriminator"].to(self.device)

        solver_cfg = config["solver"]
        self.phase1_iters = solver_cfg["phase1_iters"]
        self.phase2_iters = solver_cfg["phase2_iters"]
        self.phase3_iters = solver_cfg["phase3_iters"]
        self.eval_every = solver_cfg.get("eval_every", 1000)
        self.save_every = solver_cfg.get("save_every", 10000)
        self.results_folder = Path(solver_cfg["results_folder"])
        os.makedirs(self.results_folder, exist_ok=True)

        # self.fid_vae_ckpt = config["fid_vae_ckpt"]

        m_cfg = config["model"]
        self.feature_size = m_cfg["feature_size"]
        self.w_gamma = m_cfg.get("w_gamma", 1.0)
        self.w_g = m_cfg.get("w_g", 1.0)

        lr = solver_cfg["lr"]
        beta1 = solver_cfg.get("beta1", 0.9)

        self.opt_e = Adam(self.encoder.parameters(), lr=lr, betas=(beta1, 0.999))
        self.opt_r = Adam(self.recovery.parameters(), lr=lr, betas=(beta1, 0.999))
        self.opt_g = Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.opt_s = Adam(self.supervisor.parameters(), lr=lr, betas=(beta1, 0.999))
        self.opt_d = Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

        self.l_mse = nn.MSELoss()
        self.l_bce = nn.BCELoss()

        wandb.init(
            project=os.getenv("WANDB_PROJECT", "no_name"),
            name=os.getenv("WANDB_NAME", "no_name"),
            config=config,
        )

    # ── checkpoint helpers ────────────────────────────────────────────────────

    def save(self, tag: str):
        ckpt_path = self.results_folder / f"checkpoint-{tag}.pt"
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "recovery": self.recovery.state_dict(),
                "generator": self.generator.state_dict(),
                "supervisor": self.supervisor.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "opt_e": self.opt_e.state_dict(),
                "opt_r": self.opt_r.state_dict(),
                "opt_g": self.opt_g.state_dict(),
                "opt_s": self.opt_s.state_dict(),
                "opt_d": self.opt_d.state_dict(),
            },
            ckpt_path,
        )

    def load(self, tag: str):
        ckpt_path = self.results_folder / f"checkpoint-{tag}.pt"
        data = torch.load(ckpt_path, map_location=self.device)

        self.encoder.load_state_dict(data["encoder"])
        self.recovery.load_state_dict(data["recovery"])
        self.generator.load_state_dict(data["generator"])
        self.supervisor.load_state_dict(data["supervisor"])
        self.discriminator.load_state_dict(data["discriminator"])

        self.opt_e.load_state_dict(data["opt_e"])
        self.opt_r.load_state_dict(data["opt_r"])
        self.opt_g.load_state_dict(data["opt_g"])
        self.opt_s.load_state_dict(data["opt_s"])
        self.opt_d.load_state_dict(data["opt_d"])

    # ── data helpers ──────────────────────────────────────────────────────────

    def _infinite_loader(self):
        """Yield batches from train_loader indefinitely."""
        while True:
            for batch in self.train_loader:
                yield batch[0].to(self.device)  # expected shape: (B, T, C)

    def _to_seq_first(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) -> (T, B, C) for GRU-style modules."""
        return x.permute(1, 0, 2)

    def _random_noise(self, seq_len: int, batch_size: int) -> torch.Tensor:
        """Standard Gaussian noise shaped as (T, B, feature_size)."""
        return torch.randn(seq_len, batch_size, self.feature_size, device=self.device)

    # ── evaluation helpers ────────────────────────────────────────────────────

    def _collect_eval_real(self) -> np.ndarray:
        batches = []
        for batch in self.valid_loader:
            batches.append(batch[0])
        return torch.cat(batches, dim=0).cpu().numpy()  # (N, T, C)

    def _generate(self, n_samples: int) -> np.ndarray:
        """Generate synthetic samples with shape (N, T, C)."""
        self.generator.eval()
        self.supervisor.eval()
        self.recovery.eval()

        all_samples = []

        with torch.no_grad():
            sample_batch = next(iter(self.valid_loader))[0]
            seq_len = sample_batch.shape[1]

            generated = 0
            while generated < n_samples:
                bs = min(128, n_samples - generated)

                z = self._random_noise(seq_len, bs)     # (T, B, C)
                e_hat = self.generator(z)               # (T, B, H)
                h_hat = self.supervisor(e_hat)          # (T, B, H)
                x_hat = self.recovery(h_hat)            # (T, B, C)
                x_hat = x_hat.permute(1, 0, 2)          # (B, T, C)

                all_samples.append(x_hat.cpu().numpy())
                generated += bs

        self.generator.train()
        self.supervisor.train()
        self.recovery.train()

        return np.concatenate(all_samples, axis=0)[:n_samples]

    def _eval_metrics(self, step: int, eval_real: np.ndarray):
        n_eval = len(eval_real)
        fake_np = self._generate(n_eval)

        # ── Save model checkpoint and generated samples ───────────────────────
        self.save(f'step{step}')
        samples_dir = self.results_folder / 'samples'
        samples_dir.mkdir(exist_ok=True)
        np.save(str(samples_dir / f'fake_step{step}.npy'), fake_np)

        real_t = torch.tensor(eval_real.astype(np.float32))
        fake_t = torch.tensor(fake_np.astype(np.float32))
        # fid_val = compute_fid(real_t, fake_t, ckpt_path=self.fid_vae_ckpt)["fid"]
        fid_val = Context_FID(real_t, fake_t)

        disc_val = discriminative_score_metrics(
            eval_real, fake_np,
            types.SimpleNamespace(input_size=feat_dim, device=self.device),
        )

        # disc_val = discriminative_score_metrics(
        #     eval_real,
        #     fake_np,
        #     input_size=self.feature_size,
        #     device=self.device,
        # )

        print(f"Step {step}: FID={fid_val:.4f}  Disc={disc_val:.4f}")
        wandb.log(
            {
                "eval/fid": fid_val,
                "eval/discriminative": disc_val,
                "step": step,
            }
        )

    # ── single-step optimizers ────────────────────────────────────────────────

    def _er_step_pretrain(self, x_t: torch.Tensor):
        """
        Phase 1:
        Encoder + Recovery pre-training
        """
        h = self.encoder(x_t)          # (T, B, H)
        x_tilde = self.recovery(h)     # (T, B, C)

        loss_er = self.l_mse(x_tilde, x_t)

        self.opt_e.zero_grad()
        self.opt_r.zero_grad()
        loss_er.backward()
        self.opt_e.step()
        self.opt_r.step()

        return loss_er.detach()

    def _s_step_pretrain(self, x_t: torch.Tensor):
        """
        Phase 2:
        Supervisor pre-training
        """
        h = self.encoder(x_t)          # (T, B, H)
        h_sup = self.supervisor(h)     # (T, B, H)

        loss_s = self.l_mse(h[1:], h_sup[:-1])

        self.opt_s.zero_grad()
        loss_s.backward()
        self.opt_s.step()

        return loss_s.detach()

    def _g_step(self, x_t: torch.Tensor):
        """
        Phase 3 generator/supervisor update.
        Consistent with official TimeGAN logic:
        - use encoder output as target only
        - do NOT update encoder here
        """
        seq_len, batch_size, _ = x_t.shape
        z = self._random_noise(seq_len, batch_size)

        # Real latent path: encoder output is target only, so detach it.
        h = self.encoder(x_t).detach()     # (T, B, H), frozen target
        h_sup = self.supervisor(h)         # (T, B, H)

        # Fake path
        e_hat = self.generator(z)          # (T, B, H)
        h_hat = self.supervisor(e_hat)     # (T, B, H)
        x_hat = self.recovery(h_hat)       # (T, B, C)

        y_fake = self.discriminator(h_hat)
        y_fake_e = self.discriminator(e_hat)

        # Official-style generator loss
        loss_g_adv = self.l_bce(y_fake, torch.ones_like(y_fake))
        loss_g_adv_e = self.l_bce(y_fake_e, torch.ones_like(y_fake_e))

        # Match mean/std in data space across batch dimension
        loss_g_v1 = torch.mean(
            torch.abs(
                torch.sqrt(torch.var(x_hat, dim=1, unbiased=False) + 1e-6)
                - torch.sqrt(torch.var(x_t, dim=1, unbiased=False) + 1e-6)
            )
        )
        loss_g_v2 = torch.mean(
            torch.abs(torch.mean(x_hat, dim=1) - torch.mean(x_t, dim=1))
        )

        loss_s = self.l_mse(h[1:], h_sup[:-1])

        loss_g = (
            loss_g_adv
            + self.w_gamma * loss_g_adv_e
            + self.w_g * loss_g_v1
            + self.w_g * loss_g_v2
            + torch.sqrt(loss_s)
        )

        # Match official behavior: only update G and S
        self.opt_g.zero_grad()
        self.opt_s.zero_grad()
        loss_g.backward()
        self.opt_g.step()
        self.opt_s.step()

        return loss_g.detach()

    def _er_step_joint(self, x_t: torch.Tensor):
        """
        Phase 3 encoder/recovery update with supervisor regularization.
        Must use a fresh forward pass, consistent with official code.
        """
        h = self.encoder(x_t)              # fresh forward
        x_tilde = self.recovery(h)
        h_sup = self.supervisor(h)         # fresh forward

        loss_er_recon = self.l_mse(x_tilde, x_t)
        loss_er_sup = self.l_mse(h[1:], h_sup[:-1])

        loss_er = 10.0 * torch.sqrt(loss_er_recon) + 0.1 * loss_er_sup

        self.opt_e.zero_grad()
        self.opt_r.zero_grad()
        loss_er.backward()
        self.opt_e.step()
        self.opt_r.step()

        return loss_er.detach()

    def _d_step(self, x_t: torch.Tensor):
        """
        Phase 3 discriminator update.
        The official code only updates D here.
        """
        seq_len, batch_size, _ = x_t.shape
        z = self._random_noise(seq_len, batch_size)

        # Upstream networks provide features; detach to avoid unnecessary grads.
        with torch.no_grad():
            h = self.encoder(x_t)
            e_hat = self.generator(z)
            h_hat = self.supervisor(e_hat)

        y_real = self.discriminator(h)
        y_fake = self.discriminator(h_hat)
        y_fake_e = self.discriminator(e_hat)

        loss_d_real = self.l_bce(y_real, torch.ones_like(y_real))
        loss_d_fake = self.l_bce(y_fake, torch.zeros_like(y_fake))
        loss_d_fake_e = self.l_bce(y_fake_e, torch.zeros_like(y_fake_e))

        loss_d = loss_d_real + loss_d_fake + self.w_gamma * loss_d_fake_e

        # Same threshold rule as official code
        if loss_d.item() > 0.15:
            self.opt_d.zero_grad()
            loss_d.backward()
            self.opt_d.step()

        return loss_d.detach()

    # ── training phases ───────────────────────────────────────────────────────

    def _phase1(self):
        """Phase 1: encoder + recovery pre-training."""
        print("\n=== Phase 1: Encoder / Recovery pre-training ===")
        loader = self._infinite_loader()

        self.encoder.train()
        self.recovery.train()

        for step in range(1, self.phase1_iters + 1):
            x = next(loader)               # (B, T, C)
            x_t = self._to_seq_first(x)    # (T, B, C)

            loss_er = self._er_step_pretrain(x_t)

            if step % 1000 == 0:
                print(
                    f"  Phase1 step {step:6d}/{self.phase1_iters} | "
                    f"recon={loss_er.item():.6f}"
                )
                wandb.log(
                    {
                        "phase1/recon_loss": loss_er.item(),
                        "phase1_step": step,
                    }
                )

    def _phase2(self):
        """Phase 2: supervisor pre-training."""
        print("\n=== Phase 2: Supervisor pre-training ===")
        loader = self._infinite_loader()

        self.encoder.train()
        self.supervisor.train()

        for step in range(1, self.phase2_iters + 1):
            x = next(loader)
            x_t = self._to_seq_first(x)

            loss_s = self._s_step_pretrain(x_t)

            if step % 1000 == 0:
                print(
                    f"  Phase2 step {step:6d}/{self.phase2_iters} | "
                    f"sup={loss_s.item():.6f}"
                )
                wandb.log(
                    {
                        "phase2/supervisor_loss": loss_s.item(),
                        "phase2_step": step,
                    }
                )

    def _phase3(self, eval_real: np.ndarray):
        """
        Phase 3:
        Joint adversarial training with official TimeGAN schedule:
            for each iteration:
                2 x [G step, ER step]
                1 x [D step]
        """
        print("\n=== Phase 3: Joint GAN training ===")
        loader = self._infinite_loader()

        self.encoder.train()
        self.recovery.train()
        self.generator.train()
        self.supervisor.train()
        self.discriminator.train()

        for step in range(1, self.phase3_iters + 1):
            tic = time.time()

            last_loss_g = None
            last_loss_er = None

            # 2 x (G + ER)
            for _ in range(2):
                # G step uses its own fresh batch, matching official style
                x_g = next(loader)
                x_g_t = self._to_seq_first(x_g)
                last_loss_g = self._g_step(x_g_t)

                # ER step uses its own fresh batch, matching official style
                x_er = next(loader)
                x_er_t = self._to_seq_first(x_er)
                last_loss_er = self._er_step_joint(x_er_t)

            # 1 x D
            x_d = next(loader)
            x_d_t = self._to_seq_first(x_d)
            loss_d = self._d_step(x_d_t)

            elapsed = time.time() - tic

            if step % 100 == 0:
                print(
                    f"  Phase3 step {step:6d}/{self.phase3_iters} | "
                    f"G={last_loss_g.item():.4f}  "
                    f"ER={last_loss_er.item():.4f}  "
                    f"D={loss_d.item():.4f} | "
                    f"t={elapsed:.2f}s"
                )
                wandb.log(
                    {
                        "phase3/loss_g": last_loss_g.item(),
                        "phase3/loss_er": last_loss_er.item(),
                        "phase3/loss_d": loss_d.item(),
                        "phase3_step": step,
                    }
                )

            if step % self.save_every == 0:
                self.save(f"step{step}")

            if step % self.eval_every == 0:
                self._eval_metrics(step, eval_real)

    # ── main entry point ──────────────────────────────────────────────────────

    def train(self):
        eval_real = self._collect_eval_real()

        self._phase1()
        self._phase2()
        self._phase3(eval_real)

        self.save("final")
        print("\nTraining complete.")