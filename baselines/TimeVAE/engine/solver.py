import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from torch.optim import Adam
import wandb
from evaluation_metrics.discriminative_torch import discriminative_score_metrics
from evaluation_metrics.ts2vec.context_fid import Context_FID


class Trainer:
    def __init__(self, config, args, model, train_loader, valid_loader):
        self.config = config
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        solver_cfg = config['solver']
        self.max_epochs  = solver_cfg['max_epochs']
        self.eval_every  = solver_cfg.get('eval_every', 100)
        self.save_every  = solver_cfg.get('save_every', 500)
        self.results_folder = Path(solver_cfg['results_folder'])
        os.makedirs(self.results_folder, exist_ok=True)

        self.optimizer = Adam(self.model.parameters(), lr=solver_cfg['lr'])

        wandb.init(
            project=os.getenv('WANDB_PROJECT', 'no_name'),
            name=os.getenv('WANDB_NAME', 'no_name'),
            config=config,
        )

    # ── checkpoint helpers ────────────────────────────────────────────────────

    def save(self, tag: str):
        torch.save(
            {'model': self.model.state_dict(), 'opt': self.optimizer.state_dict()},
            self.results_folder / f'checkpoint-{tag}.pt',
        )

    def load(self, tag: str):
        data = torch.load(self.results_folder / f'checkpoint-{tag}.pt', map_location=self.device)
        self.model.load_state_dict(data['model'])
        self.optimizer.load_state_dict(data['opt'])

    # ── evaluation helpers ────────────────────────────────────────────────────

    def _collect_eval_real(self) -> np.ndarray:
        batches, collected = [], 0
        for batch in self.valid_loader:
            batch = batch[0]
            batches.append(batch)
            collected += batch.shape[0]
        return torch.cat(batches, dim=0).cpu().numpy()  # (N, T, C)

    def _eval_metrics(self, epoch: int, eval_real: np.ndarray):
        n_eval = len(eval_real)
        feat_dim = eval_real.shape[2]

        # generate fake samples  →  (N, T, C) numpy
        self.model.eval()
        with torch.no_grad():
            fake_np = self.model.get_prior_samples(n_eval)  # (N, T, C)
        self.model.train()

        # ── Save model checkpoint and generated samples ───────────────────────
        self.save(f'epoch{epoch}')
        samples_dir = self.results_folder / 'samples'
        samples_dir.mkdir(exist_ok=True)
        np.save(str(samples_dir / f'fake_epoch{epoch}.npy'), fake_np)

        # ── FID ───────────────────────────────────────────────────────────────
        real_t = torch.tensor(eval_real.astype(np.float32))
        fake_t = torch.tensor(fake_np.astype(np.float32))
        fid_val = Context_FID(real_t, fake_t)

        # ── Discriminative score ──────────────────────────────────────────────
        disc_val = discriminative_score_metrics(
            eval_real, fake_np,
            input_size=feat_dim,
            device=self.device,
        )

        print(f"Epoch {epoch}: FID={fid_val:.4f}  Disc={disc_val:.4f}")
        wandb.log({'eval/fid': fid_val, 'eval/discriminative': disc_val, 'epoch': epoch})

    # ── training loop ─────────────────────────────────────────────────────────

    def train(self):
        eval_real = self._collect_eval_real()
        best_val_loss = float('inf')

        for epoch in range(self.max_epochs):
            tic = time.time()

            # ── train ─────────────────────────────────────────────────────────
            self.model.train()
            train_loss = train_recon = train_kl = 0.0
            for batch in self.train_loader:
                X = batch[0].to(self.device)               # (B, T, C)
                z_mean, z_log_var, z = self.model.encoder(X)
                X_recon = self.model.decoder(z)

                loss, recon_loss, kl_loss = self.model.loss_function(X, X_recon, z_mean, z_log_var)
                loss = loss / X.size(0)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                train_loss  += loss.item()
                train_recon += (recon_loss / X.size(0)).item()
                train_kl    += (kl_loss    / X.size(0)).item()

            n = len(self.train_loader)
            train_loss  /= n
            train_recon /= n
            train_kl    /= n

            # ── validate ──────────────────────────────────────────────────────
            self.model.eval()
            val_loss = val_recon = val_kl = 0.0
            with torch.no_grad():
                for batch in self.valid_loader:
                    X = batch[0].to(self.device)
                    z_mean, z_log_var, z = self.model.encoder(X)
                    X_recon = self.model.decoder(z)
                    loss, recon_loss, kl_loss = self.model.loss_function(X, X_recon, z_mean, z_log_var)
                    loss = loss / X.size(0)
                    val_loss  += loss.item()
                    val_recon += (recon_loss / X.size(0)).item()
                    val_kl    += (kl_loss    / X.size(0)).item()

            m = len(self.valid_loader)
            val_loss  /= m
            val_recon /= m
            val_kl    /= m

            elapsed = time.time() - tic
            print(
                f"Epoch {epoch:4d} | "
                f"train loss={train_loss:.4f} (recon={train_recon:.4f} kl={train_kl:.4f}) | "
                f"val loss={val_loss:.4f} (recon={val_recon:.4f} kl={val_kl:.4f}) | "
                f"t={elapsed:.1f}s"
            )

            wandb.log({
                'train/loss': train_loss,
                'train/recon': train_recon,
                'train/kl': train_kl,
                'valid/loss': val_loss,
                'valid/recon': val_recon,
                'valid/kl': val_kl,
                'epoch': epoch,
            })

            # ── save best ─────────────────────────────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save('best')

            # ── periodic checkpoint ───────────────────────────────────────────
            if (epoch + 1) % self.save_every == 0:
                self.save(f'epoch{epoch+1}')

            # ── FID + discriminative score every eval_every epochs ────────────
            if epoch % self.eval_every == 0:
                self._eval_metrics(epoch, eval_real)