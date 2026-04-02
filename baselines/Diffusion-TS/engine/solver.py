import os
import sys
import time
import torch
import numpy as np
import torch.nn.functional as F

from pathlib import Path
from tqdm.auto import tqdm
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from Utils.io_utils import instantiate_from_config, get_model_parameters_info
import copy
import wandb
from evaluation_metrics.discriminative_torch import discriminative_score_metrics
from evaluation_metrics.ts2vec.context_fid import Context_FID

class Trainer(object):
    def __init__(self, config, args, model, dataloader):
        super().__init__()
        self.model = model
        self.device = self.model.betas.device
        self.train_num_epochs = config['solver']['max_epochs']
        self.train_dataloader = dataloader['train_dataloader']
        self.valid_dataloader = dataloader['valid_dataloader']
        self.step = 0
        self.milestone = 0
        self.args, self.config = args, config

        self.results_folder = Path(config['solver']['results_folder'])
        os.makedirs(self.results_folder, exist_ok=True)

        start_lr = config['solver'].get('base_lr', 1.0e-4)
        self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=start_lr)

        self.ema_decay = 0.999
        self.ema_model = copy.deepcopy(self.model).to(self.device)
        self.ema_model.eval()

        wandb.init(
            project=os.getenv("WANDB_PROJECT", "no_name"),
            name=os.getenv("WANDB_NAME", "no_name"),
            config=self.config
        )

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'opt': self.opt.state_dict(),
        }
        torch.save(data, str(self.results_folder / f'checkpoint-{milestone}.pt'))

    def update_ema(self):
        with torch.no_grad():
            for ema_p, p in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_p.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)

    def load(self, milestone):
        device = self.device
        data = torch.load(str(self.results_folder / f'checkpoint-{milestone}.pt'), map_location=device)
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema_model.load_state_dict(data['ema'])
        self.milestone = milestone

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _collect_eval_real(self) -> np.ndarray:
        """Collect up to max_samples windows from the validation set."""
        batches = []
        collected = 0
        for batch in self.valid_dataloader:
            batches.append(batch)
            collected += batch.shape[0]
        return torch.cat(batches, dim=0).numpy()  # (N, T, C)

    def _eval_metrics(self, epoch: int, real_np: np.ndarray):
        """Generate samples, save checkpoint + fake array, log FID + discriminative score."""
        window, var_num = real_np.shape[1], real_np.shape[2]
        n_eval = len(real_np)

        # generate fake samples  →  (n_eval, T, C) numpy
        fake_np = self.sample(n_eval, size_every=128, shape=[window, var_num])[:n_eval]

        # ── Save model checkpoint and generated samples ───────────────────────
        self.save(f'epoch{epoch}')
        samples_dir = self.results_folder / 'samples'
        samples_dir.mkdir(exist_ok=True)
        np.save(str(samples_dir / f'fake_epoch{epoch}.npy'), fake_np)

        # ── FID ──────────────────────────────────────────────────────────────
        real_t = torch.tensor(real_np.astype(np.float32))  # (N, T, C)
        fake_t = torch.tensor(fake_np.astype(np.float32))  # (N, T, C)
        # fid_val = compute_fid(real_t, fake_t, ckpt_path=self.fid_vae_ckpt)['fid']
        fid_val = Context_FID(real_t, fake_t)

        # ── Discriminative score ─────────────────────────────────────────────
        disc_val = discriminative_score_metrics(
            real_np, fake_np,
            input_size=var_num,
            device=self.device,
        )

        print(f"Epoch {epoch}: FID={fid_val:.4f}  Disc={disc_val:.4f}")
        wandb.log({
            'eval/fid': fid_val,
            'eval/discriminative': disc_val,
            'epoch': epoch,
        })

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self):
        # Collect fixed real validation windows once for evaluation
        eval_real = self._collect_eval_real()
        seq_len, feature_size = eval_real.shape[1], eval_real.shape[2]

        step = 0
        best_val_loss = float('inf')
        for epoch in range(step, self.train_num_epochs):
            train_loss_avg = 0.0
            tic = time.time()
            self.model.train()
            for batch in self.train_dataloader:
                batch = batch.to(self.device)
                loss = self.model(batch, target=batch)
                train_loss_avg += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.opt.zero_grad()
                self.step += 1
                step += 1
                self.update_ema()
            train_loss_avg = train_loss_avg / len(self.train_dataloader)
            toc = time.time()
            print(f"Epoch {epoch}: Train Loss: {train_loss_avg:.6f}, Time: {toc - tic:.2f}s")
            wandb.log({
                "train/epoch_loss": train_loss_avg,
                "train/learning_rate": self.opt.param_groups[0]['lr'],
                "epoch": epoch
            })

            if epoch % 100 == 0:
                # ── Validation loss ───────────────────────────────────────────
                val_loss_avg = 0.0
                self.ema_model.eval()
                for batch in self.valid_dataloader:
                    batch = batch.to(self.device)
                    with torch.no_grad():
                        loss = self.ema_model(batch, target=batch)
                    val_loss_avg += loss.item()
                val_loss_avg = val_loss_avg / len(self.valid_dataloader)

                print(
                    f"Epoch {epoch}: Train Loss: {train_loss_avg:.6f}, "
                    f"Val Loss: {val_loss_avg:.6f}, Time: {toc - tic:.2f}s"
                )
                wandb.log({
                    "valid/loss": val_loss_avg,
                    "epoch": epoch
                })
                if val_loss_avg < best_val_loss:
                    best_val_loss = val_loss_avg
                    self.save("best")

                # ── FID + Discriminative score ────────────────────────────────
                self._eval_metrics(epoch, eval_real)

    def sample(self, num, size_every, shape=None, model_kwargs=None, cond_fn=None):
        samples = np.empty([0, shape[0], shape[1]])
        num_cycle = int(num // size_every) + 1
        tic = time.time()
        for _ in range(num_cycle):
            sample = self.ema_model.generate_mts(batch_size=size_every, model_kwargs=model_kwargs, cond_fn=cond_fn)
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            torch.cuda.empty_cache()
        print('Sampling done, time: {:.2f}'.format(time.time() - tic))
        return samples