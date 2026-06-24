import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.components.tokenizer import ShallowSeekTokenizer
from src.components.encoder import DataEncoder
from src.components.model import ShallowSeek
from src.components.dataset import ShallowSeekDataset
from src.entities.config_entity import TrainingConfig
from src.entities.artifact_entity import TrainingArtifact
from src.configurations.paths_config import BEST_CHECKPOINT_PATH

class TrainingPipeline:
    def __init__(self, cfg: TrainingConfig = TrainingConfig()):
        self.cfg    = cfg
        self.device = torch.device(cfg.device)
        # phase 1 — tokenization (skipped if artifacts already exist)
        self._tokenize()

        # phase 2 — model + optimiser + dataloaders
        self.model = ShallowSeek(cfg.model).to(self.device)

        decay_params   = [p for n, p in self.model.named_parameters() if p.dim() >= 2]
        nodecay_params = [p for n, p in self.model.named_parameters() if p.dim() < 2]
        self.optimizer = torch.optim.AdamW(
            [{"params": decay_params, "weight_decay": cfg.weight_decay},
             {"params": nodecay_params, "weight_decay": 0.0}],
            lr=cfg.lr,
        )

        self.train_loader = DataLoader(ShallowSeekDataset(cfg.train_dataset), batch_size = cfg.batch_size, shuffle = True, drop_last  = True)
        self.val_loader = DataLoader(ShallowSeekDataset(cfg.val_dataset), batch_size = cfg.batch_size, shuffle = False, drop_last  = True)

        total_steps = len(self.train_loader) * cfg.max_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=cfg.lr / 10)
        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _tokenize(self):
        print("--- Phase 1: Tokenization ---")

        # Step 1: train BPE tokenizer (skips if tokenizer dir already exists)
        tok = ShallowSeekTokenizer(self.cfg.tokenizer)
        tok.train()
        tok_art = tok._load(self.cfg.tokenizer.tokenizer_path)
        print(f"Tokenizer vocab_size: {tok_art.vocab_size:,}")

        # Step 2: encode raw text -> binary token file (skips if encoded file exists)
        encoded_path = self.cfg.encoder.output_path
        if encoded_path.exists():
            print(f"Encoded file already exists: {encoded_path} — skipping encoding")
        else:
            encoder = DataEncoder(self.cfg.encoder)
            enc_art = encoder.encode()
            print(f"Encoded tokens: {enc_art.num_tokens:,}")
            print(f"Saved to: {enc_art.output_path}")
        print('\n')

    def _eval(self) -> float:
        self.model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                _, loss, _ = self.model(x, y)
                total += loss.item()
                count += 1
        self.model.train()
        return total / max(count, 1)

    def train(self) -> TrainingArtifact:
        print("--- Phase 2: Training ---")
        best_ckpt = BEST_CHECKPOINT_PATH
        if best_ckpt.exists():
            print(f"Checkpoint already exists: {best_ckpt} — skipping training")
            state = torch.load(best_ckpt, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state.get("model", state))
            return TrainingArtifact(
                final_train_loss = 0.0,
                final_val_loss   = 0.0,
                total_steps      = state.get("step", 0),
                checkpoint_path  = best_ckpt,
            )

        self.model.train()
        step        = 0
        train_loss  = 0.0
        val_loss    = 0.0
        best_val    = float("inf")

        for epoch in range(self.cfg.max_epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)

                _, loss, _ = self.model(x, y)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()
                self.scheduler.step()

                train_loss = loss.item()
                step += 1

                if step % self.cfg.eval_interval == 0:
                    val_loss = self._eval()
                    lr_now   = self.scheduler.get_last_lr()[0]
                    print(f"epoch {epoch+1}  step {step:>6}  lr {lr_now:.2e}  "
                          f"train_loss {train_loss:.4f}  val_loss {val_loss:.4f}")

                    if val_loss < best_val:
                        best_val = val_loss
                        torch.save({"model": self.model.state_dict(), "step": step}, best_ckpt)
                        print(f"  ↳ best val loss {best_val:.4f} — saved {best_ckpt.name}")

            val_loss = self._eval()
            print(f"=== epoch {epoch+1} done  step {step}  val_loss {val_loss:.4f} ===")

        final_ckpt = self.cfg.checkpoint_dir / f"checkpoint_{step}.pt"
        torch.save({"model": self.model.state_dict(), "step": step}, final_ckpt)
        print(f"Saved final checkpoint -> {final_ckpt}")

        return TrainingArtifact(final_train_loss = train_loss, final_val_loss = val_loss, total_steps = step, checkpoint_path  = best_ckpt)
