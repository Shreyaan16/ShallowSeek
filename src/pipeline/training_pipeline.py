import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.components.model import ShallowSeek
from src.components.dataset import ShallowSeekDataset
from src.entities.config_entity import TrainingConfig
from src.entities.artifact_entity import TrainingArtifact

class TrainingPipeline:
    def __init__(self, cfg: TrainingConfig = TrainingConfig()):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = ShallowSeek(cfg.model).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)

        self.train_loader = DataLoader(
            ShallowSeekDataset(cfg.train_dataset),
            batch_size = cfg.batch_size,
            shuffle    = True,
            drop_last  = True,
        )
        self.val_loader = DataLoader(
            ShallowSeekDataset(cfg.val_dataset),
            batch_size = cfg.batch_size,
            shuffle    = False,
            drop_last  = True,
        )

        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
        self.model.train()
        step = 0
        train_loss = 0.0
        val_loss   = 0.0

        for epoch in range(self.cfg.max_epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)

                _, loss, _ = self.model(x, y)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()

                train_loss = loss.item()
                step += 1

                if step % self.cfg.eval_interval == 0:
                    val_loss = self._eval()
                    print(f"epoch {epoch+1}  step {step:>6}  "
                          f"train_loss {train_loss:.4f}  val_loss {val_loss:.4f}")

            # Eval at end of every epoch too
            val_loss = self._eval()
            print(f"=== epoch {epoch+1} done  step {step}  val_loss {val_loss:.4f} ===")

        checkpoint_path = self.cfg.checkpoint_dir / f"checkpoint_{step}.pt"
        torch.save({"model": self.model.state_dict(), "step": step}, checkpoint_path)
        print(f"Saved checkpoint → {checkpoint_path}")

        return TrainingArtifact(
            final_train_loss = train_loss,
            final_val_loss   = val_loss,
            total_steps      = step,
            checkpoint_path  = checkpoint_path,
        )













