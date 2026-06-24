from src.pipeline.training_pipeline import TrainingPipeline
from src.entities.config_entity import TrainingConfig

def run_training():
    cfg = TrainingConfig(max_epochs = 3, batch_size = 16, lr = 3e-4, grad_clip = 1.0, eval_interval = 100,device = "cuda")
    print("=== ShallowSeek Training ===")
    print(f"device: {cfg.device}")
    print(f"epochs: {cfg.max_epochs}")
    print(f"batch_size: {cfg.batch_size}")
    print(f"lr: {cfg.lr}")
    print(f"weight_decay: {cfg.weight_decay}")
    print(f"checkpoint_dir: {cfg.checkpoint_dir}\n")

    pipeline = TrainingPipeline(cfg)
    model_art = pipeline.model.get_artifact()
    print(f"vocab_size: {model_art.vocab_size:,}")
    print(f"d_model: {model_art.d_model}")
    print(f"n_layers: {model_art.n_layers}")
    print(f"unique params: {model_art.unique_params:,}\n")

    artifact = pipeline.train()

    print(f"\n=== Training Complete ===")
    print(f"total steps: {artifact.total_steps}")
    print(f"final train loss: {artifact.final_train_loss:.4f}")
    print(f"final val loss: {artifact.final_val_loss:.4f}")
    print(f"checkpoint saved: {artifact.checkpoint_path}")
    return artifact

if __name__ == "__main__":
    run_training()

