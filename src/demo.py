from src.pipeline.training_pipeline import TrainingPipeline
from src.entities.config_entity import TrainingConfig
from src.pipeline.inference_pipeline import InferencePipeline, InferencePipelineConfig, InferenceConfig
import torch

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

def run_inference():
    # use_chat_format=False — pretrained model, raw text completion only.
    # set use_chat_format=True after instruction fine-tuning (SFT).
    cfg = InferencePipelineConfig(
        inference = InferenceConfig(
            max_new_tokens = 30,
            temperature    = 0.8,
            top_k          = 50,
            device         = "cuda" if torch.cuda.is_available() else "cpu",
        ),
        use_chat_format = False,
    )

    print("\n=== ShallowSeek Inference (pretraining — text completion) ===")
    pipeline  = InferencePipeline(cfg)
    model_art = pipeline.engine.model.get_artifact()
    print(f"vocab_size    : {model_art.vocab_size:,}")
    print(f"d_model       : {model_art.d_model}")
    print(f"n_layers      : {model_art.n_layers}")
    print(f"unique params : {model_art.unique_params:,}\n")

    # raw text prompts — model continues the sentence, no chat wrapper
    prompts = [
        "The capital of France is",
        "Gravity is a force that",
        "Grandet",
    ]

    artifact = pipeline.run(prompts)
    print(f"Total prompts : {artifact.total_prompts}\n")

    vocab_size = pipeline.engine.model.cfg.vocab_size
    for prompt, r in zip(prompts, artifact.results):
        ids       = r.generated_ids
        low_ids   = [i for i in ids if i < 1000]
        high_ids  = [i for i in ids if i >= vocab_size - 1000]
        print(f"Prompt          : {prompt!r}")
        print(f"Prompt tokens   : {r.prompt_tokens}")
        print(f"Generated tokens: {r.generated_tokens}")
        print(f"Token IDs (first 10) : {ids[:10]}")
        print(f"Low IDs  (<1000)     : {len(low_ids)}  — likely special/common tokens")
        print(f"High IDs (>={vocab_size-1000}) : {len(high_ids)}  — rare/untrained tokens -> decode to ''")
        print(f"Completion      : {r.response!r}")
        print(f"Raw             : {r.raw_response!r}")
        print("-" * 50)

if __name__ == "__main__":
    run_training()
    run_inference()
