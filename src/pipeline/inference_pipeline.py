import torch
from pathlib import Path
from src.components.inference import ShallowSeekInference
from src.entities.config_entity import InferenceConfig, InferencePipelineConfig
from src.entities.artifact_entity import InferenceArtifact, InferencePipelineArtifact
from src.configurations.paths_config import CHECKPOINTS_DIR

def latest_checkpoint(checkpoints_dir: Path) -> Path:
    # prefer best-val checkpoint saved during training
    best = checkpoints_dir / "checkpoint_best.pt"
    if best.exists():
        return best
    # fall back to highest-step checkpoint
    ckpts = sorted(checkpoints_dir.glob("checkpoint_*.pt"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    return ckpts[-1]


class InferencePipeline:
    def __init__(self, cfg: InferencePipelineConfig = InferencePipelineConfig()):
        self.cfg = cfg

        # auto-resolve checkpoint if not explicitly set
        ckpt_dir = Path(CHECKPOINTS_DIR)
        if not cfg.inference.checkpoint_path.exists() and ckpt_dir.exists():
            cfg.inference.checkpoint_path = latest_checkpoint(ckpt_dir)

        self.engine = ShallowSeekInference(cfg.inference)

    def run(self, prompts: list[str] = None) -> InferencePipelineArtifact:
        prompts = prompts or self.cfg.prompts
        results: list[InferenceArtifact] = []

        for prompt in prompts:
            # chat format: wrap in |SYSTEM|/|USER|/|ASSISTANT| template (SFT model)
            # raw format:  feed text directly as a completion prompt (pretrained model)
            if self.cfg.use_chat_format:
                prompt = self.engine.build_prompt(prompt)
            artifact = self.engine.generate(prompt)
            results.append(artifact)

        return InferencePipelineArtifact(
            total_prompts = len(results),
            results       = results,
        )

    def get_artifact(self) -> InferencePipelineArtifact:
        return self.run(["The meaning of life is"])








