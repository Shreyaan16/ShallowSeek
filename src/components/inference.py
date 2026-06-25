import torch
from pathlib import Path
from transformers import PreTrainedTokenizerFast
from src.components.model import ShallowSeek
from src.components.system_prompt import SystemPromptCache
from src.entities.config_entity import ModelConfig, InferenceConfig
from src.entities.artifact_entity import InferenceArtifact
from src.constants import SOS_TOKEN, EOS_TOKEN, SYSTEM, USER, ASSISTANT, BLOCK_SIZE
from src.utils import encode, decode

_DEFAULT_SYSTEM = "You are ShallowSeek, a helpful AI assistant."
_SYS_PREFIX     = f"{SOS_TOKEN} {SYSTEM} {_DEFAULT_SYSTEM} {USER} "


class ShallowSeekInference:
    def __init__(self, cfg: InferenceConfig = InferenceConfig()):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.tokenizer = self._load_tokenizer()
        self.model     = self._load_model()
        vocab = self.tokenizer.get_vocab()
        self.eos_id     = vocab.get(EOS_TOKEN)
        self._sys_cache = None  # built lazily on first chat request

    def _load_tokenizer(self) -> PreTrainedTokenizerFast:
        tok = PreTrainedTokenizerFast.from_pretrained(str(self.cfg.tokenizer_path))
        tok.pad_token = "<PAD>"
        return tok

    def _load_model(self) -> ShallowSeek:
        model = ShallowSeek(ModelConfig()).to(self.device)
        if self.cfg.checkpoint_path.exists():
            state = torch.load(self.cfg.checkpoint_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state.get("model", state))
            print(f"Loaded checkpoint : {self.cfg.checkpoint_path}")
        else:
            print("WARNING: no checkpoint found — random weights, responses will be gibberish")
        model.eval()
        return model

    def _build_sys_cache(self) -> SystemPromptCache:
        sys_prefix = f"{SOS_TOKEN} {SYSTEM} {_DEFAULT_SYSTEM} {USER} "
        sys_ids    = encode(self.tokenizer, sys_prefix)
        return SystemPromptCache(self.model, sys_ids, BLOCK_SIZE, self.device)

    def build_prompt(self, user_msg: str, system: str = _DEFAULT_SYSTEM) -> str:
        return f"{SOS_TOKEN} {SYSTEM} {system} {USER} {user_msg} {ASSISTANT}"

    def generate(self, prompt: str) -> InferenceArtifact:
        prompt_ids = encode(self.tokenizer, prompt)

        # Split on the string boundary to avoid BPE boundary merges corrupting the token count.
        # Tokenizing the full prompt as one string can produce fewer tokens than the prefix alone
        # (BPE merges across the boundary), making prompt_tensor[:, sys_len:] empty or wrong.
        if prompt.startswith(_SYS_PREFIX):
            if self._sys_cache is None:
                self._sys_cache = self._build_sys_cache()
            user_ids  = encode(self.tokenizer, prompt[len(_SYS_PREFIX):])
            kv_caches = self._sys_cache.clone_caches()
        else:
            user_ids  = prompt_ids
            kv_caches = None

        user_tensor = torch.tensor([user_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            generated = self.model.generate(
                prompt_ids         = user_tensor,
                max_new_tokens     = self.cfg.max_new_tokens,
                temperature        = self.cfg.temperature,
                top_k              = self.cfg.top_k,
                eos_id             = self.eos_id,
                repetition_penalty = self.cfg.repetition_penalty,
                kv_caches          = kv_caches,
            )

        token_ids = generated[0].tolist()
        if self.eos_id is not None and self.eos_id in token_ids:
            token_ids = token_ids[: token_ids.index(self.eos_id)]

        # decode with special tokens visible so we can see what was generated
        response_raw  = decode(self.tokenizer, token_ids, skip_special_tokens=False)
        response_clean = decode(self.tokenizer, token_ids, skip_special_tokens=True)

        return InferenceArtifact(
            prompt           = prompt,
            prompt_tokens    = len(prompt_ids),
            generated_tokens = len(token_ids),
            generated_ids    = token_ids,
            response         = response_clean,
            raw_response     = response_raw,
            device           = str(self.device),
        )

    def get_artifact(self) -> InferenceArtifact:
        return self.generate(self.build_prompt("Hello!"))


if __name__ == "__main__":
    cfg = InferenceConfig(
        tokenizer_path  = Path("models/tokenizer"),
        checkpoint_path = Path("models/checkpoint.pt"),
        max_new_tokens  = 20,
        temperature     = 0.8,
        top_k           = 50,
        device          = "cuda" if torch.cuda.is_available() else "cpu",
    )

    engine = ShallowSeekInference(cfg)

    model_art = engine.model.get_artifact()
    print(f"vocab_size     : {model_art.vocab_size:,}")
    print(f"d_model        : {model_art.d_model}")
    print(f"n_layers       : {model_art.n_layers}")
    print(f"unique params  : {model_art.unique_params:,}")
    print(f"device         : {cfg.device}\n")

    prompts = [
        "What is the capital of France?",
        "Explain gravity in one sentence.",
    ]

    for user_msg in prompts:
        prompt   = engine.build_prompt(user_msg)
        artifact = engine.generate(prompt)
        print(f"User            : {user_msg}")
        print(f"Prompt tokens   : {artifact.prompt_tokens}")
        print(f"Generated tokens: {artifact.generated_tokens}")
        print(f"Response (clean): {artifact.response!r}")
        print(f"Response (raw)  : {artifact.raw_response!r}")




