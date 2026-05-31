"""GPU-free development backend.

Builds a real :class:`SteeringService` whose model bundle and SAE loader are tiny
in-memory CPU stand-ins. Every inspect / compare / steer call runs the *real*
activation-capture, contrast, and residual-steering code paths -- only the weights
are fake. This lets the web UI run end-to-end locally with no GPU and no downloads.

Switch to the real Qwen path by serving with ``--config configs/...`` instead of
``--dev`` (see ``serve_web.py``); nothing else changes.
"""
from __future__ import annotations

import re
from typing import Any

import torch
import torch.nn as nn

from .config import SteeringConfig
from .model_loader import ModelBundle
from .sae_loader import SAELayer
from .service import SteeringService

_SEED = 7
_D_MODEL = 64
_D_SAE = 256
_NUM_LAYERS = 6
_DEFAULT_LAYER = 3
_MAX_VOCAB = 4096

# A small, readable vocabulary with a few thematic clusters so steering produces
# legible, themed shifts in generation. Index in this list == token id.
BASE_WORDS: list[str] = [
    "<s>", ".", ",", "!", "?", ":", ";", "{", "}", "\"",
    "the", "a", "an", "of", "and", "to", "in", "is", "are", "was",
    "it", "that", "this", "for", "with", "as", "on", "by", "at", "from",
    "Paris", "France", "French", "capital", "city", "Europe", "European", "river", "Seine", "country",
    "art", "museum", "cafe", "bread", "light", "street", "rooftops", "golden", "evening", "morning",
    "yes", "no", "simply", "just", "direct", "clear", "brief", "short", "exactly", "answer",
    "however", "moreover", "furthermore", "perhaps", "gently", "wandered", "storied", "resplendent", "imagination", "centuries",
    "name", "age", "value", "key", "object", "list", "number", "true", "false", "null",
    "explain", "describe", "write", "return", "sparse", "autoencoder", "feature", "model", "vector", "residual",
    "one", "sentence", "about", "long", "rambling", "story", "concise", "factual", "please", "make",
    "good", "great", "well", "made", "price", "worth", "very", "really", "quite", "rather",
    "he", "she", "they", "we", "you", "I", "people", "world", "time", "place",
    "known", "renowned", "famous", "beautiful", "major", "cultural", "political", "center", "history", "across",
    "its", "their", "our", "many", "some", "few", "more", "most", "than", "over",
    "be", "have", "has", "had", "do", "does", "did", "can", "will", "would",
    "city.", "Paris.", "France.", "here", "there", "now", "then", "so", "but", "or",
]
_WORD_TO_ID = {w: i for i, w in enumerate(BASE_WORDS)}
_VOCAB = len(BASE_WORDS)

# feature id -> theme words. These features get encoder/decoder directions aligned
# with the mean embedding of their cluster, so they (a) activate on those tokens
# during inspect and (b) bias generation toward those tokens when steered.
_THEMES: dict[int, list[str]] = {
    42: ["Paris", "France", "capital", "city", "Europe", "river", "Seine"],
    77: [".", "yes", "no", "direct", "brief", "short", "concise", "exactly"],
    101: ["however", "moreover", "furthermore", "perhaps", "gently", "storied", "resplendent"],
    150: ["{", "}", "name", "age", "value", "key", "object", "true", "false"],
    198: ["art", "museum", "cafe", "bread", "light", "golden", "rooftops"],
}


def build_dev_config() -> SteeringConfig:
    return SteeringConfig(
        model_id="dev/qwen-scope-mini",
        sae_id="dev/sae-mini",
        top_k=12,
        num_layers=_NUM_LAYERS,
        d_model=_D_MODEL,
        d_sae=_D_SAE,
        default_layer=_DEFAULT_LAYER,
        default_max_new_tokens=24,
        torch_dtype="float32",
        device="cpu",
        sae_cache_max_layers=_NUM_LAYERS,
        hf_cache_dir="/tmp/qwen-scope-dev-cache",
        trust_remote_code=False,
        device_map=None,
        low_cpu_mem_usage=False,
        local_files_only=True,
        notebook_path="/tmp/qwen-scope-dev-notebook.json",
    )


class _DevTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __init__(self) -> None:
        self.id2word: dict[int, str] = dict(enumerate(BASE_WORDS))
        self._next_id = _VOCAB

    def _word_id(self, word: str) -> int:
        if word in _WORD_TO_ID:
            return _WORD_TO_ID[word]
        # out-of-vocabulary word: assign a stable id (embedded but never generated)
        for wid, w in self.id2word.items():
            if w == word:
                return wid
        wid = self._next_id if self._next_id < _MAX_VOCAB else (hash(word) % (_MAX_VOCAB - _VOCAB) + _VOCAB)
        self._next_id = min(self._next_id + 1, _MAX_VOCAB - 1)
        self.id2word[wid] = word
        return wid

    def __call__(self, text: str, return_tensors: str = "pt", truncation: bool = False, max_length: int | None = None, **_kwargs):
        pieces = re.findall(r"\w+|[^\w\s]", text or "")
        ids = [self._word_id(p) for p in pieces] or [self._word_id("<s>")]
        if truncation and max_length:
            ids = ids[:max_length]
        tensor = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": tensor, "attention_mask": torch.ones_like(tensor)}

    def encode(self, text: str, add_special_tokens: bool = True, **_kwargs) -> list[int]:
        # flat id list, matching HF tokenizer.encode + the mlx-lm TokenizerWrapper.encode
        pieces = re.findall(r"\w+|[^\w\s]", text or "")
        return [self._word_id(p) for p in pieces] or [self._word_id("<s>")]

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self.id2word.get(int(i), f"<{int(i)}>") for i in ids]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        words = []
        for i in ids:
            w = self.id2word.get(int(i), "")
            if skip_special_tokens and w == "<s>":
                continue
            words.append(w)
        text = ""
        for w in words:
            if w in {".", ",", "!", "?", ":", ";", "}", "\""} and text:
                text += w
            elif text:
                text += " " + w
            else:
                text = w
        return text


class _Inner(nn.Module):
    def __init__(self, num_layers: int) -> None:
        super().__init__()
        # identity blocks: residual == embedding, but each is a real module so the
        # capture / steering forward-hooks fire exactly as they do on a real model.
        self.layers = nn.ModuleList([nn.Identity() for _ in range(num_layers)])


class _DevModel(nn.Module):
    def __init__(self, embedding: torch.Tensor, lm_head: torch.Tensor) -> None:
        super().__init__()
        self.model = _Inner(_NUM_LAYERS)
        self.embedding = nn.Embedding(_MAX_VOCAB, _D_MODEL)
        with torch.no_grad():
            self.embedding.weight.copy_(embedding)
        self.lm_head = nn.Linear(_D_MODEL, _VOCAB, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(lm_head)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        hidden = self.embedding(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        logits = self.lm_head(hidden)
        return type("Output", (), {"logits": logits})

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                 max_new_tokens: int = 8, **_kwargs) -> torch.Tensor:
        ids = input_ids
        for _ in range(int(max_new_tokens)):
            logits = self(ids).logits[0, -1, :_VOCAB]
            order = torch.argsort(logits, descending=True).tolist()
            recent = set(ids[0, -3:].tolist())
            nxt = next((t for t in order if t not in recent and t >= 10), order[0])
            ids = torch.cat([ids, torch.tensor([[nxt]], dtype=ids.dtype)], dim=1)
        return ids


class _DevSAELoader:
    def __init__(self, layers: dict[int, SAELayer]) -> None:
        self._layers = layers
        self._used: list[int] = []

    @property
    def cached_layers(self) -> list[int]:
        return list(self._used)

    def load_layer(self, layer: int) -> SAELayer:
        if layer not in self._layers:
            raise ValueError(f"dev backend has no SAE for layer {layer}; valid: {sorted(self._layers)}")
        if layer not in self._used:
            self._used.append(layer)
        return self._layers[layer]


def _build_weights() -> tuple[torch.Tensor, torch.Tensor, dict[int, SAELayer]]:
    g = torch.Generator().manual_seed(_SEED)
    embedding = torch.randn(_MAX_VOCAB, _D_MODEL, generator=g) * 0.4
    lm_head = embedding[:_VOCAB].clone()  # tie -> greedy picks nearest-embedding word

    w_enc = torch.randn(_D_SAE, _D_MODEL, generator=g) * 0.5
    w_enc = w_enc / w_enc.norm(dim=1, keepdim=True).clamp_min(1e-6)
    w_dec = torch.randn(_D_MODEL, _D_SAE, generator=g) * 0.5
    w_dec = w_dec / w_dec.norm(dim=0, keepdim=True).clamp_min(1e-6)

    for fid, words in _THEMES.items():
        wids = [_WORD_TO_ID[w] for w in words if w in _WORD_TO_ID]
        theme = embedding[wids].mean(dim=0)
        theme = theme / theme.norm().clamp_min(1e-6)
        w_enc[fid] = theme * 3.0
        w_dec[:, fid] = theme * 3.0

    # Themed feature *families*: bands of features whose decoder directions are a
    # shared theme direction + noise, so a 2D projection of W_dec has legible
    # cluster structure for the latent-space map to reveal. Real SAEs get this
    # structure from learned geometry; here it is seeded, exactly like the singletons
    # above (which exist so steering produces themed text). Encoder scale is kept
    # below the singletons' so feature 42 et al. still dominate inspect/steer.
    _FAMILY_BASE = 200
    _FAMILY_SIZE = 8
    for ti, words in enumerate(_THEMES.values()):
        wids = [_WORD_TO_ID[w] for w in words if w in _WORD_TO_ID]
        theme = embedding[wids].mean(dim=0)
        theme = theme / theme.norm().clamp_min(1e-6)
        for k in range(_FAMILY_SIZE):
            mid = _FAMILY_BASE + ti * _FAMILY_SIZE + k
            if mid >= _D_SAE:
                continue
            noisy = theme + torch.randn(_D_MODEL, generator=g) * 0.10
            noisy = noisy / noisy.norm().clamp_min(1e-6)
            w_enc[mid] = noisy * 2.0
            w_dec[:, mid] = noisy

    b_enc = torch.zeros(_D_SAE)
    b_dec = torch.zeros(_D_MODEL)
    layers = {
        layer: SAELayer(layer=layer, path=f"dev://sae/layer{layer}",
                        W_enc=w_enc.clone(), W_dec=w_dec.clone(), b_enc=b_enc.clone(), b_dec=b_dec.clone())
        for layer in range(_NUM_LAYERS)
    }
    return embedding, lm_head, layers


def build_dev_service() -> SteeringService:
    config = build_dev_config()
    embedding, lm_head, layers = _build_weights()
    model = _DevModel(embedding, lm_head).eval()
    bundle = ModelBundle(tokenizer=_DevTokenizer(), model=model, device=torch.device("cpu"), dtype=torch.float32)
    service = SteeringService(config, "dev://config")
    service.bundle = bundle
    service.sae_loader = _DevSAELoader(layers)  # type: ignore[assignment]
    return service


def themed_feature_ids() -> dict[int, list[str]]:
    """Feature ids with intentional behavior, for demos and tests."""
    return {fid: list(words) for fid, words in _THEMES.items()}
