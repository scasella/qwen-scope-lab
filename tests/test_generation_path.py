import torch

from qwen_scope_lab.config import load_config
from qwen_scope_lab.generation import steer_generation
from qwen_scope_lab.model_loader import ModelBundle
from qwen_scope_lab.sae_loader import SAELayer


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, text, return_tensors="pt", **_kwargs):
        ids = [1, 2, 3] if text else [1]
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"tok{int(i)}" for i in ids)


class FakeInner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4, bias=False), torch.nn.Linear(4, 4, bias=False)])
        for layer in self.layers:
            torch.nn.init.eye_(layer.weight)


class FakeCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeInner()
        self.lm_head = torch.nn.Linear(4, 8, bias=False)
        with torch.no_grad():
            self.lm_head.weight.fill_(0.0)
            self.lm_head.weight[4, 0] = 1.0
            self.lm_head.weight[5, 1] = 1.0

    def _hidden(self, input_ids):
        base = torch.zeros(input_ids.shape[0], input_ids.shape[1], 4)
        base[..., 0] = input_ids.float()
        hidden = base
        for layer in self.model.layers:
            hidden = layer(hidden)
        return hidden

    def forward(self, input_ids, attention_mask=None):
        hidden = self._hidden(input_ids)
        return type("Output", (), {"logits": self.lm_head(hidden)})

    def generate(self, input_ids, attention_mask=None, max_new_tokens=1, **_kwargs):
        logits = self(input_ids, attention_mask=attention_mask).logits
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        return torch.cat([input_ids, next_id.repeat(1, max_new_tokens)], dim=1)


def test_steer_generation_returns_real_hook_deltas():
    config = load_config("configs/fake_test.yaml")
    bundle = ModelBundle(tokenizer=FakeTokenizer(), model=FakeCausalLM(), device=torch.device("cpu"), dtype=torch.float32)
    sae = SAELayer(
        layer=1,
        path=config.hf_cache_dir,
        W_enc=torch.zeros(config.d_sae, config.d_model),
        W_dec=torch.eye(config.d_model, config.d_sae),
        b_enc=torch.zeros(config.d_sae),
        b_dec=torch.zeros(config.d_model),
    )

    result = steer_generation(
        bundle=bundle,
        sae=sae,
        config=config,
        prompt="hello",
        layer=1,
        feature_id=1,
        strength=5.0,
        max_new_tokens=2,
        temperature=0.0,
        mode="all_positions",
    )

    assert result["hook_fired"] is True
    assert result["hidden_delta_norm"] > 0
    assert result["logits_delta_norm"] > 0
    assert result["unsteered_text"] != result["steered_text"]
