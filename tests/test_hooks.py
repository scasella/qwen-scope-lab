import torch

from qwen_scope_lab.hooks import HookTrace, register_capture_hook, register_steering_hook


class TinyInner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4, bias=False), torch.nn.Linear(4, 4, bias=False)])
        for layer in self.layers:
            torch.nn.init.eye_(layer.weight)

    def forward(self, hidden):
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyInner()

    def forward(self, hidden):
        return self.model(hidden)


def test_capture_hook_fires_and_removes():
    model = TinyModel()
    capture = {}
    handle = register_capture_hook(model, 0, capture)
    model(torch.ones(1, 2, 4))
    handle.remove()
    assert "residual" in capture
    capture.clear()
    model(torch.ones(1, 2, 4))
    assert capture == {}


def test_capture_hook_can_keep_tensor_on_active_device():
    model = TinyModel()
    capture = {}
    handle = register_capture_hook(model, 0, capture, to_cpu=False)
    x = torch.ones(1, 2, 4)
    model(x)
    handle.remove()
    assert capture["residual"].device == x.device


def test_steering_hook_no_accumulation():
    model = TinyModel()
    x = torch.zeros(1, 2, 4)
    vector = torch.tensor([1.0, 0.0, 0.0, 0.0])
    trace = HookTrace()
    handle = register_steering_hook(model, 0, vector, 2.0, "all_positions", trace)
    out = model(x)
    handle.remove()
    assert trace.fired_count == 1
    assert trace.hidden_delta_norm > 0
    assert torch.allclose(out[..., 0], torch.full((1, 2), 2.0))
    out2 = model(x)
    assert torch.allclose(out2, x)
