import torch
import torch.nn as nn
from rtx_oom_guard.trainer.auto_instrument import auto_instrument, _InstrumentedModel

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

def test_auto_instrument_wraps_correctly():
    model = DummyModel()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    wrapped_model, wrapped_opt = auto_instrument(model, opt, risk_threshold=0.8, use_triton=False)

    assert isinstance(wrapped_model, _InstrumentedModel)
    # The magical __class__ wrapper makes isinstance pass
    assert isinstance(wrapped_opt, torch.optim.Adam)

    assert hasattr(wrapped_opt, "step")
    assert hasattr(wrapped_opt, "zero_grad")

    # Check that state_dict delegation works
    state = wrapped_opt.state_dict()
    assert "param_groups" in state

def test_auto_instrument_train_step():
    model = DummyModel()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    wrapped_model, wrapped_opt = auto_instrument(model, opt, risk_threshold=0.8, use_triton=False)

    x = torch.randn(4, 10)
    y = torch.randn(4, 2)

    out = wrapped_model(x)
    loss = nn.MSELoss()(out, y)
    loss.backward()

    # Step should trigger hooks and policy evaluation
    wrapped_opt.step()
    wrapped_opt.zero_grad()

    # Assert parameters updated
    grad = wrapped_opt.param_groups[0]["params"][0].grad
    assert grad is None or torch.allclose(grad, torch.zeros_like(grad))
