import copy

import torch

from kirl.engine.optim import CPUOffloadAdamW


def test_adamw_update_and_resume():
    p = torch.nn.Parameter(torch.tensor([0.125, -0.75]))
    ref = torch.nn.Parameter(p.detach().clone())
    opt = CPUOffloadAdamW([p])
    baseline = torch.optim.AdamW([ref], foreach=False)
    for i in range(4):
        p.grad = ref.grad = torch.tensor([0.3, -0.2]) * (i + 1)
        opt.step()
        baseline.step()
        torch.testing.assert_close(p, ref)
        if i == 1:
            saved = copy.deepcopy(opt.state_dict())
            opt = CPUOffloadAdamW([p])
            opt.load_state_dict(saved)
    assert opt.state[p]["master"].device.type == "cpu"


def test_swappable_algorithms():
    from kirl.engine.optim import HybridDeviceOptimizer
    for cls in (torch.optim.Adam, torch.optim.AdamW, torch.optim.Muon):
        p = torch.nn.Parameter(torch.eye(4))
        ref = torch.nn.Parameter(p.detach().clone())
        kwargs = dict(lr=0.01, weight_decay=0.02)
        opt = HybridDeviceOptimizer([p], cls, **kwargs)
        baseline = cls([ref], **kwargs)
        for _ in range(3):
            p.grad = ref.grad = torch.ones_like(p)
            opt.step()
            baseline.step()
            torch.testing.assert_close(p, ref)
