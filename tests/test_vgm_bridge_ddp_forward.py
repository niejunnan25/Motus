from contextlib import nullcontext
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.train_vgm_bridge_stage1 import VGMBridgeStage1Trainer


class _InnerModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, first_frame: torch.Tensor, **_: object) -> dict[str, torch.Tensor]:
        loss = self.weight * first_frame.float().mean()
        return {"total_loss": loss, "video_loss": loss, "middle_loss": loss}


class _ForwardWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module
        self.forward_calls = 0

    def forward(self, **kwargs: object) -> dict[str, torch.Tensor]:
        self.forward_calls += 1
        return self.module(**kwargs)


class _FakeAccelerator:
    sync_gradients = True

    @staticmethod
    def accumulate(_: torch.nn.Module):
        return nullcontext()

    @staticmethod
    def backward(loss: torch.Tensor) -> None:
        loss.backward()

    @staticmethod
    def clip_grad_norm_(parameters: object, max_norm: float) -> None:
        torch.nn.utils.clip_grad_norm_(parameters, max_norm)


def test_trainer_forward_uses_distributed_wrapper(tmp_path: Path) -> None:
    model = _ForwardWrapper(_InnerModel())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = VGMBridgeStage1Trainer(
        model=model,
        train_dataloader=[],
        optimizer=optimizer,
        scheduler=None,
        device=torch.device("cpu"),
        world_size=1,
        checkpoint_dir=str(tmp_path),
        accelerator=_FakeAccelerator(),
    )
    batch = {
        "first_frame": torch.ones(1, 1, 1, 1),
        "video_frames": torch.ones(1, 1, 1, 1, 1),
        "language_embedding": None,
    }

    trainer.train_step(batch)

    assert model.forward_calls == 1
