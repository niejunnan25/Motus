from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from torch.utils.data import RandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import train_vgm_bridge_stage1
from train.train_vgm_bridge_stage1 import VGMBridgeStage1Trainer, create_train_dataloader


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
        config=SimpleNamespace(training=SimpleNamespace(grad_clip_norm=1.0)),
    )
    batch = {
        "first_frame": torch.ones(1, 1, 1, 1),
        "video_frames": torch.ones(1, 1, 1, 1, 1),
        "language_embedding": None,
    }

    trainer.train_step(batch)

    assert model.forward_calls == 1


class _Dataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 16

    def __getitem__(self, index: int) -> int:
        return index


def test_dataloader_leaves_distributed_sharding_to_accelerate(monkeypatch: object) -> None:
    monkeypatch.setattr(train_vgm_bridge_stage1, "VideoBridgeDataset", lambda **_: _Dataset())
    config = OmegaConf.create(
        {
            "dataset": {
                "type": "video_bridge",
                "dataset_dir": ["unused"],
            },
            "common": {
                "global_downsample_rate": 1,
                "num_video_frames": 52,
                "video_height": 224,
                "video_width": 448,
                "state_condition_mode": "none",
            },
            "training": {"batch_size": 2},
            "system": {"num_workers": 0, "pin_memory": False},
        }
    )

    dataloader = create_train_dataloader(config, rank=0, world_size=2)

    assert isinstance(dataloader.sampler, RandomSampler)
