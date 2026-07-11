# Motus Policy variant for RoboTwin visualization.
#
# This file keeps the original policy behavior, but changes the visual logging
# strategy so every generated future frame is saved. For the RoboTwin checkpoint
# this is normally 8 predicted frames per inference call.

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw

try:
    from .deploy_policy import MotusPolicy as BaseMotusPolicy
    from .deploy_policy import encode_obs
except ImportError:
    from deploy_policy import MotusPolicy as BaseMotusPolicy
    from deploy_policy import encode_obs


logger = logging.getLogger(__name__)


class MotusSave8Policy(BaseMotusPolicy):
    """
    RoboTwin Motus policy with full future-frame visualization.

    Outputs per inference step:
      images/<task>/episode_XXXX_step_XXXX_future8_sheet.png
      images/<task>/episode_XXXX_step_XXXX_frames/condition.png
      images/<task>/episode_XXXX_step_XXXX_frames/pred_01.png ... pred_08.png
      images/<task>/episode_XXXX_step_XXXX_frames/metadata.json
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        wan_path: str,
        vlm_path: str,
        device: str = "cuda",
        log_dir: Optional[str] = None,
        task_name: Optional[str] = None,
    ):
        super().__init__(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            wan_path=wan_path,
            vlm_path=vlm_path,
            device=device,
            log_dir=log_dir,
            task_name=task_name,
        )
        self.max_saved_frames = int(
            os.environ.get(
                "MOTUS_SAVE_NUM_FUTURE_FRAMES",
                self.config_dict["common"].get("num_video_frames", 8),
            )
        )
        self.save_individual_frames = os.environ.get("MOTUS_SAVE_INDIVIDUAL_FRAMES", "1") != "0"
        logger.info(
            "MotusSave8Policy enabled: max_saved_frames=%s, save_individual_frames=%s",
            self.max_saved_frames,
            self.save_individual_frames,
        )

    @staticmethod
    def _tensor_to_uint8_hwc(tensor: torch.Tensor) -> np.ndarray:
        """Convert [C,H,W] or [H,W,C] tensor in [0,1] to uint8 HWC."""
        if tensor.dim() != 3:
            raise ValueError(f"Expected 3D image tensor, got shape {tuple(tensor.shape)}")
        if tensor.shape[0] in (1, 3):
            tensor = tensor.permute(1, 2, 0)
        tensor = tensor.detach().cpu().float().clamp(0, 1)
        array = (tensor.numpy() * 255.0).round().astype(np.uint8)
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        return array

    @staticmethod
    def _normalize_predicted_frames(predicted_frames: torch.Tensor) -> torch.Tensor:
        """Return predicted frames as [T,C,H,W]."""
        if predicted_frames.dim() != 4:
            raise ValueError(f"Expected 4D predicted frames, got shape {tuple(predicted_frames.shape)}")
        if predicted_frames.shape[1] in (1, 3):
            return predicted_frames
        if predicted_frames.shape[0] in (1, 3):
            return predicted_frames.permute(1, 0, 2, 3)
        return predicted_frames

    @staticmethod
    def _with_label(image: Image.Image, label: str) -> Image.Image:
        label_height = 24
        canvas = Image.new("RGB", (image.width, image.height + label_height), (0, 0, 0))
        canvas.paste(image, (0, label_height))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 5), label, fill=(255, 255, 255))
        return canvas

    def _create_frame_grid(self, condition_frame: torch.Tensor, predicted_frames: torch.Tensor) -> Image.Image:
        """Create a single-row sheet: condition + all saved future frames."""
        predicted_frames = self._normalize_predicted_frames(predicted_frames)
        num_frames = min(predicted_frames.shape[0], self.max_saved_frames)

        condition_img = Image.fromarray(self._tensor_to_uint8_hwc(condition_frame)).convert("RGB")
        panels = [self._with_label(condition_img, "condition")]

        for idx in range(num_frames):
            frame_img = Image.fromarray(self._tensor_to_uint8_hwc(predicted_frames[idx])).convert("RGB")
            panels.append(self._with_label(frame_img, f"pred_{idx + 1:02d}"))

        sheet_width = sum(panel.width for panel in panels)
        sheet_height = max(panel.height for panel in panels)
        sheet = Image.new("RGB", (sheet_width, sheet_height), (0, 0, 0))

        x = 0
        for panel in panels:
            sheet.paste(panel, (x, 0))
            x += panel.width

        return sheet

    def _save_individual_prediction_frames(
        self,
        frame_dir: Path,
        condition_frame: torch.Tensor,
        predicted_frames: torch.Tensor,
    ) -> Dict[str, Any]:
        predicted_frames = self._normalize_predicted_frames(predicted_frames)
        num_available = int(predicted_frames.shape[0])
        num_saved = min(num_available, self.max_saved_frames)

        frame_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self._tensor_to_uint8_hwc(condition_frame)).save(frame_dir / "condition.png")

        saved_files: List[str] = []
        for idx in range(num_saved):
            file_name = f"pred_{idx + 1:02d}.png"
            Image.fromarray(self._tensor_to_uint8_hwc(predicted_frames[idx])).save(frame_dir / file_name)
            saved_files.append(file_name)

        metadata: Dict[str, Any] = {
            "episode": int(self.episode_count),
            "step": int(self.step_count),
            "num_available_predicted_frames": num_available,
            "num_saved_predicted_frames": num_saved,
            "files": saved_files,
        }
        with open(frame_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return metadata

    def _save_frame_grid(self, condition_frame: torch.Tensor, predicted_frames: torch.Tensor):
        """Save all generated future frames plus a compact inspection sheet."""
        if not self.save_images:
            return

        try:
            prefix = f"episode_{self.episode_count:04d}_step_{self.step_count:04d}"
            sheet = self._create_frame_grid(condition_frame, predicted_frames)
            sheet_path = self.save_dir / f"{prefix}_future8_sheet.png"
            sheet.save(sheet_path)

            frame_dir = self.save_dir / f"{prefix}_frames"
            metadata = None
            if self.save_individual_frames:
                metadata = self._save_individual_prediction_frames(
                    frame_dir=frame_dir,
                    condition_frame=condition_frame,
                    predicted_frames=predicted_frames,
                )

            logger.info(
                "Saved full future-frame visualization: sheet=%s, frame_dir=%s, metadata=%s",
                sheet_path,
                frame_dir if self.save_individual_frames else None,
                metadata,
            )
        except Exception as exc:
            logger.warning("Failed to save full future-frame visualization: %s", exc)


def get_model(usr_args):
    """
    Initialize the Motus policy variant that saves all future frames.

    Args:
        usr_args: Arguments from eval script. Must include ckpt_setting,
            wan_path, and vlm_path.
    """
    checkpoint_path = usr_args.get("ckpt_setting")
    wan_path = usr_args.get("wan_path")
    vlm_path = usr_args.get("vlm_path")

    if not wan_path:
        raise ValueError("wan_path not provided in usr_args")
    if not vlm_path:
        raise ValueError("vlm_path not provided in usr_args")

    policy_dir = Path(__file__).parent
    config_path = policy_dir / "utils" / "robotwin.yml"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return MotusSave8Policy(
        checkpoint_path=checkpoint_path,
        wan_path=wan_path,
        vlm_path=vlm_path,
        config_path=str(config_path),
        device=device,
        log_dir=usr_args.get("log_dir"),
        task_name=usr_args.get("task_name"),
    )


def eval(TASK_ENV, model, observation):
    """Evaluation function compatible with RoboTwin policy loading."""
    obs = encode_obs(observation)

    instruction = TASK_ENV.get_instruction()
    model.set_instruction(instruction)
    model.update_obs(obs)

    actions = model.get_action()

    for action in actions:
        TASK_ENV.take_action(action, action_type="qpos")


def reset_model(model):
    """Reset model cache at episode start."""
    model.obs_cache.clear()
    model.action_cache.clear()
    model.current_state = None
    model.is_first_step = True
    model.prev_action = None
    model.episode_count += 1
    model.step_count = 0
    logger.info("Model reset completed for episode %s", model.episode_count)
