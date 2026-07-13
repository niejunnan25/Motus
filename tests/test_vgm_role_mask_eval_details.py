from pathlib import Path
import sys

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_vgm_role_mask_eval_details import extract_panel, read_sheet_rows, windows_match


def test_windows_match_checks_caption_and_embedding_source() -> None:
    reference = [
        {
            "episode_name": "episode_000001",
            "condition_idx": 0,
            "frame_indices": [0, 2, 4],
            "total_frames": 5,
            "task_index": 1,
            "task_text": "move object",
            "language_caption": "Move the object to the target.",
            "language_caption_version": "detailed_v1",
            "language_embedding_path": "/embeddings/modified/task_000001.pt",
        }
    ]
    candidate = [dict(reference[0])]

    assert windows_match(reference, candidate)

    candidate[0]["language_embedding_path"] = "/embeddings/original/task_000001.pt"
    assert not windows_match(reference, candidate)


def test_sheet_reader_and_panel_extraction_support_manifest_frame_count(tmp_path: Path) -> None:
    frame_count = 3
    frame_width = 4
    label_height = 2
    sheet = Image.new("RGB", (frame_count * frame_width, label_height + 2 * frame_width), "white")
    for frame_idx in range(frame_count):
        x = frame_idx * frame_width
        sheet.paste(Image.new("RGB", (2, 4), (10 + frame_idx, 0, 0)), (x, label_height))
        sheet.paste(Image.new("RGB", (2, 4), (0, 10 + frame_idx, 0)), (x + 2, label_height))
    path = tmp_path / "sheet.png"
    sheet.save(path)

    gt, pred, detected_width = read_sheet_rows(path, frame_count)
    rgb, panel_width = extract_panel(gt, detected_width, frame_count, "rgb")
    mask, _ = extract_panel(gt, detected_width, frame_count, "mask")

    assert pred.size == (frame_count * frame_width, frame_width)
    assert detected_width == frame_width
    assert panel_width == 2
    assert rgb.size == (frame_count * 2, frame_width)
    assert mask.size == (frame_count * 2, frame_width)
    assert rgb.getpixel((0, 0)) == (10, 0, 0)
    assert mask.getpixel((0, 0)) == (0, 10, 0)
