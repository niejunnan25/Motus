#!/usr/bin/env python3
"""Create a lightweight human review pack for MaskWAM-style mask annotations."""

from __future__ import annotations

import argparse
import shutil
import csv
import html
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from maskwam_view_schema import (
    BASE_TRAIN_ROLES,
    FINAL_ROLE,
    VIEW_NAMES,
    VIEW_TRAIN_ROLES,
    frame_view_slices,
    role_color,
    view_role,
)

ROLE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "object": (255, 60, 60),
    "target": (45, 175, 255),
    "robot": (185, 100, 255),
    "final_train_mask": (235, 40, 40),
    **{role: role_color(role) for role in VIEW_TRAIN_ROLES},
}

REVIEW_STATUSES = ["ok", "wrong_object", "missing_object", "missing_target", "too_broad", "bad_tracking", "exclude"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation_dir", default="artifacts/maskwam_libero_annotation_debug16/first_pass")
    parser.add_argument("--manifest", default="artifacts/maskwam_libero_annotation_debug16/manifest.json")
    parser.add_argument("--output_dir", default="artifacts/maskwam_libero_annotation_debug16/review_first_pass")
    parser.add_argument("--review_frames", nargs="+", type=int, default=None)
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def rel(path: Path, base: Path) -> str:
    return html.escape(os.path.relpath(path.resolve(), base.resolve().parent))


def font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def load_rgb(sample_dir: Path, frame_idx: int) -> np.ndarray:
    return np.asarray(Image.open(sample_dir / "rgb" / f"frame_{frame_idx:03d}.png").convert("RGB"))


def load_mask(sample_dir: Path, role: str, frame_idx: int, shape: Tuple[int, int]) -> np.ndarray:
    path = sample_dir / "masks" / role / f"frame_{frame_idx:03d}.png"
    if not path.exists():
        return np.zeros(shape, dtype=bool)
    return np.asarray(Image.open(path).convert("L")) > 0


def overlay(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    out = image.astype(np.float32).copy()
    if mask.any():
        color_arr = np.asarray(color, dtype=np.float32)
        out[mask] = (1.0 - alpha) * out[mask] + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def crop_view(array: np.ndarray, view_layout: str, view: str) -> np.ndarray:
    slices = frame_view_slices(array.shape[:2], view_layout)
    ys, xs = slices.get(view, (slice(0, array.shape[0]), slice(0, array.shape[1])))
    return array[ys, xs]


def bordered(image: Image.Image, label: str, color: Tuple[int, int, int], label_font: ImageFont.ImageFont) -> Image.Image:
    pad_top = 24
    out = Image.new("RGB", (image.width, image.height + pad_top), (248, 249, 251))
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, out.width - 1, out.height - 1], outline=(218, 222, 228), width=1)
    draw.rectangle([0, 0, out.width - 1, pad_top - 1], fill=color)
    draw.text((6, 4), label, fill=(255, 255, 255), font=label_font)
    out.paste(image, (0, pad_top))
    return out


def make_sample_sheet(sample_meta: Dict[str, Any], annotation_dir: Path, output_path: Path, review_frames: Sequence[int]) -> None:
    sample = sample_meta["sample"]
    sample_dir = annotation_dir / sample["sample_name"]
    label_font = font(14)
    body_font = font(15)
    title_font = font(20)
    frames = []
    view_layout = sample.get("view_layout", "vertical")
    for frame_idx in review_frames:
        rgb = load_rgb(sample_dir, frame_idx)
        shape = rgb.shape[:2]
        final_mask = load_mask(sample_dir, "final_train_mask", frame_idx, shape)
        col = [
            bordered(Image.fromarray(rgb), f"F{frame_idx:02d} composed RGB", (80, 86, 96), label_font),
        ]
        for view in VIEW_NAMES:
            rgb_view = crop_view(rgb, view_layout, view)
            final_view = crop_view(final_mask, view_layout, view)
            role_overlay = rgb.copy()
            for base in BASE_TRAIN_ROLES:
                role = view_role(view, base)
                role_overlay = overlay(role_overlay, load_mask(sample_dir, role, frame_idx, shape), ROLE_COLORS[role], 0.42)
            role_view = crop_view(role_overlay, view_layout, view)
            final_overlay = overlay(rgb, final_mask, ROLE_COLORS[FINAL_ROLE], 0.50)
            final_overlay_view = crop_view(final_overlay, view_layout, view)
            col.extend(
                [
                    bordered(Image.fromarray(rgb_view), f"{view} RGB", (80, 86, 96), label_font),
                    bordered(Image.fromarray(role_view), f"{view} roles", (35, 92, 150), label_font),
                    bordered(Image.fromarray(final_overlay_view), f"{view} final", (180, 45, 45), label_font),
                ]
            )
        col_w = max(cell.width for cell in col)
        col_h = sum(cell.height for cell in col)
        col_img = Image.new("RGB", (col_w, col_h), (255, 255, 255))
        y = 0
        for cell in col:
            col_img.paste(cell, (0, y))
            y += cell.height
        frames.append(col_img)

    gutter = 10
    header_h = 134
    legend_h = 34
    width = sum(frame.width for frame in frames) + gutter * (len(frames) - 1)
    height = header_h + max(frame.height for frame in frames) + legend_h
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), sample["sample_name"], fill=(18, 24, 32), font=title_font)
    draw.text((8, 38), sample.get("task_text", ""), fill=(42, 48, 56), font=body_font)
    draw.text((8, 66), "Human action: inspect main and wrist separately; mark OK only when both views are correct.", fill=(88, 96, 105), font=body_font)
    draw.text((8, 88), "View-aware roles: main_object/main_target/main_robot and wrist_object/wrist_target/wrist_robot.", fill=(88, 96, 105), font=body_font)
    x = 8
    legend_y = 112
    for role, color in [("object", ROLE_COLORS["object"]), ("target", ROLE_COLORS["target"]), ("robot", ROLE_COLORS["robot"]), (FINAL_ROLE, ROLE_COLORS[FINAL_ROLE])]:
        draw.rectangle([x, legend_y, x + 18, legend_y + 18], fill=color)
        draw.text((x + 24, legend_y), role, fill=(45, 50, 58), font=body_font)
        x += 155

    x = 0
    for frame in frames:
        sheet.paste(frame, (x, header_h))
        x += frame.width + gutter
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def write_review_csv(samples: List[Dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample",
        "task_text",
        "allowed_statuses",
        "human_status",
        "needs_correction",
        "correction_hint",
        "notes",
    ]
    with open(output_csv, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "sample": sample["sample_name"],
                    "task_text": sample.get("task_text", ""),
                    "allowed_statuses": " ".join(REVIEW_STATUSES),
                    "human_status": "",
                    "needs_correction": "",
                    "correction_hint": "If not OK, add point/box prompts in corrections_template.json.",
                    "notes": "",
                }
            )


def write_auto_diagnostics(samples: List[Dict[str, Any]], annotation_dir: Path, output_csv: Path) -> None:
    rows = []
    for sample in samples:
        meta_path = annotation_dir / sample["sample_name"] / "annotation_metadata.json"
        if not meta_path.exists():
            continue
        meta = load_json(meta_path)
        row = {
            "sample": sample["sample_name"],
            "task_text": sample.get("task_text", ""),
            "final_zero": meta.get(FINAL_ROLE, {}).get("zero_frame_count", ""),
            "final_mean": meta.get(FINAL_ROLE, {}).get("mean_area_ratio", ""),
        }
        for role in [*VIEW_TRAIN_ROLES, "object", "target", "robot"]:
            stats = meta.get("role_summaries", {}).get(role, {})
            row[f"{role}_zero"] = stats.get("zero_frame_count", "")
            row[f"{role}_mean"] = stats.get("mean_area_ratio", "")
        rows.append(row)
    if not rows:
        return
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def copy_click_frames(
    samples: List[Dict[str, Any]],
    annotation_dir: Path,
    output_dir: Path,
    review_frame_map: Dict[str, Sequence[int]],
) -> Dict[str, List[Path]]:
    click_dir = output_dir / "click_frames"
    click_dir.mkdir(parents=True, exist_ok=True)
    paths_by_sample: Dict[str, List[Path]] = {}
    for sample in samples:
        sample_name = sample["sample_name"]
        paths_by_sample[sample_name] = []
        for frame_idx in review_frame_map[sample_name]:
            src = annotation_dir / sample_name / "rgb" / f"frame_{frame_idx:03d}.png"
            if not src.exists():
                continue
            dst = click_dir / f"{sample_name}_frame_{frame_idx:03d}.png"
            shutil.copyfile(src, dst)
            paths_by_sample[sample_name].append(dst)
    return paths_by_sample


def write_corrections_template(samples: List[Dict[str, Any]], output_json: Path) -> None:
    payload = {
        "schema_version": "maskwam_corrections_v1",
        "notes": "Fill only samples that need correction. Coordinates are absolute pixels on the composed 448x224 frame.",
        "samples": {
            str(sample["sample_id"]): {
                "status": "",
                "notes": "",
                "prompts": [
                    {
                        "name": sample["prompts"][0]["name"],
                        "role": sample["prompts"][0]["role"],
                        "frame_index": 0,
                        "points_abs": [],
                        "point_labels": [],
                        "boxes_xywh_abs": [],
                        "box_labels": []
                    }
                ],
            }
            for sample in samples
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def write_html(
    samples: List[Dict[str, Any]],
    sheet_paths: List[Path],
    click_paths_by_sample: Dict[str, List[Path]],
    output_html: Path,
) -> None:
    correction_tasks_json = output_html.parent / "correction_tasks.json"
    task_payload: Dict[str, Any] = {
        "schema_version": "maskwam_correction_tasks_v1",
        "task_count": 0,
        "tasks": [],
        "by_priority": {},
    }
    if correction_tasks_json.exists():
        task_payload = load_json(correction_tasks_json)
    task_payload_json = json.dumps(task_payload, ensure_ascii=False)
    cards = []
    for sample, sheet_path in zip(samples, sheet_paths):
        href = rel(sheet_path, output_html)
        prompt_options = [
            "<option value='manual' data-role='' data-name=''>manual role/name</option>"
        ]
        for prompt in sample.get("prompts", []):
            if prompt.get("role") == "context":
                continue
            name = html.escape(str(prompt.get("name", "")), quote=True)
            for view in VIEW_NAMES:
                role_value = view_role(view, str(prompt.get("role", "")))
                role = html.escape(role_value, quote=True)
                label = html.escape(f"{role_value}: {prompt.get('name', '')}")
                prompt_options.append(
                    f"<option value='{role}:{name}' data-role='{role}' data-name='{name}'>{label}</option>"
                )
        click_imgs = []
        for image_path in click_paths_by_sample.get(sample["sample_name"], []):
            image_href = rel(image_path, output_html)
            frame_label = image_path.stem.rsplit("_frame_", 1)[-1]
            click_imgs.append(
                f"<figure><figcaption>click frame {html.escape(frame_label)}</figcaption>"
                f"<img class='click-frame' src='{image_href}' data-sample='{html.escape(str(sample['sample_id']))}' "
                f"data-sample-name='{html.escape(sample['sample_name'])}' data-frame='{html.escape(str(int(frame_label)))}' "
                "data-role='main_object' data-name='manual point' alt='click target'></figure>"
            )
        cards.append(
            "\n".join(
                [
                    f"<section class='case' id='{html.escape(sample['sample_name'])}'>",
                    f"<h2>{html.escape(sample['sample_name'])}</h2>",
                    f"<p>{html.escape(sample.get('task_text', ''))}</p>",
                    "<div class='case-controls'>",
                    "<label>prompt <select class='case-prompt'>",
                    "".join(prompt_options),
                    "</select></label>",
                    "</div>",
                    f"<a href='{href}'><img src='{href}' alt='{html.escape(sample['sample_name'])}'></a>",
                    "<div class='click-grid'>",
                    "".join(click_imgs),
                    "</div>",
                    "</section>",
                ]
            )
        )
    review_csv = output_html.parent / "human_review_template.csv"
    corrections = output_html.parent / "corrections_template.json"
    diagnostics = output_html.parent / "auto_diagnostics.csv"
    correction_tasks = output_html.parent / "correction_tasks.csv"
    correction_tasks_html = output_html.parent / "correction_tasks.html"
    correction_tasks_json_link = output_html.parent / "correction_tasks.json"
    correction_draft = output_html.parent / "corrections_tasks_draft.json"
    pending_summary_md = output_html.parent / "pending_corrections_summary.md"
    pending_summary_csv = output_html.parent / "pending_corrections_summary.csv"
    pending_summary_json = output_html.parent / "pending_corrections_summary.json"
    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>MaskWAM LIBERO Annotation Review</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1f2933; background: #f6f7f9; }}
    h1 {{ font-size: 26px; margin-bottom: 6px; }}
    .top {{ background: white; border: 1px solid #dde1e7; padding: 16px; border-radius: 8px; margin-bottom: 16px; }}
    .case {{ background: white; border: 1px solid #dde1e7; padding: 14px; border-radius: 8px; margin-bottom: 18px; }}
    .case h2 {{ font-size: 20px; margin: 0 0 4px; }}
    .case p {{ color: #52606d; margin: 0 0 10px; }}
    .case-controls {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 8px 0 12px; align-items: center; }}
    .case-controls label {{ font-size: 13px; color: #52606d; }}
    .case-controls select {{ font-size: 13px; padding: 4px 6px; border: 1px solid #cbd2d9; border-radius: 4px; }}
    img {{ max-width: 100%; height: auto; display: block; border: 1px solid #d7dce2; }}
    .click-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; margin-top: 12px; }}
    .click-grid figure {{ margin: 0; }}
    .click-grid figcaption {{ font-size: 12px; color: #52606d; margin-bottom: 4px; }}
    .click-frame {{ cursor: crosshair; }}
    .click-frame.recommended-frame {{ outline: 4px solid #dd6b20; outline-offset: 2px; }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin: 10px 0; }}
    .controls label {{ font-size: 13px; color: #52606d; }}
    .controls input, .controls select {{ font-size: 13px; padding: 4px 6px; border: 1px solid #cbd2d9; border-radius: 4px; }}
    .controls button {{ font-size: 13px; padding: 5px 10px; border: 1px solid #a7b1bd; border-radius: 4px; background: #fff; cursor: pointer; }}
    .task-panel {{ background: #fff; border: 1px solid #dde1e7; padding: 14px; border-radius: 8px; margin-bottom: 16px; }}
    .task-panel h2 {{ font-size: 20px; margin: 0 0 8px; }}
    .task-summary {{ color: #52606d; font-size: 13px; margin-bottom: 10px; }}
    .task-list {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 8px; }}
    .task-button {{ text-align: left; border: 1px solid #d8dee6; border-radius: 6px; background: #fff; padding: 8px; cursor: pointer; }}
    .task-button:hover {{ border-color: #7b93ad; background: #f8fafc; }}
    .task-button.aggregate {{ opacity: 0.74; }}
    .task-button.covered {{ border-color: #38a169; background: #f0fff4; }}
    .task-button.uncovered-critical {{ border-color: #e53e3e; }}
    .task-button.uncovered-actionable {{ border-color: #dd6b20; background: #fffaf0; }}
    .task-title {{ font-weight: 650; color: #1f2933; margin-bottom: 3px; }}
    .task-meta {{ font-size: 12px; color: #52606d; }}
    .task-reason {{ font-size: 12px; color: #334e68; margin-top: 4px; }}
    .coverage-bar {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0; }}
    .coverage-pill {{ border: 1px solid #d8dee6; border-radius: 999px; padding: 4px 9px; font-size: 12px; background: #fff; color: #334e68; }}
    .coverage-pill.ready {{ border-color: #38a169; background: #f0fff4; color: #276749; }}
    .coverage-pill.blocked {{ border-color: #e53e3e; background: #fff5f5; color: #9b2c2c; }}
    .autosave-status {{ color: #52606d; font-size: 12px; margin-left: 8px; }}
    .case.active-task {{ outline: 3px solid #2f80ed; outline-offset: 2px; }}
    textarea {{ width: 100%; min-height: 180px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }}
    code {{ background: #eef1f5; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <div class="top">
    <h1>MaskWAM LIBERO Annotation Review</h1>
    <p>Human action is intentionally small: mark each case as <code>ok</code>, <code>wrong_object</code>, <code>missing_target</code>, <code>too_broad</code>, <code>bad_tracking</code>, or <code>exclude</code>.</p>
    <p>Review CSV: <a href="{rel(review_csv, output_html)}">human_review_template.csv</a></p>
    <p>Correction template: <a href="{rel(corrections, output_html)}">corrections_template.json</a></p>
    <p>Correction tasks: <a href="{rel(correction_tasks, output_html)}">correction_tasks.csv</a></p>
    <p>Correction dashboard: <a href="{rel(correction_tasks_html, output_html)}">correction_tasks.html</a></p>
    <p>Correction task JSON: <a href="{rel(correction_tasks_json_link, output_html)}">correction_tasks.json</a></p>
    <p>Correction draft: <a href="{rel(correction_draft, output_html)}">corrections_tasks_draft.json</a></p>
    <p>Pending summary: <a href="{rel(pending_summary_md, output_html)}">pending_corrections_summary.md</a> · <a href="{rel(pending_summary_csv, output_html)}">csv</a> · <a href="{rel(pending_summary_json, output_html)}">json</a></p>
    <p>Auto diagnostics: <a href="{rel(diagnostics, output_html)}">auto_diagnostics.csv</a></p>
    <p>Click a raw frame below to update a complete corrections JSON. Coordinates are absolute pixels on the composed 224x448 frame. Use <code>role=target</code> for plate, basket, drawer, cabinet-top, or placement corrections.</p>
    <div class="controls">
      <label>role
        <select id="point-role">
          {''.join(f"<option value='{role}'>{role}</option>" for role in VIEW_TRAIN_ROLES)}
        </select>
      </label>
      <label>name <input id="point-name" value="manual point"></label>
      <label>mode
        <select id="prompt-mode">
          <option value="point">point</option>
          <option value="box">box</option>
        </select>
      </label>
      <label>label
        <select id="prompt-label">
          <option value="1">positive</option>
          <option value="0">negative</option>
        </select>
      </label>
      <button id="reset-json" type="button">reset JSON</button>
      <button id="undo-json" type="button">undo last prompt</button>
      <button id="load-json" type="button">load JSON from textarea</button>
      <button id="download-json" type="button">download corrections JSON</button>
      <button id="clear-saved-json" type="button">clear saved copy</button>
    </div>
    <p><span id="box-status">Box mode: click top-left then bottom-right.</span><span id="autosave-status" class="autosave-status"></span></p>
    <textarea id="point-log" placeholder="Click frames below, then copy this complete JSON into corrections_template.json."></textarea>
  </div>
  <div class="task-panel">
    <h2>Correction Tasks</h2>
    <div id="task-summary" class="task-summary">No correction_tasks.json found in this review directory.</div>
    <div id="coverage-summary" class="coverage-bar"></div>
    <div id="task-list" class="task-list"></div>
  </div>
  {''.join(cards)}
  <script>
    const taskPayload = {task_payload_json};
    const pointLog = document.getElementById('point-log');
    const pointRole = document.getElementById('point-role');
    const pointName = document.getElementById('point-name');
    const promptMode = document.getElementById('prompt-mode');
    const promptLabel = document.getElementById('prompt-label');
    const boxStatus = document.getElementById('box-status');
    const resetButton = document.getElementById('reset-json');
    const undoButton = document.getElementById('undo-json');
    const loadButton = document.getElementById('load-json');
    const downloadButton = document.getElementById('download-json');
    const clearSavedButton = document.getElementById('clear-saved-json');
    const autosaveStatus = document.getElementById('autosave-status');
    const taskSummary = document.getElementById('task-summary');
    const coverageSummary = document.getElementById('coverage-summary');
    const taskList = document.getElementById('task-list');
    let pendingBox = null;
    let correctionHistory = [];
    const storageKey = `maskwam_review_corrections:${{taskPayload.manifest || window.location.pathname}}`;
    function defaultCorrectionData() {{
      return {{
        schema_version: "maskwam_corrections_v1",
        notes: "Generated from review HTML clicks. Coordinates are absolute pixels on the composed 224x448 frame.",
        samples: {{}}
      }};
    }}
    let correctionData = defaultCorrectionData();
    function autosaveCorrectionData() {{
      try {{
        localStorage.setItem(storageKey, JSON.stringify({{
          saved_at: new Date().toISOString(),
          correctionData
        }}));
        autosaveStatus.textContent = "autosaved";
      }} catch (error) {{
        autosaveStatus.textContent = `autosave failed: ${{error.message}}`;
      }}
    }}
    function restoreSavedCorrections() {{
      try {{
        const raw = localStorage.getItem(storageKey);
        if (!raw) {{
          autosaveStatus.textContent = "no saved copy";
          return false;
        }}
        const parsed = JSON.parse(raw);
        const restored = parsed.correctionData || parsed;
        if (!restored || typeof restored !== "object" || !restored.samples || typeof restored.samples !== "object") {{
          autosaveStatus.textContent = "saved copy ignored";
          return false;
        }}
        correctionData = restored;
        autosaveStatus.textContent = `restored ${{parsed.saved_at || "saved copy"}}`;
        return true;
      }} catch (error) {{
        autosaveStatus.textContent = `restore failed: ${{error.message}}`;
        return false;
      }}
    }}
    function refreshJson(save = true) {{
      pointLog.value = JSON.stringify(correctionData, null, 2);
      renderTasks();
      if (save) {{
        autosaveCorrectionData();
      }}
    }}
    function saveHistory() {{
      correctionHistory.push(JSON.stringify(correctionData));
      if (correctionHistory.length > 100) {{
        correctionHistory.shift();
      }}
    }}
    function resetCorrectionData() {{
      correctionData = defaultCorrectionData();
    }}
    resetButton.addEventListener('click', () => {{
      saveHistory();
      resetCorrectionData();
      pendingBox = null;
      boxStatus.textContent = "Box mode: click top-left then bottom-right.";
      refreshJson();
    }});
    undoButton.addEventListener('click', () => {{
      if (!correctionHistory.length) {{
        boxStatus.textContent = "Nothing to undo.";
        return;
      }}
      correctionData = JSON.parse(correctionHistory.pop());
      pendingBox = null;
      boxStatus.textContent = "Undid the last correction edit.";
      refreshJson();
    }});
    loadButton.addEventListener('click', () => {{
      try {{
        const parsed = JSON.parse(pointLog.value);
        if (!parsed || typeof parsed !== "object" || !parsed.samples || typeof parsed.samples !== "object") {{
          throw new Error("JSON must contain a samples object.");
        }}
        saveHistory();
        correctionData = parsed;
        pendingBox = null;
        boxStatus.textContent = "Loaded corrections JSON from textarea.";
        refreshJson();
      }} catch (error) {{
        boxStatus.textContent = `Could not load JSON: ${{error.message}}`;
      }}
    }});
    downloadButton.addEventListener('click', () => {{
      const blob = new Blob([JSON.stringify(correctionData, null, 2)], {{ type: "application/json" }});
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = "corrections_from_html.json";
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(link.href);
    }});
    clearSavedButton.addEventListener('click', () => {{
      localStorage.removeItem(storageKey);
      autosaveStatus.textContent = "saved copy cleared";
    }});
    restoreSavedCorrections();
    refreshJson(false);
    function ensureSample(sample, sampleName) {{
      if (!correctionData.samples[sample]) {{
        correctionData.samples[sample] = {{
          status: "needs_correction",
          notes: sampleName,
          prompts: []
        }};
      }}
      return correctionData.samples[sample];
    }}
    function listValue(value) {{
      return Array.isArray(value) ? value : [];
    }}
    function isActivePrompt(prompt) {{
      return Boolean(
        listValue(prompt.points_abs).length ||
        listValue(prompt.points_rel).length ||
        listValue(prompt.boxes_xywh_abs).length ||
        listValue(prompt.boxes_xywh_rel).length
      );
    }}
    function activePrompts() {{
      const prompts = [];
      Object.entries(correctionData.samples || {{}}).forEach(([sampleId, sample]) => {{
        listValue(sample.prompts).forEach((prompt, promptIdx) => {{
          if (prompt && typeof prompt === "object" && isActivePrompt(prompt)) {{
            prompts.push({{
              sample_id: Number(sampleId),
              prompt_idx: promptIdx,
              role: prompt.role,
              name: prompt.name,
              frame_index: Number(prompt.frame_index || 0)
            }});
          }}
        }});
      }});
      return prompts;
    }}
    const viewTrainRoles = {json.dumps(list(VIEW_TRAIN_ROLES))};
    function isActionableTask(task) {{
      return viewTrainRoles.includes(task.role) && Boolean(task.name);
    }}
    function promptMatchesTask(prompt, task) {{
      if (Number(prompt.sample_id) !== Number(task.sample_id)) {{
        return false;
      }}
      if (prompt.role !== task.role) {{
        return false;
      }}
      if (prompt.name === task.name) {{
        return true;
      }}
      return ["", "manual point", "manual box", null, undefined].includes(prompt.name);
    }}
    function coverageForTask(task, prompts) {{
      if (!isActionableTask(task)) {{
        return {{
          actionable: false,
          covered: false,
          aggregate: task.role === "final_train_mask",
          matches: []
        }};
      }}
      const matches = prompts.filter((prompt) => promptMatchesTask(prompt, task));
      return {{
        actionable: true,
        covered: matches.length > 0,
        aggregate: false,
        matches
      }};
    }}
    function buildCoverage() {{
      const tasks = Array.isArray(taskPayload.tasks) ? taskPayload.tasks : [];
      const prompts = activePrompts();
      const rows = tasks.map((task) => ({{
        task,
        ...coverageForTask(task, prompts)
      }}));
      const actionable = rows.filter((row) => row.actionable);
      const covered = actionable.filter((row) => row.covered);
      const critical = actionable.filter((row) => row.task.priority === "critical");
      const criticalCovered = critical.filter((row) => row.covered);
      const high = actionable.filter((row) => row.task.priority === "high");
      const highCovered = high.filter((row) => row.covered);
      const medium = actionable.filter((row) => row.task.priority === "medium");
      const mediumCovered = medium.filter((row) => row.covered);
      const matchedKeys = new Set();
      rows.forEach((row) => {{
        row.matches.forEach((prompt) => matchedKeys.add(`${{prompt.sample_id}}:${{prompt.prompt_idx}}`));
      }});
      const unmatchedActive = prompts.filter((prompt) => !matchedKeys.has(`${{prompt.sample_id}}:${{prompt.prompt_idx}}`));
      return {{
        rows,
        actionable,
        covered,
        critical,
        criticalCovered,
        high,
        highCovered,
        medium,
        mediumCovered,
        activePromptCount: prompts.length,
        unmatchedActive
      }};
    }}
    function renderCoverage(coverage) {{
      const allCriticalCovered = coverage.critical.length > 0 && coverage.criticalCovered.length === coverage.critical.length;
      const allActionableCovered = coverage.actionable.length > 0 && coverage.covered.length === coverage.actionable.length;
      const pill = (label, ok) => `<span class="coverage-pill ${{ok ? "ready" : "blocked"}}">${{label}}</span>`;
      coverageSummary.innerHTML = [
        pill(`formal train-ready ${{coverage.covered.length}}/${{coverage.actionable.length}}`, allActionableCovered),
        pill(`critical ${{coverage.criticalCovered.length}}/${{coverage.critical.length}}`, allCriticalCovered),
        `<span class="coverage-pill">high ${{coverage.highCovered.length}}/${{coverage.high.length}}</span>`,
        `<span class="coverage-pill">medium ${{coverage.mediumCovered.length}}/${{coverage.medium.length}}</span>`,
        `<span class="coverage-pill">active prompts ${{coverage.activePromptCount}}</span>`,
        `<span class="coverage-pill ${{coverage.unmatchedActive.length ? "blocked" : "ready"}}">unmatched ${{coverage.unmatchedActive.length}}</span>`
      ].join("");
    }}
    function selectedPrompt(img, mode) {{
      const section = img.closest('.case');
      const select = section ? section.querySelector('.case-prompt') : null;
      const option = select ? select.selectedOptions[0] : null;
      if (option && option.value !== "manual") {{
        return {{
          role: option.dataset.role || "object",
          name: option.dataset.name || "manual point"
        }};
      }}
      let typedName = pointName.value || "";
      if (!typedName || (mode === "box" && typedName === "manual point")) {{
        typedName = mode === "box" ? "manual box" : "manual point";
      }}
      return {{
        role: pointRole.value || "object",
        name: typedName
      }};
    }}
    function setManualPrompt(role, name) {{
      pointRole.value = role || "object";
      pointName.value = name || "manual point";
    }}
    function chooseCasePrompt(section, role, name) {{
      const select = section ? section.querySelector('.case-prompt') : null;
      if (!select) {{
        setManualPrompt(role, name);
        return;
      }}
      const value = `${{role}}:${{name}}`;
      const option = Array.from(select.options).find((item) => item.value === value);
      if (option) {{
        select.value = value;
      }} else {{
        select.value = "manual";
        setManualPrompt(role, name);
      }}
    }}
    function focusTask(task) {{
      const section = document.getElementById(task.sample_name);
      if (!section) {{
        return;
      }}
      section.scrollIntoView({{ behavior: "smooth", block: "start" }});
      section.classList.add('active-task');
      setTimeout(() => section.classList.remove('active-task'), 1800);
      document.querySelectorAll('.click-frame.recommended-frame').forEach((img) => img.classList.remove('recommended-frame'));
      if (task.recommended_frame_index !== undefined && task.recommended_frame_index !== null && task.recommended_frame_index !== "") {{
        const frame = String(task.recommended_frame_index).padStart(3, "0");
        const recommendedImage = section.querySelector(`.click-frame[data-frame="${{frame}}"]`) || section.querySelector(`.click-frame[data-frame="${{Number(task.recommended_frame_index)}}"]`);
        if (recommendedImage) {{
          recommendedImage.classList.add('recommended-frame');
          setTimeout(() => recommendedImage.classList.remove('recommended-frame'), 5000);
        }}
      }}
      if (task.role && task.role !== "final_train_mask") {{
        chooseCasePrompt(section, task.role, task.name);
        promptMode.value = task.priority === "critical" ? "box" : "point";
        promptLabel.value = "1";
        boxStatus.textContent = `Selected ${{task.priority}} task: ${{task.sample_name}} ${{task.role}}:${{task.name}}. Recommended frame F${{task.recommended_frame_index ?? "?"}}.`;
      }} else {{
        boxStatus.textContent = `Selected aggregate task: ${{task.sample_name}}. Fix object/target/robot tasks for this sample.`;
      }}
    }}
    function renderTasks() {{
      const tasks = Array.isArray(taskPayload.tasks) ? taskPayload.tasks : [];
      const byPriority = taskPayload.by_priority || {{}};
      if (!tasks.length) {{
        taskSummary.textContent = "No embedded correction tasks. Run maskwam_make_correction_tasks.py, then regenerate this review page.";
        coverageSummary.innerHTML = "";
        return;
      }}
      const coverage = buildCoverage();
      renderCoverage(coverage);
      taskSummary.textContent = `Formal target: cover every object/target/robot task before running --require_all_actionable. tasks=${{tasks.length}} priorities=${{JSON.stringify(byPriority)}}. Click a task to jump to that sample and select its prompt.`;
      taskList.innerHTML = "";
      tasks.forEach((task, idx) => {{
        const row = coverage.rows[idx] || {{ actionable: false, covered: false, matches: [] }};
        const classes = ["task-button"];
        if (task.role === "final_train_mask") {{
          classes.push("aggregate");
        }}
        if (row.covered) {{
          classes.push("covered");
        }}
        if (row.actionable && !row.covered && task.priority === "critical") {{
          classes.push("uncovered-critical");
        }} else if (row.actionable && !row.covered) {{
          classes.push("uncovered-actionable");
        }}
        const coverageText = row.actionable
          ? (row.covered ? `covered by ${{row.matches.length}} prompt(s)` : "not covered")
          : "aggregate";
        const button = document.createElement('button');
        button.type = "button";
        button.className = classes.join(" ");
        button.innerHTML = `
          <div class="task-title">${{task.priority}} · ${{task.sample_name}} · ${{task.role}}:${{task.name}}</div>
          <div class="task-meta">zero=${{task.zero_frame_count}} mean=${{task.mean_area_ratio}} · recommended F${{task.recommended_frame_index ?? "?"}} · ${{coverageText}}</div>
          <div class="task-meta">${{task.recommended_frame_reason || ""}}</div>
          <div class="task-reason">${{task.reason}}</div>
        `;
        button.addEventListener('click', () => focusTask(task));
        taskList.appendChild(button);
      }});
    }}
    renderTasks();
    document.querySelectorAll('.click-frame').forEach((img) => {{
      img.addEventListener('click', (event) => {{
        const rect = img.getBoundingClientRect();
        const scaleX = img.naturalWidth / rect.width;
        const scaleY = img.naturalHeight / rect.height;
        const x = Math.round((event.clientX - rect.left) * scaleX);
        const y = Math.round((event.clientY - rect.top) * scaleY);
        const sample = img.dataset.sample;
        const sampleName = img.dataset.sampleName;
        const frame = Number(img.dataset.frame);
        const mode = promptMode.value || "point";
        const promptInfo = selectedPrompt(img, mode);
        const label = Number(promptLabel.value || 1);
        if (mode === "box") {{
          if (!pendingBox || pendingBox.sample !== sample || pendingBox.frame !== frame) {{
            pendingBox = {{ sample, sampleName, frame, x, y }};
            boxStatus.textContent = `Box start: sample ${{sample}}, frame ${{frame}}, (${{x}}, ${{y}}). Click bottom-right.`;
            return;
          }}
          const x0 = Math.min(pendingBox.x, x);
          const y0 = Math.min(pendingBox.y, y);
          const w = Math.abs(x - pendingBox.x);
          const h = Math.abs(y - pendingBox.y);
          if (w > 0 && h > 0) {{
            saveHistory();
            const sampleEntry = ensureSample(sample, sampleName);
            sampleEntry.prompts.push({{
              name: promptInfo.name,
              role: promptInfo.role,
              frame_index: frame,
              points_abs: [],
              point_labels: [],
              boxes_xywh_abs: [[x0, y0, w, h]],
              box_labels: [label]
            }});
          }}
          pendingBox = null;
          boxStatus.textContent = "Box mode: click top-left then bottom-right.";
          refreshJson();
          return;
        }}
        saveHistory();
        const sampleEntry = ensureSample(sample, sampleName);
        sampleEntry.prompts.push({{
          name: promptInfo.name,
          role: promptInfo.role,
          frame_index: frame,
          points_abs: [[x, y]],
          point_labels: [label],
          boxes_xywh_abs: [],
          box_labels: []
        }});
        refreshJson();
      }});
    }});
  </script>
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    annotation_dir = Path(args.annotation_dir)
    output_dir = Path(args.output_dir)
    manifest = load_json(args.manifest)
    samples = manifest["samples"]
    sheet_paths: List[Path] = []
    review_frame_map: Dict[str, Sequence[int]] = {}
    for sample in samples:
        sample_dir = annotation_dir / sample["sample_name"]
        meta_path = sample_dir / "annotation_metadata.json"
        if not meta_path.exists():
            continue
        sample_meta = load_json(meta_path)
        review_frames = args.review_frames or sample.get("review_frame_indices", [0, 4, 8, 12, 16])
        review_frame_map[sample["sample_name"]] = review_frames
        output_path = output_dir / "sheets" / f"{sample['sample_name']}_review.png"
        make_sample_sheet(sample_meta, annotation_dir, output_path, review_frames)
        sheet_paths.append(output_path)

    reviewed_samples = [sample for sample in samples if (annotation_dir / sample["sample_name"] / "annotation_metadata.json").exists()]
    write_review_csv(reviewed_samples, output_dir / "human_review_template.csv")
    write_auto_diagnostics(reviewed_samples, annotation_dir, output_dir / "auto_diagnostics.csv")
    write_corrections_template(reviewed_samples, output_dir / "corrections_template.json")
    click_paths_by_sample = copy_click_frames(reviewed_samples, annotation_dir, output_dir, review_frame_map)
    write_html(reviewed_samples, sheet_paths, click_paths_by_sample, output_dir / "index.html")
    print(output_dir / "index.html")


if __name__ == "__main__":
    main()
