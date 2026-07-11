#!/usr/bin/env python3
"""Create a focused task-queue annotator for MaskWAM point/box corrections."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from maskwam_view_schema import VIEW_TRAIN_ROLES


DEFAULT_REVIEW_DIR = Path("artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click")
TRAIN_ROLES = set(VIEW_TRAIN_ROLES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review_dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def rel(path: Path, base_dir: Path) -> str:
    return os.path.relpath(path.resolve(), base_dir.resolve())


def frame_index_from_path(path: Path) -> int:
    match = re.search(r"_frame_(\d+)\.png$", path.name)
    if not match:
        raise ValueError(f"cannot parse frame index from {path}")
    return int(match.group(1))


def resolve_task_frame_path(raw_path: str, review_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or path.exists():
        return path
    review_path = review_dir / path
    if review_path.exists():
        return review_path
    if raw_path.startswith("artifacts/"):
        artifact_root = next((parent for parent in review_dir.resolve().parents if parent.name == "artifacts"), None)
        if artifact_root is not None:
            return artifact_root.parent / path
    return review_path


def build_tasks(tasks_payload: Dict[str, Any], review_dir: Path, output_dir: Path) -> List[Dict[str, Any]]:
    tasks = []
    for task in tasks_payload.get("tasks", []):
        if task.get("role") not in TRAIN_ROLES or not task.get("name"):
            continue
        frames = []
        for raw_path in str(task.get("click_frames", "")).split():
            frame_path = resolve_task_frame_path(raw_path, review_dir)
            frames.append(
                {
                    "frame_index": frame_index_from_path(frame_path),
                    "src": rel(frame_path, output_dir),
                }
            )
        frames.sort(key=lambda item: item["frame_index"])
        tasks.append(
            {
                "task_idx": len(tasks),
                "priority": task.get("priority", ""),
                "sample_id": int(task.get("sample_id")),
                "sample_name": task.get("sample_name", ""),
                "role": task.get("role", ""),
                "name": task.get("name", ""),
                "task_text": task.get("task_text", ""),
                "reason": task.get("reason", ""),
                "suggested_action": task.get("suggested_action", ""),
                "recommended_frame_index": int(task.get("recommended_frame_index", 0)),
                "recommended_frame_reason": task.get("recommended_frame_reason", ""),
                "zero_frame_count": task.get("zero_frame_count", ""),
                "zero_frame_indices": task.get("zero_frame_indices", ""),
                "frames": frames,
            }
        )
    return tasks


def write_html(output_path: Path, tasks: List[Dict[str, Any]], review_dir: Path) -> None:
    tasks_json = json.dumps(tasks, ensure_ascii=False)
    review_href = html.escape(rel(review_dir / "index.html", output_path.parent))
    workbench_href = html.escape(rel(review_dir / "correction_workbench" / "index.html", output_path.parent))
    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>MaskWAM Task Queue Annotator</title>
  <style>
    :root {{ color-scheme: light; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f4f6f8; color: #16202a; }}
    header {{ display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 14px 18px; background: #1f2d3d; color: #fff; position: sticky; top: 0; z-index: 4; }}
    header h1 {{ font-size: 21px; margin: 0; }}
    header a {{ color: #b9dcff; }}
    button, select {{ font: inherit; }}
    button {{ border: 1px solid #b8c2cc; background: #fff; border-radius: 6px; padding: 7px 10px; cursor: pointer; }}
    button.primary {{ background: #276ef1; border-color: #276ef1; color: #fff; }}
    button.danger {{ border-color: #d64545; color: #9b1c1c; }}
    button:disabled {{ opacity: 0.45; cursor: not-allowed; }}
    .layout {{ display: grid; grid-template-columns: minmax(300px, 380px) 1fr; min-height: calc(100vh - 58px); }}
    aside {{ border-right: 1px solid #d8dee6; background: #fff; overflow: auto; max-height: calc(100vh - 58px); }}
    .summary {{ padding: 12px 14px; border-bottom: 1px solid #e3e8ef; font-size: 13px; color: #425466; }}
    .task-list {{ padding: 8px; }}
    .task {{ width: 100%; text-align: left; margin: 0 0 7px; border: 1px solid #d8dee6; background: #fff; border-radius: 7px; padding: 9px; }}
    .task.active {{ border-color: #276ef1; box-shadow: 0 0 0 2px rgba(39, 110, 241, 0.12); }}
    .task.covered {{ border-color: #2f9e44; background: #f1fff5; }}
    .task.critical:not(.covered) {{ border-color: #d64545; }}
    .task.high:not(.covered), .task.medium:not(.covered) {{ border-color: #dd8a00; background: #fffaf0; }}
    .task-title {{ font-weight: 700; font-size: 13px; margin-bottom: 3px; }}
    .task-meta {{ color: #536171; font-size: 12px; line-height: 1.3; }}
    main {{ padding: 16px 18px 36px; overflow: auto; }}
    .panel {{ background: #fff; border: 1px solid #d8dee6; border-radius: 8px; padding: 14px; margin-bottom: 14px; }}
    .current h2 {{ margin: 0 0 7px; font-size: 22px; }}
    .current p {{ margin: 5px 0; color: #425466; }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
    .controls label {{ display: inline-flex; align-items: center; gap: 5px; color: #425466; font-size: 13px; }}
    .frames {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }}
    .frame-button.recommended {{ border-color: #d64545; color: #9b1c1c; font-weight: 700; }}
    .frame-button.active {{ background: #1f2d3d; color: #fff; border-color: #1f2d3d; }}
    .canvas-wrap {{ display: inline-block; background: #e8edf3; padding: 12px; border-radius: 8px; border: 1px solid #d8dee6; }}
    canvas {{ display: block; width: min(448px, calc(100vw - 460px)); height: auto; cursor: crosshair; background: #000; }}
    .hint {{ color: #5b6778; font-size: 13px; margin-top: 8px; }}
    .json-row {{ display: grid; grid-template-columns: 1fr; gap: 8px; }}
    textarea {{ min-height: 210px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; border: 1px solid #cfd7e2; border-radius: 6px; padding: 8px; }}
    .status {{ font-size: 13px; color: #425466; }}
    .status strong {{ color: #16202a; }}
    @media (max-width: 900px) {{
      .layout {{ grid-template-columns: 1fr; }}
      aside {{ max-height: none; border-right: 0; border-bottom: 1px solid #d8dee6; }}
      canvas {{ width: min(448px, calc(100vw - 62px)); }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>MaskWAM Task Queue Annotator</h1>
    <div><a href="{review_href}">full review</a> · <a href="{workbench_href}">image workbench</a></div>
  </header>
  <div class="layout">
    <aside>
      <div id="summary" class="summary"></div>
      <div id="task-list" class="task-list"></div>
    </aside>
    <main>
      <section class="panel current">
        <h2 id="task-title"></h2>
        <p id="task-text"></p>
        <p id="task-reason"></p>
        <p id="task-action"></p>
        <div id="frame-buttons" class="frames"></div>
      </section>
      <section class="panel">
        <div class="controls">
          <label>mode
            <select id="mode">
              <option value="box">box drag</option>
              <option value="point">point click</option>
            </select>
          </label>
          <label>label
            <select id="label">
              <option value="1">positive</option>
              <option value="0">negative</option>
            </select>
          </label>
          <button id="prev-task" type="button">prev</button>
          <button id="next-task" type="button">next</button>
          <button id="next-open" type="button" class="primary">next uncovered</button>
          <button id="mark-absent" type="button">mark not visible</button>
          <button id="clear-task" type="button" class="danger">clear current task</button>
          <button id="undo" type="button">undo</button>
          <button id="download" type="button" class="primary">download corrections_from_html.json</button>
        </div>
        <div class="hint" id="draw-hint">Box mode: drag a tight box around the selected target on the selected frame.</div>
      </section>
      <section class="panel">
        <div class="canvas-wrap"><canvas id="canvas"></canvas></div>
        <div id="canvas-status" class="hint"></div>
      </section>
      <section class="panel json-row">
        <div class="controls">
          <button id="load-json" type="button">load JSON from textarea</button>
          <button id="clear-all" type="button" class="danger">clear all</button>
          <span id="save-status" class="status"></span>
        </div>
        <textarea id="json-box"></textarea>
      </section>
    </main>
  </div>
  <script>
    const TASKS = {tasks_json};
    const storageKey = `maskwam_task_queue:${{window.location.pathname}}`;
    const summaryEl = document.getElementById("summary");
    const taskListEl = document.getElementById("task-list");
    const taskTitleEl = document.getElementById("task-title");
    const taskTextEl = document.getElementById("task-text");
    const taskReasonEl = document.getElementById("task-reason");
    const taskActionEl = document.getElementById("task-action");
    const frameButtonsEl = document.getElementById("frame-buttons");
    const modeEl = document.getElementById("mode");
    const labelEl = document.getElementById("label");
    const hintEl = document.getElementById("draw-hint");
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const canvasStatusEl = document.getElementById("canvas-status");
    const jsonBoxEl = document.getElementById("json-box");
    const saveStatusEl = document.getElementById("save-status");

    let currentTaskIdx = 0;
    let currentFrameIndex = null;
    let image = new Image();
    let dragStart = null;
    let dragCurrent = null;
    let history = [];
    let corrections = {{
      schema_version: "maskwam_corrections_v1",
      notes: "Generated from MaskWAM task queue annotator. Coordinates are absolute pixels on the composed 224x448 frame.",
      samples: {{}}
    }};

    function taskKey(task) {{
      return `${{task.sample_id}}:${{task.role}}:${{task.name}}`;
    }}
    function clone(value) {{
      return JSON.parse(JSON.stringify(value));
    }}
    function saveHistory() {{
      history.push(JSON.stringify(corrections));
      if (history.length > 100) history.shift();
    }}
    function ensureSample(task) {{
      const key = String(task.sample_id);
      if (!corrections.samples[key]) {{
        corrections.samples[key] = {{
          status: "needs_correction",
          notes: `${{task.sample_name}}: ${{task.task_text}}`,
          prompts: []
        }};
      }}
      return corrections.samples[key];
    }}
    function promptMatchesTask(prompt, task) {{
      return prompt && prompt.role === task.role && prompt.name === task.name;
    }}
    function taskPrompts(task) {{
      const sample = corrections.samples[String(task.sample_id)];
      if (!sample || !Array.isArray(sample.prompts)) return [];
      return sample.prompts.filter((prompt) => promptMatchesTask(prompt, task));
    }}
    function isTaskCovered(task) {{
      return taskPrompts(task).some((prompt) => {{
        return prompt.accepted_absent || (prompt.points_abs && prompt.points_abs.length) || (prompt.boxes_xywh_abs && prompt.boxes_xywh_abs.length);
      }});
    }}
    function isTaskAcceptedAbsent(task) {{
      return taskPrompts(task).some((prompt) => prompt.accepted_absent);
    }}
    function framePrompts(task, frameIndex) {{
      return taskPrompts(task).filter((prompt) => Number(prompt.frame_index) === Number(frameIndex));
    }}
    function addPrompt(task, prompt) {{
      saveHistory();
      ensureSample(task).prompts.push(prompt);
      persist();
      renderAll();
    }}
    function removeCurrentTaskPrompts() {{
      const task = TASKS[currentTaskIdx];
      const sample = corrections.samples[String(task.sample_id)];
      if (!sample || !Array.isArray(sample.prompts)) return;
      saveHistory();
      sample.prompts = sample.prompts.filter((prompt) => !promptMatchesTask(prompt, task));
      if (!sample.prompts.length) delete corrections.samples[String(task.sample_id)];
      persist();
      renderAll();
    }}
    function persist() {{
      localStorage.setItem(storageKey, JSON.stringify({{ saved_at: new Date().toISOString(), corrections }}));
      jsonBoxEl.value = JSON.stringify(corrections, null, 2);
      saveStatusEl.innerHTML = "<strong>autosaved</strong>";
    }}
    function restore() {{
      const raw = localStorage.getItem(storageKey);
      if (!raw) {{
        jsonBoxEl.value = JSON.stringify(corrections, null, 2);
        saveStatusEl.textContent = "no saved copy";
        return;
      }}
      try {{
        const parsed = JSON.parse(raw);
        corrections = parsed.corrections || parsed;
        saveStatusEl.textContent = `restored ${{parsed.saved_at || "saved copy"}}`;
      }} catch (error) {{
        saveStatusEl.textContent = `restore failed: ${{error.message}}`;
      }}
      jsonBoxEl.value = JSON.stringify(corrections, null, 2);
    }}
    function coverage() {{
      const covered = TASKS.filter(isTaskCovered).length;
      const critical = TASKS.filter((task) => task.priority === "critical");
      const criticalCovered = critical.filter(isTaskCovered).length;
      const high = TASKS.filter((task) => task.priority === "high");
      const highCovered = high.filter(isTaskCovered).length;
      const medium = TASKS.filter((task) => task.priority === "medium");
      const mediumCovered = medium.filter(isTaskCovered).length;
      return {{ covered, total: TASKS.length, critical: critical.length, criticalCovered, high: high.length, highCovered, medium: medium.length, mediumCovered }};
    }}
    function renderTaskList() {{
      const cov = coverage();
      summaryEl.innerHTML = `formal train-ready <strong>${{cov.covered}}/${{cov.total}}</strong><br>` +
        `critical ${{cov.criticalCovered}}/${{cov.critical}} · high ${{cov.highCovered}}/${{cov.high}} · medium ${{cov.mediumCovered}}/${{cov.medium}}`;
      taskListEl.innerHTML = "";
      TASKS.forEach((task, idx) => {{
        const button = document.createElement("button");
        button.type = "button";
        button.className = `task ${{task.priority}} ${{idx === currentTaskIdx ? "active" : ""}} ${{isTaskCovered(task) ? "covered" : ""}}`;
        button.innerHTML = `<div class="task-title">#${{idx + 1}} ${{task.priority}} · ${{task.sample_name}}</div>` +
          `<div class="task-meta">${{task.role}}:${{task.name}} · rec F${{task.recommended_frame_index}}</div>` +
          `<div class="task-meta">${{isTaskAcceptedAbsent(task) ? "accepted absent" : (isTaskCovered(task) ? "covered" : "open")}} · zero=${{task.zero_frame_count}}</div>`;
        button.addEventListener("click", () => selectTask(idx));
        taskListEl.appendChild(button);
      }});
    }}
    function selectTask(idx) {{
      currentTaskIdx = Math.max(0, Math.min(TASKS.length - 1, idx));
      const task = TASKS[currentTaskIdx];
      currentFrameIndex = task.recommended_frame_index;
      if (!task.frames.some((frame) => Number(frame.frame_index) === Number(currentFrameIndex)) && task.frames.length) {{
        currentFrameIndex = task.frames[0].frame_index;
      }}
      renderAll();
    }}
    function selectNextUncovered() {{
      const start = currentTaskIdx + 1;
      for (let offset = 0; offset < TASKS.length; offset++) {{
        const idx = (start + offset) % TASKS.length;
        if (!isTaskCovered(TASKS[idx])) {{
          selectTask(idx);
          return;
        }}
      }}
      canvasStatusEl.textContent = "All tasks are covered.";
    }}
    function currentTask() {{
      return TASKS[currentTaskIdx];
    }}
    function currentFrame() {{
      const task = currentTask();
      return task.frames.find((frame) => Number(frame.frame_index) === Number(currentFrameIndex)) || task.frames[0];
    }}
    function renderCurrentTask() {{
      const task = currentTask();
      taskTitleEl.textContent = `#${{currentTaskIdx + 1}} ${{task.priority.toUpperCase()}} · ${{task.sample_name}} · ${{task.role}}:${{task.name}}`;
      taskTextEl.textContent = `Task: ${{task.task_text}}`;
      taskReasonEl.textContent = `Reason: ${{task.reason}}`;
      taskActionEl.textContent = `Suggested: ${{task.suggested_action}}`;
      frameButtonsEl.innerHTML = "";
      task.frames.forEach((frame) => {{
        const button = document.createElement("button");
        button.type = "button";
        button.className = `frame-button ${{Number(frame.frame_index) === Number(task.recommended_frame_index) ? "recommended" : ""}} ${{Number(frame.frame_index) === Number(currentFrameIndex) ? "active" : ""}}`;
        button.textContent = `F${{frame.frame_index}}${{Number(frame.frame_index) === Number(task.recommended_frame_index) ? " recommended" : ""}}`;
        button.addEventListener("click", () => {{
          currentFrameIndex = frame.frame_index;
          renderAll();
        }});
        frameButtonsEl.appendChild(button);
      }});
      hintEl.textContent = modeEl.value === "box"
        ? "Box mode: drag a tight box around the selected target on the selected frame."
        : "Point mode: click a positive or negative point on the selected target.";
    }}
    function drawCanvas(previewBox = null) {{
      if (!image.complete || !image.naturalWidth) return;
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(image, 0, 0);
      const task = currentTask();
      const prompts = framePrompts(task, currentFrameIndex);
      prompts.forEach((prompt) => {{
        if (prompt.accepted_absent) return;
        (prompt.boxes_xywh_abs || []).forEach((box, idx) => {{
          const label = (prompt.box_labels || [1])[idx] ?? 1;
          ctx.strokeStyle = Number(label) > 0 ? "#19a974" : "#d64545";
          ctx.lineWidth = 3;
          ctx.strokeRect(box[0], box[1], box[2], box[3]);
        }});
        (prompt.points_abs || []).forEach((point, idx) => {{
          const label = (prompt.point_labels || [1])[idx] ?? 1;
          ctx.fillStyle = Number(label) > 0 ? "#19a974" : "#d64545";
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(point[0], point[1], 6, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
        }});
      }});
      if (previewBox) {{
        ctx.strokeStyle = "#276ef1";
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.strokeRect(previewBox.x, previewBox.y, previewBox.w, previewBox.h);
        ctx.setLineDash([]);
      }}
      const absent = isTaskAcceptedAbsent(task) ? " · accepted absent" : "";
      canvasStatusEl.textContent = `Frame F${{currentFrameIndex}} · current-frame prompts: ${{prompts.length}} · task prompts: ${{taskPrompts(task).length}}${{absent}}`;
    }}
    function loadFrame() {{
      const frame = currentFrame();
      if (!frame) {{
        canvasStatusEl.textContent = "No frame available for this task.";
        return;
      }}
      image = new Image();
      image.onload = () => drawCanvas();
      image.src = frame.src;
    }}
    function renderAll() {{
      renderTaskList();
      renderCurrentTask();
      loadFrame();
      jsonBoxEl.value = JSON.stringify(corrections, null, 2);
    }}
    function canvasPoint(event) {{
      const rect = canvas.getBoundingClientRect();
      const rawX = Math.round((event.clientX - rect.left) * canvas.width / rect.width);
      const rawY = Math.round((event.clientY - rect.top) * canvas.height / rect.height);
      return {{
        x: Math.max(0, Math.min(canvas.width, rawX)),
        y: Math.max(0, Math.min(canvas.height, rawY))
      }};
    }}
    canvas.addEventListener("click", (event) => {{
      if (modeEl.value !== "point") return;
      const task = currentTask();
      const point = canvasPoint(event);
      const label = Number(labelEl.value || 1);
      addPrompt(task, {{
        name: task.name,
        role: task.role,
        frame_index: Number(currentFrameIndex),
        points_abs: [[point.x, point.y]],
        point_labels: [label],
        boxes_xywh_abs: [],
        box_labels: []
      }});
    }});
    canvas.addEventListener("mousedown", (event) => {{
      if (modeEl.value !== "box") return;
      dragStart = canvasPoint(event);
      dragCurrent = dragStart;
      event.preventDefault();
    }});
    canvas.addEventListener("mousemove", (event) => {{
      if (!dragStart || modeEl.value !== "box") return;
      dragCurrent = canvasPoint(event);
      const x = Math.min(dragStart.x, dragCurrent.x);
      const y = Math.min(dragStart.y, dragCurrent.y);
      const w = Math.abs(dragCurrent.x - dragStart.x);
      const h = Math.abs(dragCurrent.y - dragStart.y);
      drawCanvas({{ x, y, w, h }});
    }});
    window.addEventListener("mouseup", (event) => {{
      if (!dragStart || modeEl.value !== "box") return;
      const end = canvasPoint(event);
      const x = Math.min(dragStart.x, end.x);
      const y = Math.min(dragStart.y, end.y);
      const w = Math.abs(end.x - dragStart.x);
      const h = Math.abs(end.y - dragStart.y);
      dragStart = null;
      dragCurrent = null;
      if (w < 2 || h < 2) {{
        drawCanvas();
        return;
      }}
      const task = currentTask();
      const label = Number(labelEl.value || 1);
      addPrompt(task, {{
        name: task.name,
        role: task.role,
        frame_index: Number(currentFrameIndex),
        points_abs: [],
        point_labels: [],
        boxes_xywh_abs: [[x, y, w, h]],
        box_labels: [label]
      }});
    }});
    document.getElementById("prev-task").addEventListener("click", () => selectTask(currentTaskIdx - 1));
    document.getElementById("next-task").addEventListener("click", () => selectTask(currentTaskIdx + 1));
    document.getElementById("next-open").addEventListener("click", selectNextUncovered);
    document.getElementById("mark-absent").addEventListener("click", () => {{
      const task = currentTask();
      addPrompt(task, {{
        name: task.name,
        role: task.role,
        frame_index: Number(currentFrameIndex),
        accepted_absent: true,
        absent_reason: "Human reviewed the available frames for this view-role and marked the target as not visible.",
        points_abs: [],
        point_labels: [],
        boxes_xywh_abs: [],
        box_labels: []
      }});
    }});
    document.getElementById("clear-task").addEventListener("click", removeCurrentTaskPrompts);
    document.getElementById("undo").addEventListener("click", () => {{
      if (!history.length) return;
      corrections = JSON.parse(history.pop());
      persist();
      renderAll();
    }});
    document.getElementById("clear-all").addEventListener("click", () => {{
      saveHistory();
      corrections = {{
        schema_version: "maskwam_corrections_v1",
        notes: "Generated from MaskWAM task queue annotator. Coordinates are absolute pixels on the composed 224x448 frame.",
        samples: {{}}
      }};
      persist();
      renderAll();
    }});
    document.getElementById("load-json").addEventListener("click", () => {{
      try {{
        const parsed = JSON.parse(jsonBoxEl.value);
        if (!parsed.samples || typeof parsed.samples !== "object") throw new Error("missing samples object");
        saveHistory();
        corrections = parsed;
        persist();
        renderAll();
      }} catch (error) {{
        saveStatusEl.textContent = `load failed: ${{error.message}}`;
      }}
    }});
    document.getElementById("download").addEventListener("click", () => {{
      const blob = new Blob([JSON.stringify(corrections, null, 2)], {{ type: "application/json" }});
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = "corrections_from_html.json";
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(link.href);
    }});
    modeEl.addEventListener("change", renderCurrentTask);
    restore();
    selectTask(0);
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    review_dir = Path(args.review_dir)
    tasks_path = Path(args.tasks) if args.tasks else review_dir / "correction_tasks.json"
    output_dir = Path(args.output_dir) if args.output_dir else review_dir / "task_queue_annotator"
    tasks_payload = load_json(tasks_path)
    tasks = build_tasks(tasks_payload, review_dir, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_html(output_dir / "index.html", tasks, review_dir)
    with open(output_dir / "task_queue_manifest.json", "w", encoding="utf-8") as file:
        json.dump({"tasks": tasks}, file, indent=2)
    print(f"tasks={len(tasks)}")
    print(f"index={output_dir / 'index.html'}")
    print(f"manifest={output_dir / 'task_queue_manifest.json'}")


if __name__ == "__main__":
    main()
