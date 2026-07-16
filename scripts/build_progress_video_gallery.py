#!/usr/bin/env python3
"""Build a static gallery for per-episode, per-model Progress videos."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_dir", type=Path, required=True)
    parser.add_argument("--benchmark_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--expected_episodes", type=int, default=100)
    return parser.parse_args()


CSS = """
:root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
body { margin: 0; color: #17202a; background: #f5f6f8; }
header { position: sticky; top: 0; z-index: 5; padding: 14px 22px; background: #fff; border-bottom: 1px solid #d8dde5; }
h1 { margin: 0; font-size: 20px; letter-spacing: 0; }
.sub { margin-top: 5px; color: #5b6472; font-size: 13px; }
main { padding: 18px 22px 36px; }
.episode-list { width: 100%; border-collapse: collapse; background: #fff; }
.episode-list th, .episode-list td { padding: 10px 12px; border-bottom: 1px solid #e2e6ec; text-align: left; }
.episode-list th { background: #eef1f5; font-size: 12px; text-transform: uppercase; }
a { color: #135f9b; text-decoration: none; }
a:hover { text-decoration: underline; }
.model-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(430px, 1fr)); gap: 12px; }
.model-item { background: #fff; border: 1px solid #d8dde5; border-radius: 6px; overflow: hidden; }
.model-head { display: flex; justify-content: space-between; gap: 12px; padding: 9px 11px; border-bottom: 1px solid #e2e6ec; }
.model-name { font-size: 16px; font-weight: 700; }
.metrics { color: #4f5967; font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; }
video { display: block; width: 100%; aspect-ratio: 16 / 9; background: #111; }
.nav { display: flex; gap: 16px; margin-bottom: 14px; align-items: center; }
.task { margin: 5px 0 14px; font-size: 15px; }
.alert { color: #a12727; font-weight: 700; }
"""


def relative_url(path: Path, page_dir: Path) -> str:
    return Path(path).resolve().relative_to(page_dir.resolve().parent.parent).as_posix()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    payload = json.loads(args.benchmark_json.read_text(encoding="utf-8"))
    episode_paths = [str(value) for value in payload["sample_path_list"]]
    task_texts = [str(value) for value in payload["sample_path_task"]]
    if len(episode_paths) != args.expected_episodes or len(task_texts) != len(
        episode_paths
    ):
        raise ValueError(
            f"Expected {args.expected_episodes} benchmark episodes, got {len(episode_paths)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    page_dir = args.output_dir / "episodes"
    page_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    episode_rows: List[Dict[str, Any]] = []

    for episode_index, (relative_path, task_text) in enumerate(
        zip(episode_paths, task_texts)
    ):
        episode_file = f"episode_{episode_index:03d}"
        model_sections: List[str] = []
        catastrophic_models: List[str] = []
        for model in args.models:
            video = args.video_dir / "videos" / model / f"{episode_file}.mp4"
            poster = args.video_dir / "posters" / model / f"{episode_file}_worst.png"
            event_path = args.video_dir / "events" / model / f"{episode_file}.json"
            if not video.is_file() or not poster.is_file() or not event_path.is_file():
                raise FileNotFoundError(
                    f"Incomplete visual assets for {model}/{episode_file}: "
                    f"video={video.is_file()} poster={poster.is_file()} event={event_path.is_file()}"
                )
            event = json.loads(event_path.read_text(encoding="utf-8"))
            catastrophic = bool(event.get("catastrophic", False))
            if catastrophic:
                catastrophic_models.append(model)
            rows.append(
                {
                    "episode": episode_index,
                    "episode_name": event.get("episode_name", relative_path),
                    "task": task_text,
                    "model": model,
                    "mae": event["mae"],
                    "rmse": event["rmse"],
                    "max_absolute_error": event["max_absolute_error"],
                    "max_absolute_jump": event["max_absolute_jump"],
                    "max_absolute_slot_jump": event["max_absolute_slot_jump"],
                    "backward_step_rate": event["backward_step_rate"],
                    "end_error": event["end_error"],
                    "catastrophic": catastrophic,
                    "video": str(video),
                    "poster": str(poster),
                }
            )
            video_rel = Path("../../") / video.relative_to(args.video_dir)
            poster_rel = Path("../../") / poster.relative_to(args.video_dir)
            alert = '<span class="alert">catastrophic</span>' if catastrophic else ""
            model_sections.append(f"""
                <section class="model-item">
                  <div class="model-head">
                    <span class="model-name">{html.escape(model)}</span>
                    <span class="metrics">MAE {float(event['mae']):.4f} | max error {float(event['max_absolute_error']):.3f} | max jump {float(event['max_absolute_jump']):.3f} | slot jump {int(event['max_absolute_slot_jump'])} {alert}</span>
                  </div>
                  <video controls preload="metadata" poster="{poster_rel.as_posix()}">
                    <source src="{video_rel.as_posix()}" type="video/mp4">
                  </video>
                </section>
                """)

        previous_link = (
            f"episode_{episode_index - 1:03d}.html"
            if episode_index
            else "../index.html"
        )
        next_link = (
            f"episode_{episode_index + 1:03d}.html"
            if episode_index + 1 < len(episode_paths)
            else "../index.html"
        )
        page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Progress episode {episode_index:03d}</title><style>{CSS}</style></head>
<body><header><h1>Episode {episode_index:03d}</h1><div class="sub">{html.escape(relative_path)}</div></header>
<main><nav class="nav"><a href="../index.html">All episodes</a><a href="{previous_link}">Previous</a><a href="{next_link}">Next</a></nav>
<p class="task">{html.escape(task_text)}</p><div class="model-grid">{''.join(model_sections)}</div></main></body></html>"""
        (page_dir / f"{episode_file}.html").write_text(page, encoding="utf-8")
        episode_rows.append(
            {
                "episode": episode_index,
                "task": task_text,
                "catastrophic_models": ", ".join(catastrophic_models) or "none",
            }
        )

    table_rows = "".join(
        f"<tr><td><a href='episodes/episode_{int(row['episode']):03d}.html'>{int(row['episode']):03d}</a></td><td>{html.escape(str(row['task']))}</td><td>{html.escape(str(row['catastrophic_models']))}</td></tr>"
        for row in episode_rows
    )
    index = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Progress full visual review</title><style>{CSS}</style></head>
<body><header><h1>Progress Full Visual Review</h1><div class="sub">{len(args.models)} models x {len(episode_rows)} fixed Robo-Dopamine Bench episodes</div></header>
<main><table class="episode-list"><thead><tr><th>Episode</th><th>Task</th><th>Catastrophic models</th></tr></thead><tbody>{table_rows}</tbody></table></main></body></html>"""
    (args.output_dir / "index.html").write_text(index, encoding="utf-8")
    write_csv(args.output_dir / "visual_asset_manifest.csv", rows)
    (args.output_dir / "visual_asset_manifest.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(
        f"Gallery complete: {len(rows)} model-episode videos, "
        f"{len(episode_rows)} pages under {args.output_dir}"
    )


if __name__ == "__main__":
    main()
