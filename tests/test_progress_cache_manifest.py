import json

from scripts.cache_vgm_progress_latents import merge_manifests


def test_merge_manifests_drops_entries_not_emitted_by_current_run(tmp_path):
    stale = {
        "schema_version": 2,
        "episode_name": "stale/episode",
        "num_progress_bins": 53,
        "num_queries": 10,
        "split": "train",
        "cache_file": "episodes/stale.pt",
    }
    current = {
        "schema_version": 2,
        "episode_name": "current/episode",
        "num_progress_bins": 53,
        "num_queries": 12,
        "split": "val",
        "cache_file": "episodes/current.pt",
    }
    (tmp_path / "manifest.jsonl").write_text(json.dumps(stale) + "\n")
    (tmp_path / "manifest.rank_000.jsonl").write_text(json.dumps(current) + "\n")

    merge_manifests(tmp_path)

    entries = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert entries == [current]
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["episodes"] == 1
    assert summary["train_episodes"] == 0
    assert summary["val_episodes"] == 1
