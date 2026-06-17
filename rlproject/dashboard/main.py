"""Lightweight training dashboard backend."""

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[2]
LOG_ROOTS = [ROOT / "src" / "logs", ROOT / "logs"]
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="League Training Dashboard")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def iter_run_dirs():
    for root in LOG_ROOTS:
        if not root.exists():
            continue
        for path in root.iterdir():
            if path.is_dir() and (path.name.startswith("dual_iterative_") or path.name.startswith("league_") or path.name.startswith("smoke_")):
                yield path


def run_info(path):
    latest = path / "training_status_latest.json"
    mtime = latest.stat().st_mtime if latest.exists() else path.stat().st_mtime
    return {
        "id": path.name,
        "path": str(path),
        "updated_at": mtime,
        "has_status": latest.exists(),
    }


def find_run(run_id):
    for path in iter_run_dirs():
        if path.name == run_id:
            return path
    raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")


def read_json(path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path, limit=200):
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    items = []
    for line in lines[-limit:]:
        if not line.strip():
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return items


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/runs")
def runs():
    result = sorted([run_info(p) for p in iter_run_dirs()], key=lambda x: x["updated_at"], reverse=True)
    return {"runs": result}


@app.get("/api/runs/latest/status")
def latest_status():
    all_runs = sorted([run_info(p) for p in iter_run_dirs()], key=lambda x: x["updated_at"], reverse=True)
    if not all_runs:
        return {"run": None, "status": None}
    run = find_run(all_runs[0]["id"])
    return {"run": all_runs[0], "status": read_json(run / "training_status_latest.json", {})}


@app.get("/api/runs/{run_id}/status")
def status(run_id: str):
    run = find_run(run_id)
    return {"run": run_info(run), "status": read_json(run / "training_status_latest.json", {})}


@app.get("/api/runs/{run_id}/events")
def events(run_id: str, limit: int = 1000):
    run = find_run(run_id)
    return {"events": read_jsonl(run / "training_status.jsonl", limit=limit)}


def read_recent_episodes(run, limit=200):
    for name in ["recent_episodes_latest.json", "recent_episodes.json"]:
        json_path = run / name
        if json_path.exists():
            data = read_json(json_path, []) or []
            return data[-limit:]
    return read_jsonl(run / "recent_episodes.jsonl", limit=limit)


@app.get("/api/runs/{run_id}/episodes")
def episodes(run_id: str, limit: int = 200):
    run = find_run(run_id)
    return {"episodes": read_recent_episodes(run, limit=limit)}


@app.get("/api/runs/{run_id}/convergence")
def convergence(run_id: str, limit: int = 100):
    run = find_run(run_id)
    return {"convergence": read_jsonl(run / "convergence_history.jsonl", limit=limit)}


@app.get("/api/runs/{run_id}/pfsp")
def pfsp(run_id: str, limit: int = 200):
    run = find_run(run_id)
    episodes = read_recent_episodes(run, limit=limit)
    pfsp_points = []
    for ep in episodes:
        probs = ep.get("pfsp_probs") or []
        if probs:
            pfsp_points.append({
                "timestep": ep.get("global_timestep"),
                "iteration": ep.get("iteration"),
                "selected": ep.get("selected_opponent_index"),
                "probs": probs,
            })
    return {"pfsp": pfsp_points}
