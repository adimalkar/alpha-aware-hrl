"""
Alpha-Aware HRL REST API Server.
Bridges Python reinforcement learning and event model backends with the React dashboard.
"""

import os
import sys
import time
import json
import threading
from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.streaming.event_buffer import EventRingBuffer
from src.streaming.feed_adapter import SyntheticMarketFeed
from src.streaming.inference_loop import StreamingInferenceEngine
from src.utils.event_pipeline import EVENT_TYPES

app = Flask(__name__)
RESULTS_DIR = Path(__file__).resolve().parent.parent / "experiments"
CORS(app)

# Initialize Live Event Engine
event_buffer = EventRingBuffer(capacity=2048, feature_dim=144)
feed = SyntheticMarketFeed(event_buffer, base_rate=25.0)
feed.start()

engine = StreamingInferenceEngine(event_buffer=event_buffer)
engine.start()

training_process = {
    "is_running": False,
    "current_step": 0,
    "total_steps": 1000000,
    "latest_loss": 0.0,
    "latest_reward": 0.0,
}


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "service": "Alpha-Aware HRL API",
        "timestamp": time.time(),
    })


@app.route("/api/events/live", methods=["GET"])
def get_live_events():
    """Returns current real-time streaming telemetry and intensity rates."""
    telemetry = engine.get_telemetry()
    return jsonify(telemetry)


@app.route("/api/events/recent", methods=["GET"])
def get_recent_events():
    """Returns the last 50 events in the ring buffer."""
    window = event_buffer.get_latest_window(window_size=50)
    events_list = []
    
    for i in range(len(window["event_types"])):
        etype = int(window["event_types"][i])
        events_list.append({
            "index": i,
            "type_id": etype,
            "type_name": EVENT_TYPES.get(etype, "unknown"),
            "timestamp": float(window["timestamps"][i]),
            "dt": float(window["inter_arrival_times"][i]),
        })
    
    return jsonify({
        "events": events_list,
        "event_rate": event_buffer.get_event_rate(),
        "total_buffered": event_buffer.count,
    })


def _read_json(path):
    """Read a results artefact, or None when the experiment has not been run."""
    f = Path(path)
    if not f.is_file():
        return None
    try:
        with open(f) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _no_data(what, how):
    """
    404 with instructions instead of invented numbers.

    Every figure previously served from these two endpoints was a literal:
    sharpe 2.14, maxDrawdown 6.8, winRate 61.4, totalTrades 5832, plus six
    ablation rows for experiments that were never run at all. The dashboard
    presented them as measurements.
    """
    return jsonify({
        "error": "no_data",
        "detail": f"{what} has not been produced by any run in this workspace.",
        "how_to_generate": how,
    }), 404


@app.route("/api/metrics/summary", methods=["GET"])
def get_summary_metrics():
    results = _read_json(RESULTS_DIR / "rl_run" / "evaluation_results.json")
    if results is None:
        return _no_data(
            "Evaluation summary",
            "python scripts/train_rl_agent.py --symbol BTC/USD --encoder lem",
        )
    return jsonify({
        "source": "experiments/rl_run/evaluation_results.json",
        "provenance": results.get("provenance"),
        "sharpePerStep": results.get("sharpe_per_step"),
        "annualised": results.get("annualised", False),
        "totalReturnPct": results.get("total_return_pct"),
        "maxDrawdownPct": results.get("max_drawdown_pct"),
        "winRatePct": results.get("win_rate_pct"),
        "var95": results.get("var_95"),
        "cvar95": results.get("cvar_95"),
        "evalSteps": results.get("eval_steps"),
        "positionChanges": results.get("n_position_changes"),
        "warning": results.get("WARNING"),
    })


@app.route("/api/baselines", methods=["GET"])
def get_baselines():
    ablations = _read_json(RESULTS_DIR / "ablations" / "ablation_results.json")
    baselines = _read_json(RESULTS_DIR / "baselines" / "baseline_metrics.json")

    if ablations is None and baselines is None:
        return _no_data(
            "Comparative results",
            "python scripts/run_ablations.py --symbol BTC/USD",
        )

    payload = {"strategies": [], "ablations": []}
    if baselines is not None:
        payload["strategies"] = baselines
        payload["strategies_source"] = "experiments/baselines/baseline_metrics.json"
    if ablations is not None:
        payload["ablations"] = ablations.get("summary", [])
        payload["ablations_per_run"] = ablations.get("per_run", [])
        payload["ablations_provenance"] = ablations.get("provenance")
        payload["ablations_source"] = "experiments/ablations/ablation_results.json"
    return jsonify(payload)


@app.route("/api/training/launch", methods=["POST"])
def launch_training():
    data = request.json or {}
    timesteps = data.get("timesteps", 100000)
    training_process["is_running"] = True
    training_process["total_steps"] = timesteps
    training_process["current_step"] = 0
    return jsonify({"status": "started", "timesteps": timesteps})


@app.route("/api/training/status", methods=["GET"])
def training_status():
    return jsonify(training_process)


if __name__ == "__main__":
    print("[API Server] Starting Alpha-Aware HRL REST & Telemetry Server on http://localhost:8000")
    app.run(host="0.0.0.0", port=8000, debug=False, use_reloader=False)
