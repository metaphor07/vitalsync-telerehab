"""
VitalSync · TeleRehab Flask API Server
──────────────────────────────────────
Bridges the VitalSync HTML dashboard with the MediaPipe-based
exercise-analysis pipeline in app.py.

Endpoints
---------
GET  /               ► serves vitalsync.html from the parent folder
POST /api/analyze     ► accepts a video upload, runs ML analysis,
                        returns full JSON results
GET  /api/status      ► basic health-check
"""

import os
import uuid
import time
import json
import traceback
from dataclasses import asdict
from flask import Flask, request, jsonify, send_from_directory

# ── import the ML pipeline from the same package ──────────────
from app import (
    analyze_video,
    INPUT_DIR,
    OUTPUT_DIR,
    SUPPORTED_EXTENSIONS,
    USER_WEIGHT_KG,
    USER_HEIGHT_M,
    USER_AGE,
    GOAL_MODE,
    EXERCISE_BODY_PARTS,
)
import app as ml_app          # so we can monkey-patch globals when needed

# ── Flask setup ───────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR   = os.path.dirname(BASE_DIR)          # Dashboardd/

flask_app = Flask(__name__, static_folder=None)
flask_app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024   # 500 MB max upload


# ──────────────────────────────────────────────────────────────
# SERVE THE DASHBOARD HTML
# ──────────────────────────────────────────────────────────────

@flask_app.route("/")
def index():
    return send_from_directory(PARENT_DIR, "vitalsync.html")


@flask_app.route("/<path:filename>")
def static_files(filename):
    """Serve any static assets (json, css, js) from the parent dir."""
    return send_from_directory(PARENT_DIR, filename)


# ──────────────────────────────────────────────────────────────
# HEALTH CHECK
# ──────────────────────────────────────────────────────────────

@flask_app.route("/api/status")
def status():
    return jsonify({"status": "ok", "engine": "TeleRehab MediaPipe"})


# ──────────────────────────────────────────────────────────────
# VIDEO ANALYSIS ENDPOINT
# ──────────────────────────────────────────────────────────────

@flask_app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Accept a video file upload, run the full ML pipeline, and return
    structured JSON that the VitalSync dashboard can consume.

    Optional form fields
    --------------------
    weight_kg   : float   (default 70)
    height_m    : float   (default 1.70)
    age         : int     (default 25)
    goal_mode   : str     (default "fitness")
    user_name   : str     (default "Participant 1")
    """
    # ── validate upload ───────────────────────────────────────
    if "video" not in request.files:
        return jsonify({"error": "No video file in request"}), 400

    video = request.files["video"]
    if video.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    ext = os.path.splitext(video.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return jsonify({
            "error": f"Unsupported format '{ext}'",
            "supported": list(SUPPORTED_EXTENSIONS),
        }), 400

    # ── save to disk ──────────────────────────────────────────
    unique_name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}{ext}"
    save_path   = os.path.join(INPUT_DIR, unique_name)
    video.save(save_path)

    # ── optionally override user params ───────────────────────
    try:
        ml_app.USER_WEIGHT_KG = float(request.form.get("weight_kg", USER_WEIGHT_KG))
    except (ValueError, TypeError):
        pass
    try:
        ml_app.USER_HEIGHT_M = float(request.form.get("height_m", USER_HEIGHT_M))
    except (ValueError, TypeError):
        pass
    try:
        ml_app.USER_AGE = int(request.form.get("age", USER_AGE))
    except (ValueError, TypeError):
        pass
    ml_app.USER_NAME = request.form.get("user_name", ml_app.USER_NAME)
    ml_app.GOAL_MODE = request.form.get("goal_mode", ml_app.GOAL_MODE)

    # ── run the ML pipeline ───────────────────────────────────
    try:
        start_ts = time.time()
        summary  = analyze_video(save_path)
        elapsed  = round(time.time() - start_ts, 2)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({
            "error": "ML processing failed",
            "detail": str(exc),
        }), 500

    # ── load the per-rep JSON if it was written ───────────────
    base_name   = os.path.splitext(unique_name)[0]
    session_dir = os.path.join(OUTPUT_DIR, base_name)
    reps_path   = os.path.join(session_dir, f"{base_name}_reps.json")

    rep_records = []
    if os.path.isfile(reps_path):
        with open(reps_path, "r", encoding="utf-8") as f:
            reps_data = json.load(f)
            rep_records = reps_data.get("reps", [])

    # ── build the response payload ────────────────────────────
    summary_dict = asdict(summary)

    payload = {
        "success": True,
        "processing_time_sec": elapsed,
        "summary": summary_dict,
        "reps": rep_records,

        # ── flattened convenience fields the UI consumes directly ──
        "workoutType":        summary.detected_exercise,
        "exercises":          [summary.detected_exercise],
        "bodyParts":          _body_parts_for(summary.detected_exercise),
        "durationEstimate":   round(summary.duration_sec / 60, 1),
        "caloriesEstimate":   round(summary.estimated_calories, 1),
        "intensityEstimate":  _intensity_to_num(summary.intensity),
        "formFeedback":       summary.recommendation,
        "summary_text":       _build_summary_text(summary),
        "formScoreAvg":       summary.avg_form_score,
        "totalReps":          summary.total_reps,
        "correctReps":        summary.correct_reps,
        "incorrectReps":      summary.incorrect_reps,
        "poseCoverage":       summary.pose_coverage_percent,
        "consistencyScore":   summary.consistency_score,
        "riskFlag":           summary.risk_flag,
        "overallRating":      summary.overall_rating,
        "topIssues":          summary.top_issues,
        "romScore":           summary.avg_rom_score,
        "tempoScore":         summary.avg_tempo_score,
        "stabilityScore":     summary.avg_stability_score,
        "symmetryGap":        summary.avg_symmetry_gap,
        "torsoLean":          summary.avg_torso_lean,
        "hydrationNote":      summary.hydration_note,
        "workoutLoadScore":   summary.workout_load_score,
        "bmi":                summary.bmi,
        "bmiCategory":        summary.bmi_category,
        "bmiNote":            summary.bmi_note,
    }

    return jsonify(payload)


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def _body_parts_for(exercise: str):
    return EXERCISE_BODY_PARTS.get(exercise, ["Full Body"])


def _intensity_to_num(intensity_label: str) -> int:
    mapping = {"Low": 3, "Moderate": 5, "High": 7, "Very High": 9}
    return mapping.get(intensity_label, 5)


def _build_summary_text(s) -> str:
    lines = [
        f"{s.detected_exercise} session — {s.total_reps} reps detected "
        f"({s.correct_reps} correct, {s.incorrect_reps} incorrect).",
        f"Average form score: {s.avg_form_score:.0f}/100 "
        f"({s.overall_rating}).",
    ]
    if s.top_issues:
        lines.append("Key observations: " + "; ".join(s.top_issues) + ".")
    return " ".join(lines)


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  VitalSync - TeleRehab API Server")
    print(f"  Serving dashboard from: {PARENT_DIR}")
    print(f"  Videos saved to:        {INPUT_DIR}")
    print(f"  Output written to:      {OUTPUT_DIR}")
    print("=" * 60)
    print("  -> Open http://localhost:5000 in your browser")
    print("=" * 60)
    flask_app.run(host="0.0.0.0", port=5000, debug=True)
