import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import csv
import json
import math
import statistics
import traceback
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

import mediapipe as mp
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# ============================================================
# PATHS / CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "videos")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MAX_VIDEO_SIZE_MB = 100
SUPPORTED_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".mpeg", ".mpg")

USER_NAME = "Participant 1"
USER_WEIGHT_KG = 70
USER_HEIGHT_M = 1.70
USER_AGE = 25
GOAL_MODE = "fitness"  # rehab / beginner / fitness

MIN_LANDMARK_VISIBILITY = 0.40
MIN_POSE_FRAMES_FOR_ANALYSIS = 30

# More forgiving thresholds for easier rep detection
SQUAT_DOWN_THRESHOLD = 125
SQUAT_UP_THRESHOLD = 145

PUSHUP_DOWN_THRESHOLD = 110
PUSHUP_UP_THRESHOLD = 145

LUNGE_DOWN_THRESHOLD = 125
LUNGE_UP_THRESHOLD = 145

CURL_DOWN_THRESHOLD = 70
CURL_UP_THRESHOLD = 130

PRESS_DOWN_THRESHOLD = 120
PRESS_UP_THRESHOLD = 155

DEADLIFT_DOWN_THRESHOLD = 110
DEADLIFT_UP_THRESHOLD = 150

SYMMETRY_WARNING_DEG = 15
TORSO_LEAN_WARNING_DEG = 40

EXERCISE_MET = {
    "Squat": 5.0,
    "Push-Up": 8.0,
    "Lunge": 6.0,
    "Bicep Curl": 3.5,
    "Shoulder Press": 5.0,
    "Deadlift": 6.0,
    "Unknown": 4.0,
}

EXERCISE_BODY_PARTS = {
    "Squat": ["Quadriceps", "Hamstrings", "Glutes", "Core"],
    "Push-Up": ["Chest", "Triceps", "Shoulders", "Core"],
    "Lunge": ["Quadriceps", "Glutes", "Hamstrings", "Calves"],
    "Bicep Curl": ["Biceps", "Forearms", "Shoulders"],
    "Shoulder Press": ["Shoulders", "Triceps", "Upper Back"],
    "Deadlift": ["Hamstrings", "Glutes", "Lower Back", "Core"],
    "Unknown": ["Full Body"],
}

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── MediaPipe Tasks API (0.10.33+ / Python 3.13) ──────────
MODEL_PATH = os.path.join(BASE_DIR, "models", "pose_landmarker_full.task")

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Pose connection pairs for drawing (same 33-landmark topology)
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),
    (12,14),(14,16),(16,18),(16,20),(16,22),
    (11,23),(12,24),(23,24),(23,25),(24,26),
    (25,27),(26,28),(27,29),(28,30),(29,31),(30,32),
    (27,31),(28,32)
]


def draw_pose_landmarks(image, landmarks, w, h):
    """Draw pose landmarks and connections on the frame using cv2."""
    points = []
    for lm in landmarks:
        px = int(lm.x * w)
        py = int(lm.y * h)
        vis = getattr(lm, 'visibility', 1.0)
        points.append((px, py, vis))

    # Draw connections
    for (a, b) in POSE_CONNECTIONS:
        if a < len(points) and b < len(points):
            if points[a][2] > 0.3 and points[b][2] > 0.3:
                cv2.line(image, (points[a][0], points[a][1]),
                         (points[b][0], points[b][1]),
                         (0, 255, 128), 2, cv2.LINE_AA)

    # Draw keypoints
    for (px, py, vis) in points:
        if vis > 0.3:
            cv2.circle(image, (px, py), 4, (0, 200, 255), -1, cv2.LINE_AA)


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class RepRecord:
    rep_number: int
    exercise: str
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    depth_label: str
    form_score: float
    quality_label: str
    correctness: str
    avg_left_knee: float
    avg_right_knee: float
    avg_left_elbow: float
    avg_right_elbow: float
    avg_torso_lean: float
    symmetry_gap: float
    rom_score: float
    tempo_score: float
    stability_score: float
    warnings: str


@dataclass
class SessionSummary:
    video_name: str
    file_size_mb: float
    accepted: bool
    reason: str
    detected_exercise: str
    duration_sec: float
    frames_total: int
    pose_detected_frames: int
    pose_coverage_percent: float
    total_reps: int
    correct_reps: int
    incorrect_reps: int
    avg_form_score: float
    avg_rep_time_sec: float
    avg_symmetry_gap: float
    avg_torso_lean: float
    avg_rom_score: float
    avg_tempo_score: float
    avg_stability_score: float
    estimated_calories: float
    bmi: float
    bmi_category: str
    bmi_note: str
    consistency_score: float
    intensity: str
    workout_load_score: float
    risk_flag: str
    hydration_note: str
    overall_rating: str
    recommendation: str
    top_issues: List[str]


# ============================================================
# UTILS
# ============================================================

def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def avg(values: List[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def calculate_angle(a, b, c) -> float:
    ax, ay = a
    bx, by = b
    cx, cy = c

    abx, aby = ax - bx, ay - by
    cbx, cby = cx - bx, cy - by

    dot = abx * cbx + aby * cby
    mag1 = math.sqrt(abx ** 2 + aby ** 2)
    mag2 = math.sqrt(cbx ** 2 + cby ** 2)

    if mag1 == 0 or mag2 == 0:
        return 0.0

    cos_angle = dot / (mag1 * mag2)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def get_video_size_mb(video_path: str) -> float:
    return os.path.getsize(video_path) / (1024 * 1024)


def is_video_allowed(video_path: str) -> Tuple[bool, str]:
    size_mb = get_video_size_mb(video_path)
    if size_mb > MAX_VIDEO_SIZE_MB:
        return False, f"Rejected: file size {size_mb:.2f} MB exceeds limit of {MAX_VIDEO_SIZE_MB} MB"
    return True, "Accepted"


def quality_label(score: float) -> str:
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Average"
    return "Poor"


def estimate_calories(duration_seconds: float, weight_kg: float, met: float) -> float:
    hours = duration_seconds / 3600.0
    return met * weight_kg * hours


def calculate_bmi(weight_kg: float, height_m: float) -> float:
    if height_m <= 0:
        return 0.0
    return weight_kg / (height_m ** 2)


def bmi_category_4(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal weight"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def bmi_health_note(bmi: float) -> str:
    category = bmi_category_4(bmi)
    if category == "Underweight":
        return "Low body mass range. Focus on strength progression and nutrition support."
    if category == "Normal weight":
        return "Healthy BMI range. Continue balanced training and recovery."
    if category == "Overweight":
        return "Elevated BMI range. Focus on safe progression, consistency, and joint-friendly exercise."
    return "High BMI range. Use controlled movement, lower-impact progression, and monitor fatigue."


def session_intensity(avg_score: float, avg_rep_time: float, total_reps: int) -> str:
    load_factor = avg_score * 0.5 + total_reps * 2.0
    if avg_rep_time > 0:
        load_factor += 10 / avg_rep_time

    if load_factor < 40:
        return "Low"
    if load_factor < 80:
        return "Moderate"
    if load_factor < 120:
        return "High"
    return "Very High"


def calculate_consistency_score(rep_scores: List[float]) -> float:
    if not rep_scores:
        return 0.0
    if len(rep_scores) == 1:
        return 100.0

    std_dev = statistics.pstdev(rep_scores)
    consistency = 100 - (std_dev * 3)
    return round(clamp(consistency, 0, 100), 2)


def workout_load(total_reps: int, avg_score: float, duration_sec: float) -> float:
    minutes = duration_sec / 60.0 if duration_sec > 0 else 0.0
    load = (total_reps * max(avg_score, 1)) * max(minutes, 1)
    return round(load / 10.0, 2)


def hydration_reminder(duration_sec: float, intensity: str) -> str:
    minutes = duration_sec / 60.0
    if minutes < 10:
        return "Small water break is enough."
    if intensity in ["High", "Very High"]:
        return "Drink water after session and take a recovery break."
    return "Hydrate normally after session."


def readiness_risk_flag(avg_score: float, avg_symmetry_gap: float, avg_torso_lean: float, bmi: float) -> str:
    risk_points = 0

    if avg_score < 60:
        risk_points += 2
    elif avg_score < 75:
        risk_points += 1

    if avg_symmetry_gap > 18:
        risk_points += 2
    elif avg_symmetry_gap > 12:
        risk_points += 1

    if avg_torso_lean > 45:
        risk_points += 2
    elif avg_torso_lean > 35:
        risk_points += 1

    if bmi >= 30 or bmi < 18.5:
        risk_points += 1

    if risk_points <= 1:
        return "Low Risk"
    if risk_points <= 3:
        return "Moderate Risk"
    return "High Risk"


def get_goal_recommendation(goal_mode: str, avg_score: float, exercise: str) -> str:
    if goal_mode == "rehab":
        if avg_score < 60:
            return f"{exercise}: 1-2 sets x 5-8 controlled reps"
        return f"{exercise}: 2 sets x 8-10 controlled reps"

    if goal_mode == "beginner":
        if avg_score < 60:
            return f"{exercise}: 2 sets x 6-8 reps"
        return f"{exercise}: 2-3 sets x 8-12 reps"

    if avg_score < 60:
        return f"{exercise}: 2-3 sets x 8-10 reps"
    return f"{exercise}: 3 sets x 10-15 reps"


def get_landmark_xy(landmarks, idx) -> Tuple[float, float]:
    return landmarks[idx].x, landmarks[idx].y


def get_visibility(landmarks, idx) -> float:
    lm = landmarks[idx]
    return getattr(lm, "visibility", 1.0)


def landmarks_ok(landmarks, ids: List[int], threshold=MIN_LANDMARK_VISIBILITY) -> bool:
    return all(get_visibility(landmarks, i) >= threshold for i in ids)


def exercise_color(exercise: str) -> Tuple[int, int, int]:
    if exercise == "Squat":
        return (0, 255, 255)
    if exercise == "Push-Up":
        return (255, 255, 0)
    if exercise == "Lunge":
        return (255, 0, 255)
    return (255, 255, 255)


# ============================================================
# EXERCISE DETECTION
# ============================================================

def detect_exercise_type(sample_frames: List[Dict]) -> str:
    if not sample_frames:
        return "Unknown"

    squat_votes = 0
    lunge_votes = 0
    pushup_votes = 0
    curl_votes = 0
    press_votes = 0
    deadlift_votes = 0

    for f in sample_frames:
        lk = f["left_knee"]
        rk = f["right_knee"]
        le = f["left_elbow"]
        re = f["right_elbow"]
        torso = f["torso_lean"]
        hip_y = f["hip_y"]
        shoulder_y = f["shoulder_y"]
        hip_angle = f.get("hip_angle", 170)
        wrist_y = f.get("wrist_y", shoulder_y)

        knee_avg = (lk + rk) / 2.0
        elbow_avg = (le + re) / 2.0
        knee_gap = abs(lk - rk)
        body_horizontal = abs(shoulder_y - hip_y) < 0.10
        body_upright = not body_horizontal and torso < 30
        wrist_above_shoulder = wrist_y < shoulder_y  # y is inverted (top=0)

        # ── Push-Up: body horizontal + elbow flexion ──
        if body_horizontal and elbow_avg < 130:
            pushup_votes += 3
        elif body_horizontal and elbow_avg < 150:
            pushup_votes += 1

        # ── Squat: symmetric knee bend, upright torso ──
        if knee_avg < 145 and knee_gap < 14 and torso < 42 and not body_horizontal:
            squat_votes += 3
        elif knee_avg < 150 and knee_gap < 18 and not body_horizontal:
            squat_votes += 1

        # ── Lunge: asymmetric knee bend ──
        if min(lk, rk) < 130 and knee_gap > 20 and not body_horizontal:
            lunge_votes += 3
        elif knee_gap > 16 and not body_horizontal:
            lunge_votes += 1

        # ── Bicep Curl: upright body, knees straight, EITHER elbow flexing significantly ──
        min_elbow = min(le, re)
        if body_upright and knee_avg > 155 and min_elbow < 90:
            curl_votes += 4
        elif body_upright and knee_avg > 150 and min_elbow < 110:
            curl_votes += 3
        elif body_upright and knee_avg > 150 and elbow_avg < 130:
            curl_votes += 1

        # ── Shoulder Press: upright, elbows extend overhead ──
        if body_upright and wrist_above_shoulder and elbow_avg > 140:
            press_votes += 3
        elif body_upright and wrist_above_shoulder and elbow_avg > 120:
            press_votes += 2
        elif body_upright and elbow_avg > 130 and wrist_above_shoulder:
            press_votes += 1

        # ── Deadlift: hip hinge WITH actual knee bend (not just bending over) ──
        if hip_angle < 140 and 130 < knee_avg < 165 and torso > 30 and not body_horizontal:
            deadlift_votes += 3
        elif hip_angle < 150 and knee_avg < 165 and knee_avg > 140 and torso > 30 and not body_horizontal:
            deadlift_votes += 1

    scores = {
        "Squat": squat_votes,
        "Lunge": lunge_votes,
        "Push-Up": pushup_votes,
        "Bicep Curl": curl_votes,
        "Shoulder Press": press_votes,
        "Deadlift": deadlift_votes,
    }

    best_exercise = max(scores, key=scores.get)
    best_score = scores[best_exercise]

    if best_score < 4:
        return "Unknown"

    return best_exercise


def detect_exercise_with_majority(sample_frames: List[Dict], window_size: int = 12) -> str:
    if len(sample_frames) < window_size:
        return detect_exercise_type(sample_frames)

    votes = []
    step = max(1, window_size // 2)
    for start in range(0, len(sample_frames) - window_size + 1, step):
        window = sample_frames[start:start + window_size]
        votes.append(detect_exercise_type(window))

    filtered_votes = [v for v in votes if v != "Unknown"]
    if not filtered_votes:
        return "Unknown"

    counts = {}
    for vote in filtered_votes:
        counts[vote] = counts.get(vote, 0) + 1

    return max(counts, key=counts.get)


# ============================================================
# REP METRICS
# ============================================================

def classify_depth(exercise: str, avg_knee: float, avg_elbow: float, avg_hip_angle: float = 170) -> str:
    if exercise == "Push-Up":
        if avg_elbow > 125:
            return "Shallow"
        if 85 <= avg_elbow <= 125:
            return "Standard"
        return "Deep"

    if exercise == "Bicep Curl":
        if avg_elbow > 110:
            return "Shallow"
        if 50 <= avg_elbow <= 110:
            return "Standard"
        return "Deep"

    if exercise == "Shoulder Press":
        if avg_elbow < 100:
            return "Shallow"
        if 100 <= avg_elbow <= 155:
            return "Standard"
        return "Deep"

    if exercise == "Deadlift":
        if avg_hip_angle > 150:
            return "Shallow"
        if 100 <= avg_hip_angle <= 150:
            return "Standard"
        return "Deep"

    # Squat / Lunge
    if avg_knee > 135:
        return "Shallow"
    if 90 <= avg_knee <= 135:
        return "Standard"
    return "Deep"


def compute_rom_score(exercise: str, min_primary_angle: float) -> float:
    if exercise == "Push-Up":
        if min_primary_angle <= 85:
            return 95
        if min_primary_angle <= 105:
            return 80
        if min_primary_angle <= 125:
            return 60
        return 35

    if exercise == "Bicep Curl":
        if min_primary_angle <= 45:
            return 95
        if min_primary_angle <= 70:
            return 80
        if min_primary_angle <= 100:
            return 60
        return 35

    if exercise == "Shoulder Press":
        # Higher angle = more extension = better
        if min_primary_angle >= 160:
            return 95
        if min_primary_angle >= 140:
            return 80
        if min_primary_angle >= 110:
            return 60
        return 35

    if exercise == "Deadlift":
        # Hip angle: lower = deeper hinge = better ROM
        if min_primary_angle <= 100:
            return 95
        if min_primary_angle <= 120:
            return 80
        if min_primary_angle <= 140:
            return 60
        return 35

    # Squat / Lunge
    if min_primary_angle <= 90:
        return 95
    if min_primary_angle <= 110:
        return 80
    if min_primary_angle <= 130:
        return 60
    return 35


def compute_tempo_score(rep_duration: float) -> float:
    if 1.5 <= rep_duration <= 4.0:
        return 90
    if 1.0 <= rep_duration < 1.5 or 4.0 < rep_duration <= 5.5:
        return 70
    if 0.7 <= rep_duration < 1.0 or 5.5 < rep_duration <= 7.0:
        return 50
    return 30


def compute_stability_score(symmetry_gap: float, torso_lean: float) -> float:
    score = 100
    if symmetry_gap > 20:
        score -= 30
    elif symmetry_gap > 12:
        score -= 15

    if torso_lean > 50:
        score -= 25
    elif torso_lean > 40:
        score -= 12

    return clamp(score, 0, 100)


def compute_form_score(exercise: str, rom_score: float, tempo_score: float, stability_score: float) -> float:
    if exercise == "Push-Up":
        score = 0.45 * rom_score + 0.20 * tempo_score + 0.35 * stability_score
    else:
        score = 0.40 * rom_score + 0.20 * tempo_score + 0.40 * stability_score
    return round(clamp(score, 0, 100), 2)


def get_warnings(exercise: str, depth_label: str, symmetry_gap: float, torso_lean: float, score: float) -> List[str]:
    warnings = []

    if depth_label == "Shallow":
        warnings.append("Insufficient range of motion")
    if symmetry_gap > SYMMETRY_WARNING_DEG:
        warnings.append("Left-right asymmetry detected")
    if torso_lean > TORSO_LEAN_WARNING_DEG and exercise in ["Squat", "Lunge"]:
        warnings.append("Excessive torso lean")
    if score < 60:
        warnings.append("Low-quality repetition")

    return warnings


# ============================================================
# FILE WRITERS / PLOTS
# ============================================================

def write_csv(path: str, rows: List[Dict]):
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str, payload: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_line_plot(values: List[float], title: str, xlabel: str, ylabel: str, out_path: str):
    if not values:
        return
    plt.figure(figsize=(10, 4))
    plt.plot(values, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def generate_pdf_report(pdf_path: str, summary: SessionSummary, rep_records: List[RepRecord], charts: List[str]):
    c = canvas.Canvas(pdf_path, pagesize=A4)
    _, height = A4
    y = height - 40

    def line(text, font="Helvetica", size=10, gap=16):
        nonlocal y
        if y < 60:
            c.showPage()
            y = height - 40
        c.setFont(font, size)
        c.drawString(40, y, text)
        y -= gap

    line("Advanced Exercise Analysis Report", font="Helvetica-Bold", size=16, gap=24)
    line(f"User: {USER_NAME}")
    line(f"Video: {summary.video_name}")
    line(f"Detected Exercise: {summary.detected_exercise}")
    line(f"Duration: {summary.duration_sec:.2f} sec")
    line(f"File Size: {summary.file_size_mb:.2f} MB")
    line(f"Pose Coverage: {summary.pose_coverage_percent:.2f}%")
    line(f"Total Reps: {summary.total_reps}")
    line(f"Correct Reps: {summary.correct_reps}")
    line(f"Incorrect Reps: {summary.incorrect_reps}")
    line(f"Average Form Score: {summary.avg_form_score:.2f}/100")
    line(f"Average Rep Time: {summary.avg_rep_time_sec:.2f} sec")
    line(f"Average Symmetry Gap: {summary.avg_symmetry_gap:.2f} deg")
    line(f"Average Torso Lean: {summary.avg_torso_lean:.2f} deg")
    line(f"Average ROM Score: {summary.avg_rom_score:.2f}")
    line(f"Average Tempo Score: {summary.avg_tempo_score:.2f}")
    line(f"Average Stability Score: {summary.avg_stability_score:.2f}")
    line(f"Estimated Calories Burned: {summary.estimated_calories:.2f} kcal")
    line(f"BMI: {summary.bmi:.2f}")
    line(f"BMI Category: {summary.bmi_category}")
    line(f"BMI Note: {summary.bmi_note}")
    line(f"Consistency Score: {summary.consistency_score:.2f}/100")
    line(f"Intensity: {summary.intensity}")
    line(f"Workout Load: {summary.workout_load_score:.2f}")
    line(f"Risk Flag: {summary.risk_flag}")
    line(f"Hydration Note: {summary.hydration_note}")
    line(f"Overall Rating: {summary.overall_rating}")
    line(f"Recommendation: {summary.recommendation}")

    y -= 8
    line("Top Issues:", font="Helvetica-Bold", size=12, gap=18)
    for issue in summary.top_issues:
        line(f"- {issue}")

    y -= 8
    line("Rep-by-Rep Details:", font="Helvetica-Bold", size=12, gap=18)
    for rep in rep_records[:20]:
        line(
            f"Rep {rep.rep_number}: Score {rep.form_score:.1f}, "
            f"Depth {rep.depth_label}, Duration {rep.duration_sec:.2f}s, "
            f"Status {rep.correctness}, Warnings: {rep.warnings}"
        )

    c.save()


# ============================================================
# DRAW DASHBOARD
# ============================================================

def draw_dashboard(frame, exercise, rep_count, correct_reps, incorrect_reps,
                   stage, score, feedback, size_mb, pose_coverage):
    h, w, _ = frame.shape
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 145), (18, 18, 18), -1)
    frame = cv2.addWeighted(overlay, 0.82, frame, 0.18, 0)

    color = exercise_color(exercise)

    texts = [
        f"Exercise: {exercise}",
        f"Reps: {rep_count}",
        f"Correct: {correct_reps}",
        f"Incorrect: {incorrect_reps}",
        f"Stage: {stage}",
        f"Score: {int(score)}",
        f"Size: {size_mb:.1f} MB",
        f"Pose Coverage: {pose_coverage:.1f}%",
        f"Feedback: {feedback}",
    ]

    positions = [
        (20, 30), (20, 65), (160, 65), (320, 65),
        (520, 30), (520, 65), (720, 30), (720, 65), (20, 110)
    ]
    scales = [0.9, 0.9, 0.8, 0.8, 0.9, 0.9, 0.8, 0.8, 0.8]

    for i, text in enumerate(texts):
        cv2.putText(
            frame, text, positions[i], cv2.FONT_HERSHEY_SIMPLEX,
            scales[i], color if i == 0 else (255, 255, 255), 2, cv2.LINE_AA
        )

    return frame


# ============================================================
# VIDEO ANALYSIS
# ============================================================

def build_default_summary(video_name: str, size_mb: float, accepted: bool, reason: str) -> SessionSummary:
    bmi_value = calculate_bmi(USER_WEIGHT_KG, USER_HEIGHT_M)

    return SessionSummary(
        video_name=video_name,
        file_size_mb=size_mb,
        accepted=accepted,
        reason=reason,
        detected_exercise="Unknown",
        duration_sec=0.0,
        frames_total=0,
        pose_detected_frames=0,
        pose_coverage_percent=0.0,
        total_reps=0,
        correct_reps=0,
        incorrect_reps=0,
        avg_form_score=0.0,
        avg_rep_time_sec=0.0,
        avg_symmetry_gap=0.0,
        avg_torso_lean=0.0,
        avg_rom_score=0.0,
        avg_tempo_score=0.0,
        avg_stability_score=0.0,
        estimated_calories=0.0,
        bmi=round(bmi_value, 2),
        bmi_category=bmi_category_4(bmi_value),
        bmi_note=bmi_health_note(bmi_value),
        consistency_score=0.0,
        intensity="Low",
        workout_load_score=0.0,
        risk_flag=readiness_risk_flag(0.0, 0.0, 0.0, bmi_value),
        hydration_note="Hydrate normally after session.",
        overall_rating="Not Processed",
        recommendation="Check input settings",
        top_issues=[reason],
    )


def analyze_video(video_path: str) -> SessionSummary:
    video_name = os.path.basename(video_path)
    base_name = os.path.splitext(video_name)[0]
    session_dir = os.path.join(OUTPUT_DIR, base_name)
    safe_mkdir(session_dir)

    size_mb = get_video_size_mb(video_path)
    allowed, reason = is_video_allowed(video_path)

    if not allowed:
        summary = build_default_summary(video_name, size_mb, False, reason)
        summary.overall_rating = "Rejected"
        summary.recommendation = "Compress or trim the video below 100 MB"
        write_json(os.path.join(session_dir, f"{base_name}_summary.json"), asdict(summary))
        return summary

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        summary = build_default_summary(video_name, size_mb, False, "Could not open video")
        summary.overall_rating = "Error"
        summary.recommendation = "Check the video format"
        write_json(os.path.join(session_dir, f"{base_name}_summary.json"), asdict(summary))
        return summary

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frames_total / fps if fps > 0 else 0.0

    out_video_path = os.path.join(session_dir, f"{base_name}_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    rep_csv_path = os.path.join(session_dir, f"{base_name}_reps.csv")
    rep_json_path = os.path.join(session_dir, f"{base_name}_reps.json")
    summary_json_path = os.path.join(session_dir, f"{base_name}_summary.json")
    pdf_path = os.path.join(session_dir, f"{base_name}_report.pdf")

    charts_dir = os.path.join(session_dir, "charts")
    safe_mkdir(charts_dir)

    sample_frames = []
    pose_detected_frames = 0

    rep_records: List[RepRecord] = []
    knee_series = []
    elbow_series = []
    score_series = []
    torso_series = []
    symmetry_series = []
    rep_time_series = []

    stage = "UP"
    rep_count = 0
    correct_reps = 0
    incorrect_reps = 0
    feedback = "Start movement"

    rep_buffer = []
    rep_start_time = None

    detected_exercise = "Unknown"
    stable_exercise = "Unknown"
    exercise_locked = False
    live_score = 0.0

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        num_poses=1,
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_index = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(frame_index * (1000.0 / fps))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)
            image = frame.copy()

            if results.pose_landmarks and len(results.pose_landmarks) > 0:
                pose_detected_frames += 1
                landmarks = results.pose_landmarks[0]

                draw_pose_landmarks(image, landmarks, width, height)

                if landmarks_ok(landmarks, [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]):
                    ls = get_landmark_xy(landmarks, 11)
                    rs = get_landmark_xy(landmarks, 12)
                    le = get_landmark_xy(landmarks, 13)
                    re = get_landmark_xy(landmarks, 14)
                    lw = get_landmark_xy(landmarks, 15)
                    rw = get_landmark_xy(landmarks, 16)
                    lh = get_landmark_xy(landmarks, 23)
                    rh = get_landmark_xy(landmarks, 24)
                    lk = get_landmark_xy(landmarks, 25)
                    rk = get_landmark_xy(landmarks, 26)
                    la = get_landmark_xy(landmarks, 27)
                    ra = get_landmark_xy(landmarks, 28)

                    left_knee = calculate_angle(lh, lk, la)
                    right_knee = calculate_angle(rh, rk, ra)
                    left_elbow = calculate_angle(ls, le, lw)
                    right_elbow = calculate_angle(rs, re, rw)
                    left_torso_ref = (lh[0], lh[1] - 0.25)
                    torso_lean = calculate_angle(left_torso_ref, lh, ls)

                    # Hip angle (shoulder-hip-knee) for deadlift detection
                    left_hip_angle = calculate_angle(ls, lh, lk)
                    right_hip_angle = calculate_angle(rs, rh, rk)
                    avg_hip_angle = (left_hip_angle + right_hip_angle) / 2

                    avg_knee = (left_knee + right_knee) / 2
                    avg_elbow = (left_elbow + right_elbow) / 2
                    symmetry_gap = abs(left_knee - right_knee)

                    knee_series.append(avg_knee)
                    elbow_series.append(avg_elbow)
                    torso_series.append(torso_lean)
                    symmetry_series.append(symmetry_gap)

                    if frame_index % max(int(fps // 4), 1) == 0:
                        sample_frames.append({
                            "left_knee": left_knee,
                            "right_knee": right_knee,
                            "left_elbow": left_elbow,
                            "right_elbow": right_elbow,
                            "torso_lean": torso_lean,
                            "hip_y": (lh[1] + rh[1]) / 2,
                            "shoulder_y": (ls[1] + rs[1]) / 2,
                            "hip_angle": avg_hip_angle,
                            "wrist_y": (lw[1] + rw[1]) / 2,
                        })

                    if not exercise_locked and len(sample_frames) >= 24:
                        detected_exercise = detect_exercise_with_majority(sample_frames, window_size=12)
                        if detected_exercise != "Unknown":
                            stable_exercise = detected_exercise

                    if not exercise_locked and len(sample_frames) >= 40:
                        locked_guess = detect_exercise_with_majority(sample_frames, window_size=12)
                        if locked_guess != "Unknown":
                            stable_exercise = locked_guess
                            exercise_locked = True

                    # Keep 'Unknown' if no exercise detected — prevents wrong rep counting
                    current_exercise = stable_exercise if stable_exercise != "Unknown" else "Unknown"

                    primary_angle = avg_knee
                    down_th = SQUAT_DOWN_THRESHOLD
                    up_th = SQUAT_UP_THRESHOLD

                    if current_exercise == "Push-Up":
                        primary_angle = avg_elbow
                        down_th = PUSHUP_DOWN_THRESHOLD
                        up_th = PUSHUP_UP_THRESHOLD
                        symmetry_gap = abs(left_elbow - right_elbow)

                    elif current_exercise == "Lunge":
                        primary_angle = min(left_knee, right_knee)
                        down_th = LUNGE_DOWN_THRESHOLD
                        up_th = LUNGE_UP_THRESHOLD
                        symmetry_gap = abs(left_knee - right_knee)

                    elif current_exercise == "Bicep Curl":
                        primary_angle = avg_elbow
                        down_th = CURL_DOWN_THRESHOLD
                        up_th = CURL_UP_THRESHOLD
                        symmetry_gap = abs(left_elbow - right_elbow)

                    elif current_exercise == "Shoulder Press":
                        primary_angle = avg_elbow
                        down_th = PRESS_DOWN_THRESHOLD
                        up_th = PRESS_UP_THRESHOLD
                        symmetry_gap = abs(left_elbow - right_elbow)

                    elif current_exercise == "Deadlift":
                        primary_angle = avg_hip_angle
                        down_th = DEADLIFT_DOWN_THRESHOLD
                        up_th = DEADLIFT_UP_THRESHOLD
                        symmetry_gap = abs(left_hip_angle - right_hip_angle)

                    depth = classify_depth(current_exercise, avg_knee, avg_elbow, avg_hip_angle)
                    current_time = frame_index / fps

                    # Live frame score before first rep
                    live_rom_score = compute_rom_score(current_exercise, primary_angle)
                    live_tempo_score = 70
                    live_stability_score = compute_stability_score(symmetry_gap, torso_lean)
                    live_score = compute_form_score(
                        current_exercise,
                        live_rom_score,
                        live_tempo_score,
                        live_stability_score
                    )

                    # Smoother rep detection
                    if stage == "UP" and primary_angle <= down_th:
                        stage = "DOWN"
                        rep_start_time = current_time
                        rep_buffer = []

                    if stage == "DOWN":
                        rep_buffer.append({
                            "time": current_time,
                            "left_knee": left_knee,
                            "right_knee": right_knee,
                            "left_elbow": left_elbow,
                            "right_elbow": right_elbow,
                            "torso_lean": torso_lean,
                            "symmetry_gap": symmetry_gap,
                            "primary_angle": primary_angle,
                            "exercise": current_exercise,
                        })

                    if stage == "DOWN" and primary_angle >= up_th and len(rep_buffer) > 5:
                        stage = "UP"
                        rep_count += 1
                        rep_end_time = current_time
                        rep_duration = max(0.0, rep_end_time - (rep_start_time if rep_start_time is not None else rep_end_time))
                        rep_time_series.append(rep_duration)

                        primary_min = min(r["primary_angle"] for r in rep_buffer)
                        avg_left_knee_rep = avg([r["left_knee"] for r in rep_buffer])
                        avg_right_knee_rep = avg([r["right_knee"] for r in rep_buffer])
                        avg_left_elbow_rep = avg([r["left_elbow"] for r in rep_buffer])
                        avg_right_elbow_rep = avg([r["right_elbow"] for r in rep_buffer])
                        avg_torso_rep = avg([r["torso_lean"] for r in rep_buffer])
                        avg_symmetry_rep = avg([r["symmetry_gap"] for r in rep_buffer])

                        rep_depth = classify_depth(
                            current_exercise,
                            avg([(r["left_knee"] + r["right_knee"]) / 2 for r in rep_buffer]),
                            avg([(r["left_elbow"] + r["right_elbow"]) / 2 for r in rep_buffer]),
                        )

                        rom_score = compute_rom_score(current_exercise, primary_min)
                        tempo_score = compute_tempo_score(rep_duration)
                        stability_score = compute_stability_score(avg_symmetry_rep, avg_torso_rep)
                        form_score = compute_form_score(current_exercise, rom_score, tempo_score, stability_score)
                        score_series.append(form_score)

                        warnings = get_warnings(
                            current_exercise,
                            rep_depth,
                            avg_symmetry_rep,
                            avg_torso_rep,
                            form_score
                        )

                        is_correct = form_score >= 70 and rep_depth in ["Standard", "Deep"]
                        if is_correct:
                            correct_reps += 1
                        else:
                            incorrect_reps += 1

                        rep = RepRecord(
                            rep_number=rep_count,
                            exercise=current_exercise,
                            start_time_sec=round(rep_start_time if rep_start_time is not None else 0.0, 2),
                            end_time_sec=round(rep_end_time, 2),
                            duration_sec=round(rep_duration, 2),
                            depth_label=rep_depth,
                            form_score=round(form_score, 2),
                            quality_label=quality_label(form_score),
                            correctness="Correct" if is_correct else "Incorrect",
                            avg_left_knee=round(avg_left_knee_rep, 2),
                            avg_right_knee=round(avg_right_knee_rep, 2),
                            avg_left_elbow=round(avg_left_elbow_rep, 2),
                            avg_right_elbow=round(avg_right_elbow_rep, 2),
                            avg_torso_lean=round(avg_torso_rep, 2),
                            symmetry_gap=round(avg_symmetry_rep, 2),
                            rom_score=round(rom_score, 2),
                            tempo_score=round(tempo_score, 2),
                            stability_score=round(stability_score, 2),
                            warnings="; ".join(warnings) if warnings else "None",
                        )
                        rep_records.append(rep)

                    shown_score = score_series[-1] if score_series else live_score

                    if depth == "Shallow":
                        feedback = "Increase range of motion"
                    elif symmetry_gap > SYMMETRY_WARNING_DEG:
                        feedback = "Improve left-right balance"
                    elif torso_lean > TORSO_LEAN_WARNING_DEG and current_exercise in ["Squat", "Lunge"]:
                        feedback = "Reduce torso lean"
                    elif shown_score >= 80:
                        feedback = "Good movement quality"
                    else:
                        feedback = "Continue controlled movement"

                    pose_coverage = (pose_detected_frames / max(frame_index + 1, 1)) * 100.0
                    shown_exercise = stable_exercise if stable_exercise != "Unknown" else "Squat"

                    if frame_index % 10 == 0:
                        print(
                            f"[DEBUG] frame={frame_index} "
                            f"exercise={shown_exercise} "
                            f"left_knee={left_knee:.1f} "
                            f"right_knee={right_knee:.1f} "
                            f"left_elbow={left_elbow:.1f} "
                            f"right_elbow={right_elbow:.1f} "
                            f"primary_angle={primary_angle:.1f} "
                            f"stage={stage} "
                            f"live_score={shown_score:.1f}"
                        )

                    image = draw_dashboard(
                        image,
                        shown_exercise,
                        rep_count,
                        correct_reps,
                        incorrect_reps,
                        stage,
                        shown_score,
                        feedback,
                        size_mb,
                        pose_coverage
                    )

            else:
                pose_coverage = (pose_detected_frames / max(frame_index + 1, 1)) * 100.0
                shown_exercise = stable_exercise if stable_exercise != "Unknown" else "Squat"
                shown_score = score_series[-1] if score_series else live_score

                image = draw_dashboard(
                    image,
                    shown_exercise,
                    rep_count,
                    correct_reps,
                    incorrect_reps,
                    stage,
                    shown_score,
                    "Pose not detected",
                    size_mb,
                    pose_coverage
                )

            out.write(image)
            frame_index += 1

    cap.release()
    out.release()

    if stable_exercise == "Unknown":
        stable_exercise = "Squat"

    bmi_value = calculate_bmi(USER_WEIGHT_KG, USER_HEIGHT_M)
    bmi_group = bmi_category_4(bmi_value)
    bmi_note = bmi_health_note(bmi_value)

    if pose_detected_frames < MIN_POSE_FRAMES_FOR_ANALYSIS:
        summary = SessionSummary(
            video_name=video_name,
            file_size_mb=size_mb,
            accepted=False,
            reason="Not enough pose frames detected for analysis",
            detected_exercise=stable_exercise,
            duration_sec=round(duration_sec, 2),
            frames_total=frames_total,
            pose_detected_frames=pose_detected_frames,
            pose_coverage_percent=round((pose_detected_frames / max(frames_total, 1)) * 100.0, 2),
            total_reps=0,
            correct_reps=0,
            incorrect_reps=0,
            avg_form_score=0.0,
            avg_rep_time_sec=0.0,
            avg_symmetry_gap=round(avg(symmetry_series), 2),
            avg_torso_lean=round(avg(torso_series), 2),
            avg_rom_score=0.0,
            avg_tempo_score=0.0,
            avg_stability_score=0.0,
            estimated_calories=0.0,
            bmi=round(bmi_value, 2),
            bmi_category=bmi_group,
            bmi_note=bmi_note,
            consistency_score=0.0,
            intensity="Low",
            workout_load_score=0.0,
            risk_flag=readiness_risk_flag(0.0, avg(symmetry_series), avg(torso_series), bmi_value),
            hydration_note="Hydrate normally after session.",
            overall_rating="Insufficient Data",
            recommendation="Capture a clearer full-body side/front view",
            top_issues=["Pose detection too low"],
        )
        write_json(summary_json_path, asdict(summary))
        return summary

    rep_rows = [asdict(r) for r in rep_records]
    write_csv(rep_csv_path, rep_rows)
    write_json(rep_json_path, {"video": video_name, "reps": rep_rows})

    chart_score = os.path.join(charts_dir, "rep_scores.png")
    chart_knee = os.path.join(charts_dir, "knee_angles.png")
    chart_elbow = os.path.join(charts_dir, "elbow_angles.png")
    chart_symmetry = os.path.join(charts_dir, "symmetry_gap.png")

    save_line_plot(score_series, "Rep Score Trend", "Rep", "Score", chart_score)
    save_line_plot(knee_series, "Knee Angle per Frame", "Frame", "Degrees", chart_knee)
    save_line_plot(elbow_series, "Elbow Angle per Frame", "Frame", "Degrees", chart_elbow)
    save_line_plot(symmetry_series, "Symmetry Gap per Frame", "Frame", "Degrees", chart_symmetry)

    avg_form_score = avg(score_series)
    avg_rep_time = avg(rep_time_series)
    avg_symmetry_gap = avg([r.symmetry_gap for r in rep_records])
    avg_torso_lean = avg([r.avg_torso_lean for r in rep_records])
    avg_rom_score = avg([r.rom_score for r in rep_records])
    avg_tempo_score = avg([r.tempo_score for r in rep_records])
    avg_stability_score = avg([r.stability_score for r in rep_records])
    pose_coverage_percent = (pose_detected_frames / max(frames_total, 1)) * 100.0

    consistency = calculate_consistency_score(score_series)
    intensity = session_intensity(avg_form_score, avg_rep_time, rep_count)
    load_score = workout_load(rep_count, avg_form_score, duration_sec)
    calories = estimate_calories(duration_sec, USER_WEIGHT_KG, EXERCISE_MET.get(stable_exercise, 4.0))
    risk_flag = readiness_risk_flag(avg_form_score, avg_symmetry_gap, avg_torso_lean, bmi_value)
    hydration_note = hydration_reminder(duration_sec, intensity)
    recommendation = get_goal_recommendation(GOAL_MODE, avg_form_score, stable_exercise)

    top_issues = []
    if any(r.depth_label == "Shallow" for r in rep_records):
        top_issues.append("Range of motion is inconsistent")
    if avg_symmetry_gap > SYMMETRY_WARNING_DEG:
        top_issues.append("Asymmetry between left and right side")
    if avg_torso_lean > TORSO_LEAN_WARNING_DEG and stable_exercise in ["Squat", "Lunge"]:
        top_issues.append("Excessive forward torso lean")
    if pose_coverage_percent < 75:
        top_issues.append("Pose visibility is limited in parts of the video")
    if consistency < 70:
        top_issues.append("Repetition quality is not consistent")
    if risk_flag == "High Risk":
        top_issues.append("Needs cautious progression and form correction")
    if not top_issues:
        top_issues.append("Movement quality is generally consistent")

    summary = SessionSummary(
        video_name=video_name,
        file_size_mb=size_mb,
        accepted=True,
        reason="Processed successfully",
        detected_exercise=stable_exercise,
        duration_sec=round(duration_sec, 2),
        frames_total=frames_total,
        pose_detected_frames=pose_detected_frames,
        pose_coverage_percent=round(pose_coverage_percent, 2),
        total_reps=rep_count,
        correct_reps=correct_reps,
        incorrect_reps=incorrect_reps,
        avg_form_score=round(avg_form_score, 2),
        avg_rep_time_sec=round(avg_rep_time, 2),
        avg_symmetry_gap=round(avg_symmetry_gap, 2),
        avg_torso_lean=round(avg_torso_lean, 2),
        avg_rom_score=round(avg_rom_score, 2),
        avg_tempo_score=round(avg_tempo_score, 2),
        avg_stability_score=round(avg_stability_score, 2),
        estimated_calories=round(calories, 2),
        bmi=round(bmi_value, 2),
        bmi_category=bmi_group,
        bmi_note=bmi_note,
        consistency_score=round(consistency, 2),
        intensity=intensity,
        workout_load_score=round(load_score, 2),
        risk_flag=risk_flag,
        hydration_note=hydration_note,
        overall_rating=quality_label(avg_form_score),
        recommendation=recommendation,
        top_issues=top_issues,
    )

    write_json(summary_json_path, asdict(summary))
    generate_pdf_report(pdf_path, summary, rep_records, [chart_score, chart_knee, chart_elbow, chart_symmetry])

    return summary


# ============================================================
# BATCH PROCESSING
# ============================================================

def collect_videos(input_dir: str) -> List[str]:
    if not os.path.isdir(input_dir):
        return []

    files = []
    for name in os.listdir(input_dir):
        path = os.path.join(input_dir, name)
        if os.path.isfile(path) and name.lower().endswith(SUPPORTED_EXTENSIONS):
            files.append(path)

    files.sort()
    return files


def save_batch_reports(summaries: List[SessionSummary]):
    batch_csv = os.path.join(OUTPUT_DIR, "batch_summary.csv")
    batch_json = os.path.join(OUTPUT_DIR, "batch_summary.json")

    rows = [asdict(s) for s in summaries]
    write_csv(batch_csv, rows)
    write_json(batch_json, {"sessions": rows})

    print(f"[OK] Batch CSV saved: {batch_csv}")
    print(f"[OK] Batch JSON saved: {batch_json}")


def main():
    print(f"[DEBUG] BASE_DIR: {BASE_DIR}")
    print(f"[DEBUG] INPUT_DIR: {INPUT_DIR}")
    print(f"[DEBUG] OUTPUT_DIR: {OUTPUT_DIR}")

    videos = collect_videos(INPUT_DIR)

    if not videos:
        print(f"[INFO] No videos found in folder: {INPUT_DIR}")
        print(f"[INFO] Supported formats: {SUPPORTED_EXTENSIONS}")
        return

    print(f"[INFO] Found {len(videos)} video(s)")
    summaries = []

    for idx, video_path in enumerate(videos, start=1):
        print("=" * 80)
        print(f"[INFO] Processing {idx}/{len(videos)}: {video_path}")
        try:
            summary = analyze_video(video_path)
            summaries.append(summary)
            print(f"[DONE] {summary.video_name}")
            print(f"       Exercise: {summary.detected_exercise}")
            print(f"       Reps: {summary.total_reps}")
            print(f"       Avg Score: {summary.avg_form_score}")
            print(f"       Calories: {summary.estimated_calories}")
            print(f"       BMI: {summary.bmi} ({summary.bmi_category})")
            print(f"       Risk: {summary.risk_flag}")
            print(f"       Status: {summary.reason}")
        except Exception as e:
            traceback.print_exc()
            size_mb = get_video_size_mb(video_path)
            error_summary = build_default_summary(
                os.path.basename(video_path),
                size_mb,
                False,
                f"Processing failed: {str(e)}"
            )
            error_summary.overall_rating = "Error"
            error_summary.recommendation = "Check logs and input video quality"
            summaries.append(error_summary)

    save_batch_reports(summaries)

    print("=" * 80)
    print("[COMPLETE] All videos processed.")
    print(f"[OUTPUT] See folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()