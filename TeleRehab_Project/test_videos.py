"""Quick test script to run ML analysis on the poseVideos samples."""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import analyze_video
from dataclasses import asdict

VIDEOS = [
    "poseVideos/bicepCurl.mp4",
    "poseVideos/backSquate.mp4",
    "poseVideos/sideBend.mp4",
    "poseVideos/toeTouch.mp4",
    "poseVideos/armRotation.mp4",
]

for vpath in VIDEOS:
    if not os.path.isfile(vpath):
        print(f"SKIP {vpath} (not found)")
        continue
    print(f"\n{'='*60}")
    print(f"VIDEO: {vpath}")
    print('='*60)
    try:
        s = analyze_video(vpath)
        d = asdict(s)
        print(f"  Exercise detected : {d['detected_exercise']}")
        print(f"  Total reps        : {d['total_reps']}")
        print(f"  Correct / Incorrect: {d['correct_reps']} / {d['incorrect_reps']}")
        print(f"  Form score avg    : {d['avg_form_score']:.1f}")
        print(f"  Pose coverage     : {d['pose_coverage_percent']:.1f}%")
        print(f"  Duration          : {d['duration_sec']:.1f}s")
        print(f"  Rating            : {d['overall_rating']}")
        print(f"  ROM / Tempo / Stab: {d['avg_rom_score']:.0f} / {d['avg_tempo_score']:.0f} / {d['avg_stability_score']:.0f}")
        print(f"  Risk flag         : {d['risk_flag']}")
        print(f"  Top issues        : {d['top_issues']}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
