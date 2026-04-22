import cv2


def draw_status_panel(image, pose_detected, landmark_count, fps):
    cv2.rectangle(image, (10, 10), (320, 140), (30, 30, 30), -1)

    cv2.putText(
        image,
        "TeleRehab Live Monitor",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    pose_text = "Detected" if pose_detected else "Not Detected"
    pose_color = (0, 255, 0) if pose_detected else (0, 0, 255)

    cv2.putText(
        image,
        f"Pose: {pose_text}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        pose_color,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        image,
        f"Landmarks: {landmark_count}",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        image,
        f"FPS: {fps:.2f}",
        (20, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return image