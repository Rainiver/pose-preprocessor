import json
import cv2
import numpy as np


def load_openpose_keypoints(json_path):
    """Load keypoints from OpenPose JSON output."""
    with open(json_path, "r") as f:
        data = json.load(f)
    if "people" not in data or len(data["people"]) == 0:
        return []
    return data["people"][0]["pose_keypoints_2d"]


def draw_openpose_skeleton(image, keypoints, threshold=0.1):
    """Draw OpenPose skeleton on the image."""
    if len(keypoints) == 0:
        return image

    # COCO body-25 connections
    body_pairs = [
        (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
        (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
        (0, 1), (0, 14), (14, 16), (0, 15), (15, 17)
    ]

    points = []
    for i in range(0, len(keypoints), 3):
        x, y, c = keypoints[i:i+3]
        if c > threshold:
            points.append((int(x), int(y)))
        else:
            points.append(None)

    # Draw joints
    for p in points:
        if p:
            cv2.circle(image, p, 3, (0, 255, 0), -1)

    # Draw limbs
    for a, b in body_pairs:
        if points[a] and points[b]:
            cv2.line(image, points[a], points[b], (255, 0, 0), 2)

    return image
