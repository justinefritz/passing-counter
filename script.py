import cv2
import time
import os
import csv
import gc
import queue
import threading
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = "yolov8n-pose.pt"
CSV_FILE = "passing_counts.csv"

SHOW_DISPLAY = True          # Set False on headless bus deployment
SKIP_FRAMES = 2              # Process AI every N frames
INFERENCE_SIZE = (320, 240)  # Input resolution for Nano model

# Lower threshold (0.25) so facial/head keypoints are not dropped
KEYPOINT_CONF_THRESH = 0.25  

# ==========================================
# ASYNC CSV LOGGING
# ==========================================
log_queue = queue.Queue()

def async_csv_writer():
    file_exists = os.path.exists(CSV_FILE)
    with open(CSV_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Track ID", "Direction", "Total Right", "Total Left"])
            f.flush()
        
        while True:
            data = log_queue.get()
            if data is None:
                break
            writer.writerow(data)
            f.flush()
            log_queue.task_done()

writer_thread = threading.Thread(target=async_csv_writer, daemon=True)
writer_thread.start()

def queue_event(track_id, direction, total_right, total_left):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_queue.put([timestamp, f"Person_{track_id}", direction, total_right, total_left])

# ==========================================
# MAIN APC ENGINE
# ==========================================
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

right_count = 0
left_count = 0
person_zone = {}

frame_count = 0
cached_tracks = []

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip horizontally to match physical room motion
        frame = cv2.flip(frame, 1)

        frame_count += 1
        height, width, _ = frame.shape
        gate_x = width // 2

        current_frame_ids = set()

        # 1. OPTIMIZED AI INFERENCE (YOLO Nano Pose)
        if frame_count % SKIP_FRAMES == 0 or len(cached_tracks) == 0:
            small_frame = cv2.resize(frame, INFERENCE_SIZE)
            scale_x = width / float(INFERENCE_SIZE[0])
            scale_y = height / float(INFERENCE_SIZE[1])

            results = model.track(small_frame, classes=[0], persist=True, verbose=False)

            cached_tracks = []
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                keypoints_data = results[0].keypoints

                if boxes.id is not None and keypoints_data is not None:
                    track_ids = boxes.id.int().cpu().tolist()
                    kpts_tensor = keypoints_data.xy.cpu().numpy()
                    conf_tensor = keypoints_data.conf.cpu().numpy()

                    for kpts, confs, track_id in zip(kpts_tensor, conf_tensor, track_ids):
                        # Head & Shoulder keypoints (0 to 6)
                        head_shoulder_kpts = kpts[0:7]
                        head_shoulder_confs = confs[0:7] if confs is not None else [1.0] * 7

                        valid_kpts = []
                        has_head_pt = False

                        for idx, (pt, conf) in enumerate(zip(head_shoulder_kpts, head_shoulder_confs)):
                            if conf > KEYPOINT_CONF_THRESH and pt[0] > 0 and pt[1] > 0:
                                valid_kpts.append(pt)
                                if idx <= 4:  # Detected nose, eye, or ear
                                    has_head_pt = True

                        if len(valid_kpts) >= 1:
                            valid_kpts = np.array(valid_kpts)
                            
                            hx1 = max(0, int(np.min(valid_kpts[:, 0]) * scale_x) - 15)
                            hy1 = max(0, int(np.min(valid_kpts[:, 1]) * scale_y) - 15)
                            hx2 = min(width, int(np.max(valid_kpts[:, 0]) * scale_x) + 15)
                            hy2 = min(height, int(np.max(valid_kpts[:, 1]) * scale_y) + 15)

                            if not has_head_pt:
                                hy1 = max(0, hy1 - 25)

                            hcx = (hx1 + hx2) // 2
                            cached_tracks.append((hx1, hy1, hx2, hy2, track_id, hcx, has_head_pt))

            if frame_count % 120 == 0:
                gc.collect()

        # 2. LINE CROSSING & ZONE LOGIC
        for hx1, hy1, hx2, hy2, track_id, hcx, has_head_pt in cached_tracks:
            current_frame_ids.add(track_id)
            current_zone = "LEFT" if hcx < gate_x else "RIGHT"

            if track_id in person_zone:
                previous_zone = person_zone[track_id]

                # Moving from RIGHT to LEFT -> Count as LEFT
                if previous_zone == "RIGHT" and current_zone == "LEFT":
                    right_count += 1
                    queue_event(track_id, "LEFT", right_count, left_count)
                    person_zone[track_id] = current_zone

                # Moving from LEFT to RIGHT -> Count as RIGHT
                elif previous_zone == "LEFT" and current_zone == "RIGHT":
                    left_count += 1
                    queue_event(track_id, "RIGHT", right_count, left_count)
                    person_zone[track_id] = current_zone
            else:
                person_zone[track_id] = current_zone

            if SHOW_DISPLAY:
                color = (0, 255, 0) if has_head_pt else (0, 255, 255)
                # Drawing bounding box & center point without any text labels
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), color, 2)
                cv2.circle(frame, (hcx, (hy1 + hy2) // 2), 4, (0, 0, 255), -1)

        # Clean stale IDs
        stale_ids = set(person_zone.keys()) - current_frame_ids
        for sid in stale_ids:
            del person_zone[sid]

        if SHOW_DISPLAY:
            cv2.line(frame, (gate_x, 0), (gate_x, height), (255, 255, 0), 2)
            cv2.putText(frame, f"Right: {right_count} | Left: {left_count}", 
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("APC System (Head + Shoulder Tracking)", frame)

            if cv2.waitKey(15) & 0xFF == ord('q'):
                break

finally:
    cap.release()
    if SHOW_DISPLAY:
        cv2.destroyAllWindows()
    log_queue.put(None)
    writer_thread.join()