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
CSV_FILE = "apc_passenger_counts.csv"

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
            writer.writerow(["Timestamp", "Track ID", "Direction", "Total Boarding", "Total Alighting"])
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

def queue_event(track_id, direction, total_in, total_out):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_queue.put([timestamp, f"Person_{track_id}", direction, total_in, total_out])

# ==========================================
# MAIN APC ENGINE (HEAD + SHOULDER DETECTION)
# ==========================================
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

boarding_count = 0
alighting_count = 0
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
                        # COCO Schema: 
                        # 0: Nose, 1: L-Eye, 2: R-Eye, 3: L-Ear, 4: R-Ear (HEAD)
                        # 5: L-Shoulder, 6: R-Shoulder (SHOULDERS)
                        head_shoulder_kpts = kpts[0:7]
                        head_shoulder_confs = confs[0:7] if confs is not None else [1.0] * 7

                        # Filter valid keypoints using the lower 0.25 threshold
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

                            # Overhead Compensation: If only shoulders were seen (no facial keypoints),
                            # expand the top boundary (hy1) upwards by 25 pixels to guarantee head inclusion.
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

                # Right to Left -> BOARDING
                if previous_zone == "RIGHT" and current_zone == "LEFT":
                    boarding_count += 1
                    queue_event(track_id, "BOARDING", boarding_count, alighting_count)
                    person_zone[track_id] = current_zone

                # Left to Right -> ALIGHTING
                elif previous_zone == "LEFT" and current_zone == "RIGHT":
                    alighting_count += 1
                    queue_event(track_id, "ALIGHTING", boarding_count, alighting_count)
                    person_zone[track_id] = current_zone
            else:
                person_zone[track_id] = current_zone

            if SHOW_DISPLAY:
                color = (0, 255, 0) if has_head_pt else (0, 255, 255) # Green if head seen, Yellow if shoulder estimated
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), color, 2)
                cv2.circle(frame, (hcx, (hy1 + hy2) // 2), 4, (0, 0, 255), -1)
                
                label = f"Head+Shoulder #{track_id}" if has_head_pt else f"Shoulder+Pad #{track_id}"
                cv2.putText(frame, label, (hx1, max(hy1 - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Clean stale IDs
        stale_ids = set(person_zone.keys()) - current_frame_ids
        for sid in stale_ids:
            del person_zone[sid]

        if SHOW_DISPLAY:
            cv2.line(frame, (gate_x, 0), (gate_x, height), (255, 255, 0), 2)
            cv2.putText(frame, f"Boarding: {boarding_count} | Alighting: {alighting_count}", 
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