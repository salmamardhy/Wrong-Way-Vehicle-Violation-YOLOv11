# === FILE: main_bytetrack.py ===
import math
import os
import cv2
import numpy as np
from scipy.stats import skew
from fastapi import FastAPI, UploadFile, File, Query, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import time
from collections import deque, Counter, defaultdict
import mysql.connector
from datetime import datetime, timedelta
import pytz
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio
import numpy as np
import cv2
from scipy.stats import skew
from threading import Thread
from queue import Queue

video_writer_thread = None
video_writer_stop = False

executor = ThreadPoolExecutor()

# === Database MySQL XAMPP ===
def get_db_connection():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="MyNewPass",
        database="traffic_violation"
    )
    try:
        yield conn
    finally:
        conn.close()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_videos"
OUTPUT_DIR = "output_videos"
VIOLATION_DIR = "violation_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VIOLATION_DIR, exist_ok=True)
app.mount("/violation_images", StaticFiles(directory=VIOLATION_DIR), name="violation_images")

# INPUT URL ATAU VIDEO
@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as f:
        content = await file.read()
        f.write(content)
    return {"filename": file.filename}

# CLAHE
def apply_clahe(frame, clip_limit, tile_size):
    # Konversi dari BGR ke LAB
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Terapkan CLAHE ke channel L
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    cl = clahe.apply(l)

    # Gabungkan kembali channel LAB
    merged_lab = cv2.merge((cl, a, b))

    # Konversi kembali ke BGR
    clahe_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    return clahe_bgr

def preprocess_frame(frame, clahe_conf=None, model=None):
    if clahe_conf and clahe_conf.get("clip") is not None:
        return apply_clahe(frame, clahe_conf["clip"], clahe_conf["tile"])
    else:
        return apply_clahe(frame, 2.0, 8)

def evaluate_brisque(img):
    try:
        return cv2.quality.QualityBRISQUE_compute(img, "models/brisque_model_live.yml", "models/brisque_range_live.yml")[0]
    except:
        return float("inf")

def find_best_clahe_config(frames, model, fb):
    starttime_best_config = time.time()
    selected_frames = frames[:3]
    if len(selected_frames) < 3:
        return None

    # Cari frame dengan kendaraan terbanyak
    vehicle_counts = []
    for f in selected_frames:
        result = model.predict(f, conf=0.2, verbose=False)[0]
        count = len(result.boxes) if result.boxes else 0
        vehicle_counts.append(count)

    # Ambil index dengan jumlah kendaraan terbanyak
    max_idx = vehicle_counts.index(max(vehicle_counts))
    chosen_frame = selected_frames[max_idx]

    # Cari konfigurasi CLAHE terbaik berdasarkan nilai BRISQUE
    best_config = {"clip": None, "tile": None}
    original_brisque = evaluate_brisque(chosen_frame)
    best_brisque = original_brisque

    for clip in [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        for tile in [4, 8, 16]:
            enhanced = apply_clahe(chosen_frame, clip, tile)
            score = evaluate_brisque(enhanced)
            if score < best_brisque:
                best_brisque = score
                best_config = {"clip": clip, "tile": tile}

    if best_config["clip"] is not None:
        print(f"- Best CLAHE = {best_config} with best BRISQUE = {best_brisque}")
        print(f"- original BRISQUE = {original_brisque}")
    else:
        print(f"- NON CLAHE with BRISQUE {original_brisque}")
        
    endtime_best_config = time.time()
    time_best_config = endtime_best_config - starttime_best_config        
    return best_config, time_best_config, best_brisque

# MENGHITUNG IOU
def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    denom = float(area1 + area2 - interArea)
    return interArea / denom if denom > 1e-6 else 0.0

def iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0

def get_road_mask_polygons(result, frame):
    mask_binary = np.zeros(frame.shape[:2], dtype=np.uint8)
    polygons = []
    min_area = 200 # 200 piksel minimal area

    if result.masks is not None and len(result.masks.xy) > 0:
        raw_polygons = []
        for seg in result.masks.xy:
            points = np.array(seg, dtype=np.int32).reshape((-1, 1, 2))
            area = cv2.contourArea(points)
            if area > min_area:
                raw_polygons.append((area, points))

        top_polygons = sorted(raw_polygons, key=lambda x: x[0], reverse=True)[:2]

        for _, points in top_polygons:
            polygons.append(points)
            cv2.fillPoly(mask_binary, [points], color=1)

    return mask_binary, polygons

def vehicle_road_overlap_checking(x1, y1, x2, y2, roi_mask, polygons, min_overlap_ratio=0.05):
    bbox_area = (x2 - x1) * (y2 - y1)
    if bbox_area == 0:
        return -1

    mask_crop = roi_mask[y1:y2, x1:x2]
    overlap_pixels = np.count_nonzero(mask_crop)
    overlap_ratio = overlap_pixels / bbox_area

    if overlap_ratio >= min_overlap_ratio:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        for idx, poly in enumerate(polygons):
            if cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0:
                return idx
    return -1


def majority_direction(directions):
    signs = [np.sign(d) for d in directions if d != 0]
    return Counter(signs).most_common(1)[0][0] if signs else 0


def overlay_image_alpha(img, img_overlay, x, y):
    if img_overlay.shape[2] != 4:
        return img
    b, g, r, a = cv2.split(img_overlay)
    overlay_color = cv2.merge((b, g, r))
    mask = cv2.merge((a, a, a)) / 255.0
    h, w = img_overlay.shape[:2]
    roi = img[y:y+h, x:x+w]
    img[y:y+h, x:x+w] = (1.0 - mask) * roi + mask * overlay_color
    return img

def find_segmentation(model, frames):
    frame_vehicle_counts, frame_masks, frame_polygons = [], [], []

    for frame in frames:
        preprocessed = preprocess_frame(frame)
        det_result = model.predict(preprocessed, conf=0.5, verbose=False)[0]
        num_objects = len(det_result.boxes) if det_result.boxes else 0
        mask, polygons = get_road_mask_polygons(det_result, frame) 
        frame_vehicle_counts.append((num_objects, mask, polygons))

    count_list = [c[0] for c in frame_vehicle_counts]
    if len(set(count_list)) == 1:
        seg_masks = []
        for (_, mask, polygons) in frame_vehicle_counts:
            if mask is not None:
                seg_masks.append(mask)
                if not frame_polygons:
                    frame_polygons = polygons
    else:
        sorted_frames = sorted(frame_vehicle_counts, key=lambda x: x[0])
        top3 = sorted_frames[:3]
        candidates = []
        for i in range(len(top3)):
            for j in range(i + 1, len(top3)):
                mask1, mask2 = top3[i][1], top3[j][1]
                if mask1 is None or mask2 is None:
                    continue
                score = iou(mask1, mask2)
                if score > 0.1:  # ambang minimal overlap
                    combined_mask = np.logical_or(mask1, mask2).astype(np.uint8)
                    candidates.append((score, combined_mask, top3[i][2] or top3[j][2]))

        if not candidates:
            return None, None

        # Pilih kandidat dengan IoU tertinggi
        candidates.sort(key=lambda x: x[0], reverse=True)
        _, roi_mask, frame_polygons = candidates[0]
        return roi_mask, frame_polygons

    if len(seg_masks) >= 2:
        best_iou = -1
        best_pair = None
        for i in range(len(seg_masks)):
            for j in range(i + 1, len(seg_masks)):
                iou_val = iou(seg_masks[i], seg_masks[j])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_pair = (seg_masks[i], seg_masks[j])
        roi_mask = np.logical_or(*best_pair).astype(np.uint8)
    elif seg_masks:
        roi_mask = seg_masks[0].astype(np.uint8)
    else:
        return None, None

    return roi_mask, frame_polygons

def error_response(message):
    img = np.zeros((200, 640, 3), dtype=np.uint8)
    cv2.putText(img, message, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    _, jpeg = cv2.imencode('.jpg', img)
    return StreamingResponse(iter([b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n']), 
                             media_type='multipart/x-mixed-replace; boundary=frame')

def create_capture(source):
    return cv2.VideoCapture(source)

def save_image_async(image, path):
    executor.submit(cv2.imwrite, path, image)

def is_opposite_direction(ref, cur):
    opposites = {
        "right": "left", "left": "right",
        "up": "down", "down": "up"
    }
    return opposites.get(ref) == cur

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return rotated

def majority_vote(directions):
    count = Counter(directions)
    return count.most_common(1)[0][0] if count else None

def get_polygon_direction(i, dx_list, dy_list):
    if not dx_list or not dy_list:
        # print("UNKNOWN")
        return 0, "unknown", None

    if len(dx_list) == len(dy_list) and len(dx_list) % 2 == 0:
        magnitudes = [(i, (dx**2 + dy**2)**0.5) for i, (dx, dy) in enumerate(zip(dx_list, dy_list))]
        min_index, _ = min(magnitudes, key=lambda x: x[1])
        dx_list.pop(min_index)
        dy_list.pop(min_index)

    horizontal_count = 0
    vertical_count = 0
    dx_list_hor, dy_list_hor = [], []
    dx_list_ver, dy_list_ver = [], []
    angle_deg_ver, angle_deg_hor =[], []
    n = 0

    for dx, dy in zip(dx_list, dy_list):
        if n >= 10:
            continue

        if dx == 0 and dy == 0:
            # print("UNKNOWN")
            continue

        mag_movement = (dx**2 + dy**2) ** 0.5
        if mag_movement < 5:
            # print("UNKNOWN")
            return 0, "unknown", None
    
        # Hitung sudut gerakan terhadap sumbu-x (kanan) dalam rentang 0–360 derajat
        angle_rad = math.atan2(dy, dx)  # atan2 sudah memperhatikan tanda dx dan dy
        angle_deg = math.degrees(angle_rad)
        angle_deg_cat = math.degrees(angle_rad)
        if angle_deg_cat < 0:
            angle_deg_cat += 360  # pastikan dalam 0–360

        n += 1
        if (0 <= angle_deg < 35) or (angle_deg > 155 and angle_deg <= 180):
            angle_deg_hor.append(angle_deg_cat)
            dx_list_hor.append(dx)
            dy_list_hor.append(dy)
            horizontal_count += 1
        else:
            angle_deg_ver.append(angle_deg_cat)
            dx_list_ver.append(dx)
            dy_list_ver.append(dy)
            vertical_count += 1

    # print(f"[JALAN] polygon {i}: horizontal {horizontal_count}, vertical {vertical_count}")
    if horizontal_count > vertical_count:
        average = np.mean(angle_deg_hor)
        average_dx_list_hor = np.mean(dx_list_hor)
        average_dy_list_hor = np.mean(dy_list_hor)
        # Arah kanan jika 0–90 derajat (kanan bawah) atau 270–360 derajat (kanan atas)
        direction = "right" if 0 <= average <= 90 or 270 <= average <= 360 else "left"
        dominant_vectors = (average_dx_list_hor, average_dy_list_hor)
        print(f"[VECTOR] dx_list_hor = {dx_list_hor}")
        print(f"[VECTOR] dy_list_hor = {dy_list_hor}")       
        print(f"[ARAH] polygon {i}: {average}° with dir - {direction} & dominant vectors - {dominant_vectors}")
        return average, direction, dominant_vectors

    elif vertical_count > horizontal_count:
        average = np.mean(angle_deg_ver)
        average_dx_list_ver = np.mean(dx_list_ver)
        average_dy_list_ver = np.mean(dy_list_ver)
        # Arah atas jika 180–360 derajat, karena 0–180 adalah arah ke bawah
        direction = "up" if 180 <= average <= 360 else "down"
        dominant_vectors = (average_dx_list_ver, average_dy_list_ver)
        print(f"[VECTOR] dx_list_ver = {dx_list_ver}")
        print(f"[VECTOR] dy_list_ver = {dy_list_ver}")
        print(f"[ARAH] polygon {i}: {average}° with dir - {direction} & dominant vectors - {dominant_vectors}")        
        return average, direction, dominant_vectors

    else:
        print(f"[ARAH] polygon {i}: horizontal {horizontal_count}, vertical {vertical_count}")
        return 0, "unknown", None

def cosine_similarity(v1, v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return np.dot(v1, v2) / (norm1 * norm2)

def compute_angle_direction(width, dominant_vector, track_id, dx, dy, threshold=0.7):
    """
    dominant_vector: vektor arah dominan, dalam bentuk (dx, dy)
    (dx, dy): perpindahan kendaraan
    """

    if dominant_vector is None or (dx == 0 and dy == 0):
        return "unknown"
    
    mag_movement = np.linalg.norm([dx, dy])
    if mag_movement < (round(width * 0.005)):
        return "unknown"

    v1 = np.array([dx, dy], dtype=float)
    v2 = np.array(dominant_vector, dtype=float)

    cos_theta = cosine_similarity(v1, v2)
    if cos_theta > threshold:
        return "searah"
    elif cos_theta < -threshold:
        return "lawan arah"
    else:
        return "unknown"

def apply_clahe_on_roi_only(frame, clip_limit, tile_size, roi_mask):
    # Ambil bounding box dari roi_mask (non-zero area)
    y_indices, x_indices = np.where(roi_mask > 0)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return frame  # Kalau tidak ada area jalan, skip CLAHE

    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    # Crop area dari frame
    roi_crop = frame[y_min:y_max+1, x_min:x_max+1]

    # Apply CLAHE pada crop
    lab = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    enhanced_roi = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Tempel hasil CLAHE ke frame asli
    result = frame.copy()
    result[y_min:y_max+1, x_min:x_max+1] = enhanced_roi

    return result

def save_violation_threaded(frame, coords, meta):
    def save_job():
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="MyNewPass",
            database="traffic_violation"
        )
        cursor = conn.cursor()
        x1, y1, x2, y2 = coords
        cropped = frame[y1:y2, x1:x2]
        cv2.imwrite(meta['img_path'], cropped)

        if meta["type"] == "insert":
            cursor.execute("""INSERT INTO violations 
                (track_id, bytetrack_id, timestamp, x1, y1, x2, y2, source, image_path, image_name, confidence, vehicle, true_direction)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                meta["db_track_id"], meta["track_id"], meta["timestamp"],
                x1, y1, x2, y2, meta["source"],
                meta["img_path"], meta["img_filename"], meta["confidence"], meta["label"], meta["direction"]
            ))
        elif meta["type"] == "update":
            print("UPDATE HRSNYA")
            cursor.execute("""UPDATE violations SET timestamp=%s, x1=%s, y1=%s, x2=%s, y2=%s,
                confidence=%s, image_path=%s, image_name=%s, vehicle=%s WHERE id=%s
            """, (
                meta["timestamp"], x1, y1, x2, y2,
                meta["confidence"], meta["img_path"], meta["img_filename"], meta["label"], meta["db_id"]
            ))

        conn.commit()
        cursor.close()
        conn.close()

    Thread(target=save_job).start()

def video_writer_worker(video_writer, write_queue):
    while True:
        frame = write_queue.get()
        if frame is None:
            break
        video_writer.write(frame)
    video_writer.release()
    print("[DEBUG] Video writer thread stopped and video file closed.")

segmentation_cache = {}
stream_control_flags = {}
violations_store = {}

@app.get("/stream")
def stream_video(source: str = Query(...), reload: str = Query(None), refresh: bool = False):
    global video_writer_stop, video_writer_thread
    def generate_stream():
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="MyNewPass",
            database="traffic_violation"
        )
        db_cursor = conn.cursor()  
        detect_time = None
        end_detect_time = None 
        unique_ids = set()
        last_seen_frame = {} 
        vehicle_positions = {}  
        vehicle_stationary_start = {} 
        time_best_config = None
        track_bboxes = {} 
        best_config = None
        best_brisque = None

        try:
            stream_control_flags[source] = True
            seg_model = YOLO("models/road_best.pt")
            det_model_tracking = YOLO("models/vehicle_best.pt")
            det_model_direction = YOLO("models/vehicle_best.pt")
            is_url_source = source.startswith(("http", "rtmp", "udp"))
            retry_count = 0

            cap = create_capture(source if is_url_source else os.path.join(UPLOAD_DIR, source))
            is_video_file = not is_url_source
            if not cap.isOpened():
                return error_response("Gagal membuka URL")
            
            display_size = None
            fps_video = int(cap.get(cv2.CAP_PROP_FPS))
            if fps_video < 15:
                fps_video = 15
            classification_history = defaultdict(lambda: deque(maxlen=fps_video*2)) 

            raw_frames = []
            for idx in range(3):
                ret, frame = cap.read()
                if not ret:
                    break
                if display_size is None:
                    display_size = (int(frame.shape[1] * 60 / 100), int(frame.shape[0] * 60 / 100))
                raw_frames.append(frame)
                
            
            output_video_path = None
            video_writer = None

            if is_video_file:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec untuk .mp4
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                width_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename_base = os.path.splitext(os.path.basename(source))[0]  

                # --- Tulis file .mp4 ---              
                output_video_filename = f"{filename_base}_output_{current_date}_nonclahe.mp4"
                output_video_path = os.path.join(OUTPUT_DIR, output_video_filename)
                video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width_vid, height_vid))

                # --- Tulis file .txt ---
                output_txt_filename = f"{filename_base}_output_{current_date}_nonclahe.txt"
                output_txt_path = os.path.join(OUTPUT_DIR, output_txt_filename)

                with open(output_txt_path, "w") as txt_file:
                    txt_file.write(f"LOG: {filename_base}\n")
                    txt_file.write("====================================\n")

                print(f"[DEBUG] Simpan hasil ke: {OUTPUT_DIR}")

            if not stream_control_flags.get(source, True):
                return error_response("Streaming dihentikan oleh user.")
            
            if len(raw_frames) < 3:
                return error_response("Tidak cukup frame untuk proses awal")

            # Buat event loop async untuk menjalankan paralel
            if is_video_file:
                stream_id = source
            else:
                stream_id = source.split('/')[-1].split('.')[0]

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            detect_time = time.time()
            
            # Mencari konfigurasi CLAHE terbaik
            try:
                best_config, time_best_config, best_brisque = loop.run_until_complete(loop.run_in_executor(executor, find_best_clahe_config, raw_frames, det_model_direction, filename_base))
                if not stream_control_flags.get(source, True):
                    return error_response("Streaming dihentikan oleh user.")
            finally:
                loop.close()

            # Step 2: Terapkan CLAHE ke semua frame untuk segmentasi
            if best_config["clip"] is not None:
                preprocessed_frames = [apply_clahe(frame, best_config["clip"], best_config["tile"]) for frame in raw_frames]
                # preprocessed_frames = [apply_clahe_on_roi_only(frame, best_config["clip"], best_config["tile"], roi_mask) for frame in raw_frames]
            else:
                preprocessed_frames = raw_frames

            # Step 3: Segmentasi berdasarkan frame yang sudah di-CLAHE
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                segmentation_result = loop.run_until_complete(loop.run_in_executor(executor, find_segmentation, seg_model, preprocessed_frames))
                if not stream_control_flags.get(source, True):
                    return error_response("Streaming dihentikan")
            finally:
                loop.close()

            if segmentation_result is None or segmentation_result[0] is None:
                return error_response("Gagal segmentasi awal")

            roi_mask, all_polygons = segmentation_result
            segmentation_cache[source] = (roi_mask, all_polygons)

            # Reset ke awal frame
            # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Tracking arah dominan
            dx_list_per_polygon = [list() for _ in all_polygons]
            dy_list_per_polygon = [list() for _ in all_polygons]
            memory_per_polygon = [{} for _ in all_polygons]
            valid_vehicle_ids_per_polygon = [set() for _ in all_polygons]  # ✅ Simpan ID kendaraan valid (bergerak)

            unique_ids = set()
            frame_count = 0
            min_initial_frames = 5

            while stream_control_flags.get(source, True):
                video_timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if video_timestamp_sec > 6.0:
                    enough_data = all(len(valid_vehicle_ids_per_polygon[idx]) >= 3 for idx in range(len(all_polygons)))

                    if enough_data and frame_count >= min_initial_frames:
                        print(f"Minimal 3 kendaraan pada frame ke-{frame_count}")
                        break

                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1

                if best_config["clip"] is not None:
                    # pre_frame = apply_clahe_on_roi_only(frame, best_config["clip"], best_config["tile"], roi_mask)
                    pre_frame = apply_clahe(frame, best_config["clip"], best_config["tile"])
                else:
                    pre_frame = frame

                results = det_model_direction.track(
                    pre_frame,
                    persist=True,
                    conf=0.4,
                    verbose=False,
                    tracker="bytetrack.yaml"
                )[0]

                if results.boxes is not None:
                    for box in results.boxes:
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = xyxy
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        track_id = int(box.id[0]) if box.id is not None else None

                        curr_pos = (cx, cy)

                        idx = vehicle_road_overlap_checking(x1, y1, x2, y2, roi_mask, all_polygons)
                        if idx != -1 and track_id is not None:
                            memory = memory_per_polygon[idx]

                            if track_id not in memory:
                                memory[track_id] = []
                            memory[track_id].append((cx, cy))

                            if len(memory[track_id]) >= 5:
                                # Hitung total perpindahan dari posisi awal ke akhir
                                start_pos = memory[track_id][0]
                                end_pos = memory[track_id][-1]
                                dx = end_pos[0] - start_pos[0]
                                dy = end_pos[1] - start_pos[1]
                                dist = np.linalg.norm([dx, dy])

                                if dist > 5:
                                    if track_id not in valid_vehicle_ids_per_polygon[idx]:
                                        valid_vehicle_ids_per_polygon[idx].add(track_id)
                                        dx_list_per_polygon[idx].append(dx)
                                        dy_list_per_polygon[idx].append(dy)

                # Cek apakah semua polygon punya minimal 9 kendaraan valid
                enough_data = all(len(valid_vehicle_ids_per_polygon[idx]) >= 9 for idx in range(len(all_polygons)))

                if enough_data and frame_count >= min_initial_frames:
                    print(f"[DEBUG] Cukup data arah dominan pada frame ke-{frame_count}")
                    break

            # Setelah dapat dx & dy, tentukan arah panah per polygon
            results = [get_polygon_direction([i], dx_list_per_polygon[i], dy_list_per_polygon[i]) for i in range(len(all_polygons))]
            direction_angle, direction, dominant_vectors = zip(*results)

            print("[DEBUG] Direction Angle = ", direction_angle)
            print("[DEBUG] Direction =", direction)
            print("[DEBUG] Dominant Vectors =", dominant_vectors)

            if direction_angle == 0 or direction == "Unknown" or dominant_vectors == None:
                print("GAGAL")
                return error_response("Gagal menemukan arah dominan arus jalan raya")

            counting_lines = []  # format: [("horizontal", y), ("vertical", x)]
            counted_ids_per_polygon = [set() for _ in all_polygons]  # per polygon

            for poly, dir_label in zip(all_polygons, direction):
                # print("[DEBUG] direction:", direction)
                x, y, w, h = cv2.boundingRect(np.array(poly))
                center_x = x + w // 2
                center_y = y + h // 2

                if dir_label in ["up", "down"]:
                    counting_lines.append(("horizontal", center_y - 30))
                elif dir_label in ["left", "right"]:
                    counting_lines.append(("vertical", center_x))
                else:
                    counting_lines.append((None, None))  # arah tidak diketahui

            # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            position_history = {}
            fps_history = deque(maxlen=10)

            arrow_img = cv2.imread("icon/right-arrow.png", cv2.IMREAD_UNCHANGED)
            rotated_arrows = []
            startsql_time = 0
            endsql_time = 0

            if arrow_img is not None and direction_angle:
                for angle in direction_angle:
                    rotated = rotate_image(arrow_img, angle)
                    rotated_arrows.append(rotated)

            video_timestamp_sec = 0
            while stream_control_flags.get(source, True):
                video_timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                frame_count += 1
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    if is_url_source and retry_count == 0:
                        retry_count += 1
                        cap.release()
                        cap = create_capture(source if is_url_source else os.path.join(UPLOAD_DIR, source))
                        continue
                    else:
                        return error_response("Streaming berhenti atau URL tidak valid")

                
                if best_config["clip"] is not None:
                    # pre_frame = apply_clahe(frame, best_config["clip"], best_config["tile"])
                    pre_frame = apply_clahe_on_roi_only(frame, best_config["clip"], best_config["tile"], roi_mask)
                else:
                    pre_frame = frame
                # pre_frame = frame

                # resized = cv2.resize(pre_frame, (640, 640))
                height, width = pre_frame.shape[:2]
                results = det_model_tracking.track(pre_frame, persist=True, conf=0.4, verbose=False, tracker="bytetrack.yaml")[0]
                # annotated = frame.copy()

                filtered_indices = []
                if results.boxes is not None:
                    boxes = list(results.boxes)
                    boxes = sorted(boxes, key=lambda b: float(b.conf[0]), reverse=True)
                    sorted_indices = sorted(range(len(boxes)), key=lambda i: float(boxes[i].conf[0]), reverse=True)

                    for i in sorted_indices:
                        box_i = boxes[i].xyxy[0].cpu().numpy()
                        class_i = int(boxes[i].cls[0])
                        keep = True

                        for j in filtered_indices:
                            box_j = boxes[j].xyxy[0].cpu().numpy()
                            class_j = int(boxes[j].cls[0])

                            # IOU threshold: lebih ketat untuk motor
                            iou_thresh = 0.4 if class_i == 2 or class_j == 2 else 0.6

                            if compute_iou(box_i, box_j) >= iou_thresh:
                                keep = False
                                break

                        if keep:
                            filtered_indices.append(i)

                    filtered_boxes = [boxes[i] for i in filtered_indices]
                else:
                    filtered_boxes = []


                for box in filtered_boxes:
                    track_bboxes[track_id] = (x1, y1, x2, y2)
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = map(int, xyxy)
                    class_id = int(box.cls[0]) if box.cls is not None else -1
                    YOLO_CLASSES = ["bus", "mobil", "motor", "pikap", "truk"]
                    label = YOLO_CLASSES[class_id] if 0 <= class_id < len(YOLO_CLASSES) else "unknown"

                    track_id = int(box.id[0]) if (box.id is not None and len(box.id) > 0) else -1     

                    if track_id == -1:
                        continue                
                    
                    min_motor = 0.015 * width if 0.015 * width > 15 else 15
                    min_car = 0.025 * width if 0.025 * width > 25 else 25

                    if class_id == 2:
                        if (x2 - x1) < min_motor:
                            continue
                    else:
                        if (x2 - x1) < min_car:
                            continue

                    if y2 > roi_mask.shape[0] or x2 > roi_mask.shape[1]:
                        continue

                    mask_crop = roi_mask[y1:y2, x1:x2]
                    if mask_crop.size == 0 or np.count_nonzero(mask_crop) == 0:
                        continue

                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    curr_pos = (cx, cy)
                    vehicle_positions[track_id] = curr_pos

                    if track_id not in position_history:
                        position_history[track_id] = deque(maxlen=fps_video*2)
                    position_history[track_id].append((cx, cy))

                    # Hitung arah gerak
                    if len(position_history[track_id]) >= 6:
                        start_pos = position_history[track_id][0]
                        end_pos = position_history[track_id][-1]
                        dx = end_pos[0] - start_pos[0]
                        dy = end_pos[1] - start_pos[1]                      

                        poly_idx = vehicle_road_overlap_checking(x1, y1, x2, y2, roi_mask, all_polygons) # poly_idx = -1 jika Kendaraan tidak masuk ke dalam polygon manapun, abaikan atau log
                        if poly_idx != -1 and 0 <= poly_idx < len(dominant_vectors):
                            angle_dir = compute_angle_direction(width, dominant_vectors[poly_idx], track_id, dx, dy)
                        else:
                            angle_dir = "unknown"
                        
                        classification_history[track_id].append(angle_dir)
                        last_seen_frame[track_id] = frame_count
                        votes = list(classification_history[track_id])
                        vote_counts = Counter(votes)
                        total_votes = len(votes)
                        top_vote, top_count = vote_counts.most_common(1)[0]

                        confidence_ratio = top_count / total_votes

                        if confidence_ratio < 0.7:
                            color = (0, 255, 255)
                        else:
                            if top_vote == "searah":
                                color = (0, 255, 0)
                            elif top_vote == "lawan arah":
                                color = (0, 0, 255)
                            else:
                                color = (0, 255, 255)

                        if color == (0, 0, 255):
                            save_direction = direction[poly_idx]

                            if is_video_file:
                                now = datetime.combine(datetime.today().date(), datetime.min.time()) + timedelta(seconds=video_timestamp_sec)
                            else:
                                now = datetime.now()
                                
                            db_cursor.execute("""
                                SELECT id, confidence, timestamp, track_id FROM violations 
                                WHERE bytetrack_id = %s AND source = %s ORDER BY timestamp DESC LIMIT 1
                            """, (track_id, source))
                            existing = db_cursor.fetchone()
                            print("existing", existing)

                            if existing:
                                db_id, old_conf, last_time, stored_track_id = existing
                                time_diff = abs((now - last_time).total_seconds())
                
                                if time_diff <= 120:
                                    if conf > old_conf:
                                        cropped = frame[y1:y2, x1:x2]
                                        filename = f"violation_{label}_{stream_id}_{stored_track_id}.jpg"
                                        path = os.path.join(VIOLATION_DIR, filename)
                                        save_image_async(cropped, path)

                                        db_cursor.execute("""
                                            UPDATE violations SET timestamp=%s, x1=%s, y1=%s, x2=%s, y2=%s,
                                            confidence=%s, image_path=%s, image_name=%s, vehicle=%s WHERE id=%s
                                        """, (now, x1, y1, x2, y2, conf, path, filename, label, db_id))
                                        conn.commit()

                                else:
                                    db_cursor.execute("SELECT MAX(track_id) FROM violations WHERE source = %s", (source,))
                                    result = db_cursor.fetchone()
                                    next_id = (result[0] or 0) + 1
                                    cropped_img = frame[y1:y2, x1:x2]
                                    img_filename = f"violation_{label}_{stream_id}_{next_id}.jpg"
                                    img_path = os.path.join(VIOLATION_DIR, img_filename)
                                    cv2.imwrite(img_path, cropped_img)
                                    db_cursor.execute("""
                                        INSERT INTO violations (track_id, bytetrack_id, timestamp, x1, y1, x2, y2, source, image_path, image_name, confidence, vehicle, true_direction)
                                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                    """, (next_id, track_id, now, x1, y1, x2, y2, source, img_path, img_filename, conf, label, save_direction))
                                    conn.commit()

                                endsql_time = time.time()
   
                            else:
                                db_cursor.execute("SELECT MAX(track_id) FROM violations WHERE source = %s", (source,))
                                result = db_cursor.fetchone()
                                next_id = (result[0] or 0) + 1
                                cropped_img = frame[y1:y2, x1:x2]
                                img_filename = f"violation_{label}_{stream_id}_{next_id}.jpg"
                                img_path = os.path.join(VIOLATION_DIR, img_filename)
                                cv2.imwrite(img_path, cropped_img)
                                db_cursor.execute("""
                                    INSERT INTO violations (track_id, bytetrack_id, timestamp, x1, y1, x2, y2, source, image_path, image_name, confidence, vehicle, true_direction)
                                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                """, (next_id, track_id, now, x1, y1, x2, y2, source, img_path, img_filename, conf, label, save_direction))
                                conn.commit()  
                                endsql_time = time.time()

                        # Gambar bounding box dan ID
                        cv2.rectangle(pre_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(pre_frame, f"ID:{track_id} - {conf:.2f}", (x1 + 1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(pre_frame, f"ID:{track_id} - {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                        # Pembersihan data: hapus track_id yang tidak terlihat dalam 20 frame terakhir
                        expired_ids = [tid for tid, last_frame in last_seen_frame.items() if frame_count - last_frame > 20]
                        for tid in expired_ids:
                            classification_history.pop(tid, None)
                            last_seen_frame.pop(tid, None)
                            vehicle_positions.pop(tid, None)
                            vehicle_stationary_start.pop(tid, None)
                        
                    elif len(position_history[track_id]) < 7:
                        color = (0, 127, 255)
                        cv2.rectangle(pre_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(pre_frame, f"ID:{track_id} - {conf:.2f}", (x1 + 1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(pre_frame, f"ID:{track_id} - {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                overlay = pre_frame.copy()
                for idx, poly in enumerate(all_polygons):
                    cv2.fillPoly(overlay, [poly], color=(0, 0, 255))
                    M = cv2.moments(poly)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        arrow_img_rotated = rotated_arrows[idx]
                        h, w = arrow_img_rotated.shape[:2]
                        overlay = overlay_image_alpha(overlay, arrow_img_rotated, cX - w // 2, cY - h // 2)

                overlayed = cv2.addWeighted(pre_frame, 1.0, overlay, 0.3, 0)
                
                before_write_time = time.time()
                frame_processing_time = before_write_time - start_time - (endsql_time - startsql_time)

                for track_id in position_history:
                    if track_id not in track_bboxes:
                        continue

                    if len(position_history[track_id]) < 2:
                        continue

                    x1, y1, x2, y2 = track_bboxes[track_id]
                    poly_idx = vehicle_road_overlap_checking(x1, y1, x2, y2, roi_mask, all_polygons)
                    if poly_idx == -1:
                        continue

                    axis, value = counting_lines[poly_idx]
                    if axis is None:
                        continue

                    prev_pos = position_history[track_id][-2]
                    curr_pos = position_history[track_id][-1]

                    if axis == "horizontal":
                        if (prev_pos[1] < value <= curr_pos[1]) or (prev_pos[1] > value >= curr_pos[1]):
                            if track_id not in counted_ids_per_polygon[poly_idx]:
                                counted_ids_per_polygon[poly_idx].add(track_id)
                                unique_ids.add(track_id)

                    elif axis == "vertical":
                        if (prev_pos[0] < value <= curr_pos[0]) or (prev_pos[0] > value >= curr_pos[0]):
                            if track_id not in counted_ids_per_polygon[poly_idx]:
                                counted_ids_per_polygon[poly_idx].add(track_id)
                                unique_ids.add(track_id)


                for (axis, value), poly in zip(counting_lines, all_polygons):
                    if axis == "horizontal" and 0 <= value < pre_frame.shape[0]:
                        cv2.line(overlayed, (0, value), (overlayed.shape[1], value), (255, 0, 0), 2)
                    elif axis == "vertical" and 0 <= value < pre_frame.shape[1]:
                        cv2.line(overlayed, (value, 0), (value, overlayed.shape[0]), (255, 0, 0), 2)

                if is_video_file:
                    fps = 1 / frame_processing_time if frame_processing_time > 0 else 0
                    fps_history.append(fps)
                    avg_fps = sum(fps_history) / len(fps_history)
                    cv2.putText(overlayed, f"FPS: {avg_fps:.2f}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                if video_writer is not None:
                    video_writer.write(overlayed)

                overlayed = cv2.resize(overlayed, display_size, interpolation=cv2.INTER_AREA)
                _, jpeg = cv2.imencode('.jpg', overlayed)

                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

        finally:
            if detect_time is not None:
                end_detect_time = time.time()
                detect_time_str = datetime.fromtimestamp(detect_time).strftime('%Y-%m-%d %H:%M:%S')
                end_detect_time_str = datetime.fromtimestamp(end_detect_time).strftime('%Y-%m-%d %H:%M:%S')
                if is_video_file:
                    with open(output_txt_path, "a") as txt_file:
                        db_cursor.execute("""
                                SELECT COUNT(*) FROM violations
                                WHERE source = %s AND timestamp BETWEEN %s AND %s
                            """, (source, detect_time_str, end_detect_time_str))
                        total_violations = db_cursor.fetchone()[0]
                        if best_config:
                            txt_file.write(f"Konfigurasi CLAHE yang didapat = {best_config} dengan skor BRISQUE {best_brisque}\n")
                            txt_file.write(f"Waktu pencarian konfigurasi CLAHE = {time_best_config}\n")
                        txt_file.write(f"1) Total unique track ID = {len(unique_ids)}\n")
                        txt_file.write(f"2) Total pelanggar = {total_violations}\n")
                        

            db_cursor.close()
            conn.close()
            if is_video_file:
                video_writer_stop = True
                if video_writer_thread is not None:
                    video_writer_thread.join()
                print(f"[DEBUG] Video hasil disimpan di: {output_video_path}")
    return StreamingResponse(generate_stream(), media_type='multipart/x-mixed-replace; boundary=frame')


@app.get("/violations")
def get_violations(
    page: int = 1,
    limit: int = 15,
    sort_by: str = "timestamp",
    order: str = "desc",
    after: str = None,
    conn = Depends(get_db_connection),
    source: str = None
):
    cursor = conn.cursor()

    allowed_sort = ["track_id", "timestamp", "source"]
    sort_by = sort_by if sort_by in allowed_sort else "timestamp"
    order = "ASC" if order.lower() == "asc" else "DESC"
    offset = (page - 1) * limit

    query = "SELECT id, track_id, timestamp, source, image_name FROM violations"
    params = []
    conditions = []

    if after:
        try:
            # Parse 'after' string as UTC datetime, convert to WIB timezone string
            after_dt_utc = datetime.strptime(after, '%Y-%m-%d %H:%M:%S')
            utc = pytz.utc
            wib = pytz.timezone('Asia/Jakarta')
            after_dt_utc = utc.localize(after_dt_utc)
            after_dt_wib = after_dt_utc.astimezone(wib)
            after_str = after_dt_wib.strftime('%Y-%m-%d %H:%M:%S')
            conditions.append("timestamp >= %s")
            params.append(after_str)
        except Exception as e:
            print(f"Error parsing 'after' parameter: {e}")

    if source:
        conditions.append("source = %s")
        params.append(source)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += f" ORDER BY {sort_by} {order} LIMIT %s OFFSET %s"

    params.extend([limit, offset])
    cursor.execute(query, params)
    records = cursor.fetchall()
    cursor.close()

    return [
        {
            "id": r[0],
            "track_id": r[1],
            "timestamp": r[2].strftime('%Y-%m-%d %H:%M:%S'),
            "source": r[3],
            "image_url": f"/violation_images/{r[4]}"
        }
        for r in records
    ]

@app.get("/violations/count")
def get_total_count(conn = Depends(get_db_connection)):
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM violations")
        result = cursor.fetchone()[0]
        cursor.close()
        return {"total": result}
    except Exception as e:
        return {"total": 0, "error": str(e)}

@app.post("/stop-stream")
async def stop_stream(payload: dict):
    source = payload.get("source")
    if not source:
        return {"error": "source tidak diberikan"}

    # Matikan flag
    stream_control_flags[source] = False
    segmentation_cache.pop(source, None)
    # RESET semua state yg berkaitan dgn stream
    global memory_per_polygon, arrow_directions, total_counts, detected_vehicles_per_polygon, unique_ids, frame_count, enough_data, best_config
    memory_per_polygon = {}
    arrow_directions = []
    total_counts = []
    detected_vehicles_per_polygon = []   
    unique_ids = set()
    frame_count = 0 
    enough_data = False
    # clahe_config_cache.pop(source, None)
    print(f"Stop signal diterima untuk source: {source}")

    return {"message": "Streaming dihentikan"}