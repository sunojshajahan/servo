import cv2
from ultralytics import YOLO
import numpy as np
import time
import serial

# =========================================================
# PERFORMANCE / DEBUG
# =========================================================
DEBUG_SHOW = True            # False = быстрее
DEBUG_DRAW = True            # False = быстрее
PRINT_TARGET = False         # print тормозит, лучше False

INFER_EVERY_N_FRAMES = 1     # 1=каждый кадр, 2=через кадр...
SEND_EVERY_N_FRAMES = 12      # 3-6 обычно норм

CAPTURE_W = 960
CAPTURE_H = 540
cv2.setNumThreads(1)

# =========================================================
# MODEL / VIDEO
# =========================================================
MODEL_PATH = "runs/detect/train/weights/best.pt"
VIDEO_SOURCE = "output.mp4"   # 0 — камера
CONF = 0.5
IOU = 0.5
TRACKER_CFG = "bytetrack.yaml"

# =========================================================
# ROI / CENTER LINE
# =========================================================
ROI_HEIGHT_FRACTION = 0.4
DIM_OUTSIDE = True
DIM_ALPHA = 0.35

LINE_TOL_PIX = None
LINE_TOL_FRAC = 0.015

# =========================================================
# SERIAL
# =========================================================
SERIAL_ENABLED = True
SERIAL_PORT = "/dev/ttyACM0"
BAUD = 115200

# =========================================================
# TARGET LOGIC (движение снизу -> вверх)
# =========================================================
TARGET_MISSING_LIMIT = 10   # кадров, после которых сброс цели


def send_line(ser, msg: str):
    if ser:
        ser.write((msg.strip() + "\n").encode("utf-8"))


def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть видео/камеру")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)

    ser = None
    if SERIAL_ENABLED:
        ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0.01)
        time.sleep(2.0)

    handled_ids = set()
    current_target_id = None
    target_missing_frames = 0
    frame_idx = 0
    sent_size = False
    last_results = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        h, w = frame.shape[:2]

        if SERIAL_ENABLED and not sent_size:
            send_line(ser, f"SIZE,{w},{h}")
            sent_size = True

        tol = LINE_TOL_PIX if LINE_TOL_PIX is not None else max(2, int(h * LINE_TOL_FRAC))

        # --- ROI ---
        roi_h = int(h * ROI_HEIGHT_FRACTION)
        y1 = (h - roi_h) // 2
        y2 = y1 + roi_h
        roi = frame[y1:y2, :]

        # --- Track (можно реже) ---
        if frame_idx % INFER_EVERY_N_FRAMES == 0:
            last_results = model.track(
                roi,
                conf=CONF,
                iou=IOU,
                persist=True,
                tracker=TRACKER_CFG,
                verbose=False
            )
        results = last_results

        # --- Visualization ---
        output = frame
        if DEBUG_SHOW:
            output = frame.copy()
            if DIM_OUTSIDE:
                dark = np.zeros_like(output)
                output = cv2.addWeighted(dark, DIM_ALPHA, output, 1 - DIM_ALPHA, 0)
                output[y1:y2, :] = frame[y1:y2, :]

        mid_y = (y1 + y2) // 2

        if DEBUG_SHOW and DEBUG_DRAW:
            cv2.rectangle(output, (0, y1), (w, y2), (0, 255, 0), 2)
            cv2.line(output, (0, mid_y), (w, mid_y), (0, 255, 255), 2)
            cv2.rectangle(output, (0, mid_y - tol), (w, mid_y + tol), (0, 255, 255), 1)

        # --- Frame logic ---
        trigger_ids = []
        target_found = False
        target_bbox = None
        target_cxcy = None

        closest_dist = None
        closest_data = None  # (id, cx, cy, BX1,BY1,BX2,BY2)

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            ids = boxes.id
            xyxy = boxes.xyxy

            if ids is not None and xyxy is not None:
                ids = ids.cpu().numpy().astype(int)
                xyxy = xyxy.cpu().numpy().astype(int)

                for i, (bx1, by1, bx2, by2) in enumerate(xyxy):
                    track_id = ids[i]

                    BX1, BX2 = int(bx1), int(bx2)
                    BY1, BY2 = int(by1 + y1), int(by2 + y1)
                    cx = (BX1 + BX2) // 2
                    cy = (BY1 + BY2) // 2

                    if DEBUG_SHOW and DEBUG_DRAW:
                        cv2.rectangle(output, (BX1, BY1), (BX2, BY2), (255, 0, 0), 2)
                        cv2.putText(output, f"ID:{track_id}", (BX1, max(20, BY1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    if track_id in handled_ids:
                        continue

                    # TRIGGER: пересёк линию
                    if abs(cy - mid_y) <= tol:
                        trigger_ids.append(track_id)

                    dist = abs(cy - mid_y)

                    # TARGET logic
                    if current_target_id is not None:
                        if track_id == current_target_id:
                            target_found = True
                            target_bbox = (BX1, BY1, BX2, BY2)
                            target_cxcy = (cx, cy)
                    else:
                        # движение снизу->вверх: кандидаты только снизу
                        if cy < mid_y:
                            continue

                        if closest_dist is None or dist < closest_dist:
                            closest_dist = dist
                            closest_data = (track_id, cx, cy, BX1, BY1, BX2, BY2)

        # назначаем цель
        if current_target_id is None and closest_data is not None:
            current_target_id = closest_data[0]
            target_found = True
            target_cxcy = (closest_data[1], closest_data[2])
            target_bbox = (closest_data[3], closest_data[4], closest_data[5], closest_data[6])

        # анти-залипание
        if current_target_id is not None:
            if not target_found:
                target_missing_frames += 1
                if target_missing_frames >= TARGET_MISSING_LIMIT:
                    current_target_id = None
                    target_missing_frames = 0
            else:
                target_missing_frames = 0

        # выделение TARGET
        if DEBUG_SHOW and DEBUG_DRAW and current_target_id and target_bbox:
            BX1, BY1, BX2, BY2 = target_bbox
            cv2.rectangle(output, (BX1, BY1), (BX2, BY2), (0, 0, 255), 3)
            cv2.putText(output, f"TARGET:{current_target_id}", (BX1, max(25, BY1 - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # отправка TARGET (cx,cy) — Arduino использует только cx
        if SERIAL_ENABLED and current_target_id and target_found:
            if frame_idx % SEND_EVERY_N_FRAMES == 0:
                cx, cy = target_cxcy
                send_line(ser, f"TARGET,{current_target_id},{cx},{cy}")
                if PRINT_TARGET:
                    print(f"[TARGET] ID={current_target_id} center=({cx},{cy})")

        # обработка TRIGGER без паузы
        if trigger_ids:
            handled_ids.update(trigger_ids)
            if current_target_id in trigger_ids:
                send_line(ser, f"FIRE,{current_target_id}")
                current_target_id = None
                target_missing_frames = 0

        # показ / выход
        if DEBUG_SHOW:
            cv2.imshow("Weed tracking (no pause)", output)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    cap.release()
    if DEBUG_SHOW:
        cv2.destroyAllWindows()
    if ser:
        ser.close()


if __name__ == "__main__":
    main()
