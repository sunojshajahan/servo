import time
import serial
import numpy as np
import cv2
from ultralytics import YOLO
import depthai as dai

# =========================================================
# SPEED SETTINGS
# =========================================================
INFER_EVERY_N_FRAMES = 3        # 2 или 3
CAPTURE_W = 640
CAPTURE_H = 480
OAK_FPS = 30

# =========================================================
# MODEL
# =========================================================
MODEL_PATH = "runs/detect/train/weights/best.pt"
CONF = 0.5
IOU = 0.5
TRACKER_CFG = "bytetrack.yaml"

# =========================================================
# ROI / LINE / DRAW
# =========================================================
ROI_HEIGHT_FRACTION = 0.8
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
# TARGET LOGIC
# =========================================================
TARGET_MISSING_LIMIT = 12
REACQUIRE_COOLDOWN_FRAMES = 2

# =========================================================
# SMART SENDING (X-only)
# =========================================================
MIN_SEND_INTERVAL_SEC = 0.10   # ~10 Гц
CX_DELTA_PX = 10               # слать только если cx изменился >= N пикселей

EMA_ENABLED = True
EMA_ALPHA = 0.35               # 0..1

# =========================================================
# UI
# =========================================================
WINDOW_NAME = "Weed tracking (OAK + optimized)"
SHOW_FPS = True


def send_line(ser, msg: str):
    if ser:
        ser.write((msg.strip() + "\n").encode("utf-8"))


def create_oak_pipeline():
    pipeline = dai.Pipeline()

    cam = pipeline.createColorCamera()
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setFps(OAK_FPS)

    cam.setPreviewSize(CAPTURE_W, CAPTURE_H)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    xout = pipeline.createXLinkOut()
    xout.setStreamName("rgb")
    cam.preview.link(xout.input)

    return pipeline


def main():
    model = YOLO(MODEL_PATH)

    ser = None
    if SERIAL_ENABLED:
        ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0.01)
        time.sleep(2.0)  # Arduino reset

    handled_ids = set()
    current_target_id = None
    target_missing = 0
    reacquire_cd = 0

    frame_idx = 0
    sent_size = False

    last_target_id = None
    last_target_cx = None
    last_target_cy = None

    cx_ema = None
    last_sent_cx = None
    last_send_time = 0.0

    last_results = None

    # FPS counter
    t_prev = time.time()
    fps = 0.0

    pipeline = create_oak_pipeline()
    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=True)

        print("ESC to exit")
        while True:
            frame = q_rgb.get().getCvFrame()
            frame_idx += 1

            h, w = frame.shape[:2]

            if SERIAL_ENABLED and not sent_size:
                send_line(ser, f"SIZE,{w},{h}")
                sent_size = True

            tol = LINE_TOL_PIX if LINE_TOL_PIX is not None else max(2, int(h * LINE_TOL_FRAC))

            # ROI
            roi_h = int(h * ROI_HEIGHT_FRACTION)
            y1 = (h - roi_h) // 2
            y2 = y1 + roi_h
            roi = frame[y1:y2, :]

            # inference only every N frames
            did_infer = False
            if frame_idx % INFER_EVERY_N_FRAMES == 0:
                last_results = model.track(
                    roi,
                    conf=CONF,
                    iou=IOU,
                    persist=True,
                    tracker=TRACKER_CFG,
                    verbose=False
                )
                did_infer = True

            results = last_results

            # OUTPUT for drawing
            output = frame.copy()
            if DIM_OUTSIDE:
                dark = np.zeros_like(output)
                output = cv2.addWeighted(dark, DIM_ALPHA, output, 1 - DIM_ALPHA, 0)
                output[y1:y2, :] = frame[y1:y2, :]

            mid_y = (y1 + y2) // 2
            cv2.rectangle(output, (0, y1), (w, y2), (0, 255, 0), 2)
            cv2.line(output, (0, mid_y), (w, mid_y), (0, 255, 255), 2)
            cv2.rectangle(output, (0, mid_y - tol), (w, mid_y + tol), (0, 255, 255), 1)

            # between inferences: smart-send last target
            if not did_infer:
                if last_target_id is not None and last_target_cx is not None:
                    now = time.time()
                    if (now - last_send_time) >= MIN_SEND_INTERVAL_SEC:
                        if last_sent_cx is None or abs(int(last_target_cx) - int(last_sent_cx)) >= CX_DELTA_PX:
                            send_line(ser, f"TARGET,{last_target_id},{int(last_target_cx)},{int(last_target_cy)}")
                            last_send_time = now
                            last_sent_cx = int(last_target_cx)

                # FPS calc + show
                if SHOW_FPS:
                    t_now = time.time()
                    dt = t_now - t_prev
                    if dt > 0:
                        fps = 1.0 / dt
                    t_prev = t_now
                    cv2.putText(output, f"FPS:{fps:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                cv2.imshow(WINDOW_NAME, output)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break
                continue

            # fresh detections/tracks
            trigger_ids = []
            target_found = False
            target_bbox = None
            target_cxcy = None

            closest_dist = None
            closest_data = None  # (id, cx, cy, BX1,BY1,BX2,BY2)

            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                ids = boxes.id
                xyxy = boxes.xyxy

                if ids is not None and xyxy is not None:
                    ids = ids.cpu().numpy().astype(int)
                    xyxy = xyxy.cpu().numpy().astype(int)

                    if reacquire_cd > 0:
                        reacquire_cd -= 1

                    for i, (bx1, by1, bx2, by2) in enumerate(xyxy):
                        track_id = ids[i]

                        BX1, BX2 = int(bx1), int(bx2)
                        BY1, BY2 = int(by1 + y1), int(by2 + y1)

                        cx = (BX1 + BX2) // 2
                        cy = (BY1 + BY2) // 2

                        if track_id in handled_ids:
                            continue

                        # draw boxes
                        cv2.rectangle(output, (BX1, BY1), (BX2, BY2), (255, 0, 0), 2)
                        cv2.putText(output, f"ID:{track_id}", (BX1, max(20, BY1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                        # trigger
                        if abs(cy - mid_y) <= tol:
                            trigger_ids.append(track_id)

                        dist = abs(cy - mid_y)

                        # keep current target
                        if current_target_id is not None:
                            if track_id == current_target_id:
                                target_found = True
                                target_bbox = (BX1, BY1, BX2, BY2)
                                target_cxcy = (cx, cy)
                        else:
                            # choose only from below
                            if reacquire_cd == 0:
                                if cy < mid_y:
                                    continue
                                if closest_dist is None or dist < closest_dist:
                                    closest_dist = dist
                                    closest_data = (track_id, cx, cy, BX1, BY1, BX2, BY2)

            # acquire new target
            if current_target_id is None and closest_data is not None:
                current_target_id = closest_data[0]
                target_found = True
                target_cxcy = (closest_data[1], closest_data[2])
                target_bbox = (closest_data[3], closest_data[4], closest_data[5], closest_data[6])
                reacquire_cd = REACQUIRE_COOLDOWN_FRAMES

            # anti-stuck
            if current_target_id is not None:
                if not target_found:
                    target_missing += 1
                    if target_missing >= TARGET_MISSING_LIMIT:
                        current_target_id = None
                        target_missing = 0
                else:
                    target_missing = 0

            # draw target
            if current_target_id is not None and target_bbox is not None:
                BX1, BY1, BX2, BY2 = target_bbox
                cv2.rectangle(output, (BX1, BY1), (BX2, BY2), (0, 0, 255), 3)
                cv2.putText(output, f"TARGET:{current_target_id}", (BX1, max(25, BY1 - 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # update last target + smart send
            if current_target_id is not None and target_found and target_cxcy is not None:
                raw_cx, raw_cy = target_cxcy

                if EMA_ENABLED:
                    if cx_ema is None:
                        cx_ema = float(raw_cx)
                    else:
                        cx_ema = EMA_ALPHA * float(raw_cx) + (1.0 - EMA_ALPHA) * cx_ema
                    use_cx = int(round(cx_ema))
                else:
                    use_cx = int(raw_cx)

                last_target_id = current_target_id
                last_target_cx = use_cx
                last_target_cy = int(raw_cy)

                now = time.time()
                if (now - last_send_time) >= MIN_SEND_INTERVAL_SEC:
                    if last_sent_cx is None or abs(int(last_target_cx) - int(last_sent_cx)) >= CX_DELTA_PX:
                        send_line(ser, f"TARGET,{last_target_id},{int(last_target_cx)},{int(last_target_cy)}")
                        last_send_time = now
                        last_sent_cx = int(last_target_cx)

            # triggers
            if trigger_ids:
                handled_ids.update(trigger_ids)
                if current_target_id in trigger_ids:
                    send_line(ser, f"FIRE,{current_target_id}")
                    current_target_id = None
                    target_missing = 0
                    reacquire_cd = REACQUIRE_COOLDOWN_FRAMES

            # FPS
            if SHOW_FPS:
                t_now = time.time()
                dt = t_now - t_prev
                if dt > 0:
                    fps = 1.0 / dt
                t_prev = t_now
                cv2.putText(output, f"FPS:{fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow(WINDOW_NAME, output)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break

    if ser:
        ser.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
