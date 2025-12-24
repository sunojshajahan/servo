import os
import json
import time
import cv2
import numpy as np
import depthai as dai
from ultralytics import YOLO

# =========================================================
# DEBUG / PERFORMANCE
# =========================================================
DEBUG_SHOW = True
DEBUG_DRAW = True
PRINT_TARGET = False

INFER_EVERY_N_FRAMES = 1

CAPTURE_W = 960
CAPTURE_H = 540
CAM_FPS = 30

cv2.setNumThreads(1)

# =========================================================
# MODEL
# =========================================================
MODEL_PATH = "runs/detect/train/weights/best.pt"
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
# TARGET LOGIC
# =========================================================
TARGET_MISSING_LIMIT = 10

# =========================================================
# SERVO (DIRECT GPIO)
# =========================================================
SERVO_ENABLED = True
SERVO_PIN_BOARD = 15
SERVO_FREQ_HZ = 50

SERVO_MIN_PULSE_MS = 0.5
SERVO_MAX_PULSE_MS = 2.5

SERVO_MIN_ANGLE = 0
SERVO_MAX_ANGLE = 180
SERVO_CENTER_ANGLE = 90

SERVO_ON_NO_TARGET = "hold"  # "hold" или "center"

# --- RUN profile (during normal tracking)
SERVO_UPDATE_EVERY_N_FRAMES_RUN = 2     # 1 = fastest
SERVO_SMOOTH_ALPHA_RUN = 0.25
SERVO_MAX_STEP_DEG_RUN = 10.0

# deadzone around target center to reduce jitter (pixels)
SERVO_DEADZONE_PX = 6

# --- CALIB profile (during calibration)
SERVO_UPDATE_EVERY_N_FRAMES_CALIB = 1
SERVO_SMOOTH_ALPHA_CALIB = 0.60
SERVO_MAX_STEP_DEG_CALIB = 60.0  # big => almost no step limiting

# manual step sizes in calibration mode
MANUAL_STEP_SMALL = 0.25   # a/d
MANUAL_STEP_BIG = 1.0      # A/D

# =========================================================
# CALIBRATION
# =========================================================
CALIB_FILE = "servo_calibration.json"

# "offset_gain" or "linear"
CALIB_MODE = "offset_gain"

# IMPORTANT: start inverted direction (your issue)
SERVO_INVERT_X = True

SERVO_OFFSET_DEG = 0.0
SERVO_GAIN = 1.00

CAL_K = 0.0
CAL_B = SERVO_CENTER_ANGLE

# =========================================================
# Headless auto-detect
# =========================================================
if not os.environ.get("DISPLAY"):
    DEBUG_SHOW = False


# -------------------------
# Servo controller
# -------------------------
class ServoController:
    def __init__(self, pin_board, freq_hz, min_pulse_ms, max_pulse_ms, min_angle, max_angle):
        self.pin = pin_board
        self.freq = freq_hz
        self.min_pulse_ms = float(min_pulse_ms)
        self.max_pulse_ms = float(max_pulse_ms)
        self.min_angle = float(min_angle)
        self.max_angle = float(max_angle)
        self._gpio = None
        self._pwm = None
        self._last_angle = None

    def start(self):
        import Jetson.GPIO as GPIO
        self._gpio = GPIO
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)
        self._pwm = GPIO.PWM(self.pin, self.freq)
        self._pwm.start(0.0)
        time.sleep(0.05)

    def _angle_to_duty(self, angle):
        angle = max(self.min_angle, min(self.max_angle, float(angle)))
        pulse_ms = self.min_pulse_ms + (angle - self.min_angle) * (
            (self.max_pulse_ms - self.min_pulse_ms) / (self.max_angle - self.min_angle)
        )
        period_ms = 1000.0 / self.freq
        return (pulse_ms / period_ms) * 100.0

    def set_angle(self, angle):
        if self._pwm is None:
            return
        duty = self._angle_to_duty(angle)
        self._pwm.ChangeDutyCycle(duty)
        self._last_angle = float(angle)

    def stop(self):
        if self._pwm is not None:
            try:
                self._pwm.ChangeDutyCycle(0.0)
                time.sleep(0.02)
                self._pwm.stop()
            except Exception:
                pass
        if self._gpio is not None:
            try:
                self._gpio.cleanup()
            except Exception:
                pass


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def smooth_angle(prev, target, alpha, max_step):
    if prev is None:
        return target
    sm = prev + alpha * (target - prev)
    if max_step is not None and max_step > 0:
        delta = sm - prev
        if abs(delta) > max_step:
            sm = prev + np.sign(delta) * max_step
    return sm


def map_cx_to_angle(cx, w):
    global CALIB_MODE, SERVO_INVERT_X, SERVO_OFFSET_DEG, SERVO_GAIN, CAL_K, CAL_B

    if w <= 1:
        return SERVO_CENTER_ANGLE

    if CALIB_MODE == "linear":
        angle = CAL_K * float(cx) + CAL_B
        return clamp(angle, SERVO_MIN_ANGLE, SERVO_MAX_ANGLE)

    # offset_gain
    x = (cx - (w / 2)) / (w / 2)   # [-1..+1]
    if SERVO_INVERT_X:
        x = -x
    x *= SERVO_GAIN

    half_range = (SERVO_MAX_ANGLE - SERVO_MIN_ANGLE) / 2.0
    angle = SERVO_CENTER_ANGLE + x * half_range + SERVO_OFFSET_DEG
    return clamp(angle, SERVO_MIN_ANGLE, SERVO_MAX_ANGLE)


def save_calibration():
    data = {
        "CALIB_MODE": CALIB_MODE,
        "SERVO_INVERT_X": SERVO_INVERT_X,
        "SERVO_OFFSET_DEG": SERVO_OFFSET_DEG,
        "SERVO_GAIN": SERVO_GAIN,
        "CAL_K": CAL_K,
        "CAL_B": CAL_B,
        "SERVO_MIN_PULSE_MS": SERVO_MIN_PULSE_MS,
        "SERVO_MAX_PULSE_MS": SERVO_MAX_PULSE_MS,
    }
    with open(CALIB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[CALIB] Saved -> {CALIB_FILE}")


def load_calibration():
    global CALIB_MODE, SERVO_INVERT_X, SERVO_OFFSET_DEG, SERVO_GAIN, CAL_K, CAL_B
    global SERVO_MIN_PULSE_MS, SERVO_MAX_PULSE_MS

    if not os.path.exists(CALIB_FILE):
        print(f"[CALIB] File not found: {CALIB_FILE}")
        return False

    with open(CALIB_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    CALIB_MODE = data.get("CALIB_MODE", CALIB_MODE)
    SERVO_INVERT_X = data.get("SERVO_INVERT_X", SERVO_INVERT_X)
    SERVO_OFFSET_DEG = float(data.get("SERVO_OFFSET_DEG", SERVO_OFFSET_DEG))
    SERVO_GAIN = float(data.get("SERVO_GAIN", SERVO_GAIN))
    CAL_K = float(data.get("CAL_K", CAL_K))
    CAL_B = float(data.get("CAL_B", CAL_B))

    SERVO_MIN_PULSE_MS = float(data.get("SERVO_MIN_PULSE_MS", SERVO_MIN_PULSE_MS))
    SERVO_MAX_PULSE_MS = float(data.get("SERVO_MAX_PULSE_MS", SERVO_MAX_PULSE_MS))

    print(f"[CALIB] Loaded <- {CALIB_FILE}")
    return True


def fit_linear(points):
    if len(points) < 2:
        return None
    xs = np.array([p["cx"] for p in points], dtype=np.float64)
    ys = np.array([p["angle"] for p in points], dtype=np.float64)
    A = np.vstack([xs, np.ones_like(xs)]).T
    k, b = np.linalg.lstsq(A, ys, rcond=None)[0]
    return float(k), float(b)


def get_bgr_frame(img_frame):
    try:
        return img_frame.getCvFrame()
    except Exception:
        data = img_frame.getFrame()
        h = img_frame.getHeight()
        w = img_frame.getWidth()
        return data.reshape((h, w, 3))


def draw_text(img, lines, x=10, y=20, dy=22, scale=0.6):
    for i, t in enumerate(lines):
        cv2.putText(img, t, (x, y + i * dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)


def main():
    global CALIB_MODE, SERVO_INVERT_X, SERVO_OFFSET_DEG, SERVO_GAIN, CAL_K, CAL_B
    global SERVO_SMOOTH_ALPHA_RUN, SERVO_MAX_STEP_DEG_RUN
    global SERVO_SMOOTH_ALPHA_CALIB, SERVO_MAX_STEP_DEG_CALIB

    model = YOLO(MODEL_PATH)

    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build()
        video_queue = cam.requestOutput(
            size=(CAPTURE_W, CAPTURE_H),
            type=dai.ImgFrame.Type.BGR888p,
            resizeMode=dai.ImgResizeMode.LETTERBOX,
            fps=CAM_FPS,
        ).createOutputQueue()

        servo = None
        last_servo_angle = SERVO_CENTER_ANGLE

        if SERVO_ENABLED:
            servo = ServoController(
                pin_board=SERVO_PIN_BOARD,
                freq_hz=SERVO_FREQ_HZ,
                min_pulse_ms=SERVO_MIN_PULSE_MS,
                max_pulse_ms=SERVO_MAX_PULSE_MS,
                min_angle=SERVO_MIN_ANGLE,
                max_angle=SERVO_MAX_ANGLE,
            )
            servo.start()
            servo.set_angle(SERVO_CENTER_ANGLE)

        handled_ids = set()
        current_target_id = None
        target_missing_frames = 0
        frame_idx = 0
        last_results = None

        # calibration state
        calib_enabled = False
        calib_points = []
        manual_angle = SERVO_CENTER_ANGLE

        pipeline.start()

        try:
            while pipeline.isRunning():
                frame_idx += 1

                video_in = video_queue.get()
                frame = get_bgr_frame(video_in)
                h, w = frame.shape[:2]

                tol = LINE_TOL_PIX if LINE_TOL_PIX is not None else max(2, int(h * LINE_TOL_FRAC))

                roi_h = int(h * ROI_HEIGHT_FRACTION)
                y1 = (h - roi_h) // 2
                y2 = y1 + roi_h
                roi = frame[y1:y2, :]

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
                    cv2.drawMarker(output, (w // 2, h // 2), (0, 255, 255),
                                   markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

                trigger_ids = []
                target_found = False
                target_bbox = None
                target_cxcy = None

                closest_dist = None
                closest_data = None

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
                                cv2.circle(output, (cx, cy), 4, (0, 255, 255), -1)

                            if track_id in handled_ids:
                                continue

                            if abs(cy - mid_y) <= tol:
                                trigger_ids.append(track_id)

                            dist = abs(cy - mid_y)

                            if current_target_id is not None:
                                if track_id == current_target_id:
                                    target_found = True
                                    target_bbox = (BX1, BY1, BX2, BY2)
                                    target_cxcy = (cx, cy)
                            else:
                                if cy < mid_y:
                                    continue
                                if closest_dist is None or dist < closest_dist:
                                    closest_dist = dist
                                    closest_data = (track_id, cx, cy, BX1, BY1, BX2, BY2)

                if current_target_id is None and closest_data is not None:
                    current_target_id = closest_data[0]
                    target_found = True
                    target_cxcy = (closest_data[1], closest_data[2])
                    target_bbox = (closest_data[3], closest_data[4], closest_data[5], closest_data[6])

                if current_target_id is not None:
                    if not target_found:
                        target_missing_frames += 1
                        if target_missing_frames >= TARGET_MISSING_LIMIT:
                            current_target_id = None
                            target_missing_frames = 0
                    else:
                        target_missing_frames = 0

                if DEBUG_SHOW and DEBUG_DRAW and current_target_id and target_bbox:
                    BX1, BY1, BX2, BY2 = target_bbox
                    cv2.rectangle(output, (BX1, BY1), (BX2, BY2), (0, 0, 255), 3)
                    cv2.putText(output, f"TARGET:{current_target_id}", (BX1, max(25, BY1 - 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # -------------------------
                # SERVO UPDATE
                # -------------------------
                if SERVO_ENABLED and servo is not None:
                    if calib_enabled:
                        # update every frame for smooth manual aiming
                        if frame_idx % SERVO_UPDATE_EVERY_N_FRAMES_CALIB == 0:
                            servo.set_angle(clamp(manual_angle, SERVO_MIN_ANGLE, SERVO_MAX_ANGLE))
                            last_servo_angle = manual_angle
                    else:
                        if frame_idx % SERVO_UPDATE_EVERY_N_FRAMES_RUN == 0:
                            if current_target_id and target_found and target_cxcy is not None:
                                cx = int(target_cxcy[0])

                                # deadzone in pixels
                                if abs(cx - (w // 2)) <= SERVO_DEADZONE_PX:
                                    target_angle = SERVO_CENTER_ANGLE + SERVO_OFFSET_DEG
                                else:
                                    target_angle = map_cx_to_angle(cx, w)

                                new_angle = smooth_angle(
                                    last_servo_angle, target_angle,
                                    alpha=SERVO_SMOOTH_ALPHA_RUN,
                                    max_step=SERVO_MAX_STEP_DEG_RUN
                                )
                                servo.set_angle(new_angle)
                                last_servo_angle = new_angle
                            else:
                                if SERVO_ON_NO_TARGET == "center":
                                    new_angle = smooth_angle(
                                        last_servo_angle, SERVO_CENTER_ANGLE,
                                        alpha=SERVO_SMOOTH_ALPHA_RUN,
                                        max_step=SERVO_MAX_STEP_DEG_RUN
                                    )
                                    servo.set_angle(new_angle)
                                    last_servo_angle = new_angle

                # trigger handling
                if trigger_ids:
                    handled_ids.update(trigger_ids)
                    if current_target_id in trigger_ids:
                        current_target_id = None
                        target_missing_frames = 0

                # -------------------------
                # UI + KEYS
                # -------------------------
                if DEBUG_SHOW:
                    cx_dbg = target_cxcy[0] if (target_cxcy is not None) else None
                    lines = [
                        f"Mode: {'CALIBRATION' if calib_enabled else 'RUN'} | CALIB_MODE={CALIB_MODE}",
                        f"invert={SERVO_INVERT_X} offset={SERVO_OFFSET_DEG:+.2f} gain={SERVO_GAIN:.3f}",
                        f"linear: k={CAL_K:.6f} b={CAL_B:.2f} | points={len(calib_points)}",
                        f"cx={cx_dbg}  angle={last_servo_angle:.2f}  manual={manual_angle:.2f}",
                        "Keys: c calib | i invert | [ ] offset(0.5) | { } offset(2)",
                        "      -/= gain(0.01) | _/+ gain(0.05)",
                        "      a/d manual(0.25) | A/D manual(1.0)",
                        "      p add point | f fit linear | r clear | s save | l load | q/ESC quit"
                    ]
                    draw_text(output, lines, 10, 22, 22, 0.55)
                    cv2.imshow("OAK + YOLO + Servo (Smooth Calib)", output)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break

                    if key == ord("c"):
                        calib_enabled = not calib_enabled
                        manual_angle = float(last_servo_angle if last_servo_angle is not None else SERVO_CENTER_ANGLE)
                        print(f"[CALIB] calibration mode = {calib_enabled}")

                    if key == ord("i"):
                        SERVO_INVERT_X = not SERVO_INVERT_X
                        print(f"[CALIB] SERVO_INVERT_X={SERVO_INVERT_X}")

                    # offset
                    if key == ord("["):
                        SERVO_OFFSET_DEG -= 0.5
                    if key == ord("]"):
                        SERVO_OFFSET_DEG += 0.5
                    if key == ord("{"):
                        SERVO_OFFSET_DEG -= 2.0
                    if key == ord("}"):
                        SERVO_OFFSET_DEG += 2.0

                    # gain
                    if key == ord("-"):
                        SERVO_GAIN = max(0.10, SERVO_GAIN - 0.01)
                    if key == ord("="):
                        SERVO_GAIN = min(3.00, SERVO_GAIN + 0.01)
                    if key == ord("_"):
                        SERVO_GAIN = max(0.10, SERVO_GAIN - 0.05)
                    if key == ord("+"):
                        SERVO_GAIN = min(3.00, SERVO_GAIN + 0.05)

                    # manual aim (only meaningful in calib mode)
                    if key == ord("a"):
                        manual_angle -= MANUAL_STEP_SMALL
                    if key == ord("d"):
                        manual_angle += MANUAL_STEP_SMALL
                    if key == ord("A"):
                        manual_angle -= MANUAL_STEP_BIG
                    if key == ord("D"):
                        manual_angle += MANUAL_STEP_BIG
                    manual_angle = clamp(manual_angle, SERVO_MIN_ANGLE, SERVO_MAX_ANGLE)

                    # store point (cx + current angle)
                    if key == ord("p"):
                        if target_cxcy is None:
                            print("[CALIB] No target cx. Put object in view.")
                        else:
                            cx = int(target_cxcy[0])
                            angle = float(last_servo_angle)
                            calib_points.append({"cx": cx, "angle": angle})
                            print(f"[CALIB] Added point: cx={cx}, angle={angle:.2f}")

                    # fit linear
                    if key == ord("f"):
                        res = fit_linear(calib_points)
                        if res is None:
                            print("[CALIB] Need >=2 points.")
                        else:
                            CAL_K, CAL_B = res
                            CALIB_MODE = "linear"
                            print(f"[CALIB] FIT: angle = {CAL_K:.6f}*cx + {CAL_B:.2f} (linear)")

                    if key == ord("r"):
                        calib_points.clear()
                        print("[CALIB] Points cleared.")

                    if key == ord("s"):
                        save_calibration()
                    if key == ord("l"):
                        load_calibration()

        finally:
            if DEBUG_SHOW:
                cv2.destroyAllWindows()
            if servo is not None:
                servo.stop()


if __name__ == "__main__":
    main()
