import os
import time
import cv2
import numpy as np
import depthai as dai
from ultralytics import YOLO

# =========================================================
# DEBUG / PERFORMANCE
# =========================================================
DEBUG_SHOW = True            # авто-выключится, если нет DISPLAY (SSH headless)
DEBUG_DRAW = True
PRINT_TARGET = False

INFER_EVERY_N_FRAMES = 1     # 1=каждый кадр, 2=через кадр...

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
# TARGET LOGIC (движение снизу -> вверх)
# =========================================================
TARGET_MISSING_LIMIT = 10   # кадров, после которых сброс цели

# =========================================================
# SERVO (DIRECT GPIO)
# =========================================================
SERVO_ENABLED = True

# Ты указал: signal на физическом пине 15
SERVO_PIN_BOARD = 15        # physical pin 15

SERVO_FREQ_HZ = 50          # стандарт
SERVO_MIN_PULSE_MS = 0.5
SERVO_MAX_PULSE_MS = 2.5

SERVO_MIN_ANGLE = 0
SERVO_MAX_ANGLE = 180
SERVO_CENTER_ANGLE = 90

SERVO_UPDATE_EVERY_N_FRAMES = 2
SERVO_SMOOTH_ALPHA = 0.25
SERVO_MAX_STEP_DEG = 10.0

SERVO_ON_NO_TARGET = "hold"  # "hold" или "center"


# -------------------------
# Headless auto-detect
# -------------------------
if not os.environ.get("DISPLAY"):
    DEBUG_SHOW = False


# -------------------------
# Servo controller
# -------------------------
class ServoController:
    def __init__(
        self,
        pin_board: int,
        freq_hz: int = 50,
        min_pulse_ms: float = 0.5,
        max_pulse_ms: float = 2.5,
        min_angle: float = 0.0,
        max_angle: float = 180.0,
    ):
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

    def _angle_to_duty(self, angle: float) -> float:
        angle = max(self.min_angle, min(self.max_angle, float(angle)))
        pulse_ms = self.min_pulse_ms + (angle - self.min_angle) * (
            (self.max_pulse_ms - self.min_pulse_ms) / (self.max_angle - self.min_angle)
        )
        period_ms = 1000.0 / self.freq
        return (pulse_ms / period_ms) * 100.0

    def set_angle(self, angle: float):
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


def map_cx_to_angle(cx: int, w: int) -> float:
    if w <= 1:
        return SERVO_CENTER_ANGLE
    t = cx / float(w)
    angle = SERVO_MIN_ANGLE + t * (SERVO_MAX_ANGLE - SERVO_MIN_ANGLE)
    return clamp(angle, SERVO_MIN_ANGLE, SERVO_MAX_ANGLE)


def smooth_angle(prev: float, target: float) -> float:
    if prev is None:
        return target
    a = SERVO_SMOOTH_ALPHA
    sm = prev + a * (target - prev)
    delta = sm - prev
    if abs(delta) > SERVO_MAX_STEP_DEG:
        sm = prev + np.sign(delta) * SERVO_MAX_STEP_DEG
    return sm


def main():
    model = YOLO(MODEL_PATH)

    # --------- DepthAI v3 pipeline (OAK) ----------
    # Пример Luxonis показывает паттерн: requestOutput(...).createOutputQueue() и getCvFrame() :contentReference[oaicite:1]{index=1}
    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build()

        # Просим BGR кадры фиксированного размера
        video_queue = cam.requestOutput(
            size=(CAPTURE_W, CAPTURE_H),
            type=dai.ImgFrame.Type.BGR888p,
            resize_mode=dai.ImgResizeMode.LETTERBOX,
            fps=CAM_FPS,
        ).createOutputQueue()

        # Servo init
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

        # Start pipeline
        pipeline.start()

        try:
            while pipeline.isRunning():
                video_in = video_queue.get()
                frame = video_in.getCvFrame()  # BGR frame (numpy)

                frame_idx += 1
                h, w = frame.shape[:2]

                tol = LINE_TOL_PIX if LINE_TOL_PIX is not None else max(2, int(h * LINE_TOL_FRAC))

                # --- ROI ---
                roi_h = int(h * ROI_HEIGHT_FRACTION)
                y1 = (h - roi_h) // 2
                y2 = y1 + roi_h
                roi = frame[y1:y2, :]

                # --- Track ---
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

                # =========================
                # SERVO UPDATE (cx only)
                # =========================
                if SERVO_ENABLED and servo is not None and (frame_idx % SERVO_UPDATE_EVERY_N_FRAMES == 0):
                    if current_target_id and target_found and target_cxcy is not None:
                        cx, cy = target_cxcy
                        target_angle = map_cx_to_angle(cx, w)
                        new_angle = smooth_angle(last_servo_angle, target_angle)
                        servo.set_angle(new_angle)
                        last_servo_angle = new_angle

                        if PRINT_TARGET:
                            print(f"[TARGET] ID={current_target_id} cx={cx} -> angle={new_angle:.1f}")
                    else:
                        if SERVO_ON_NO_TARGET == "center":
                            new_angle = smooth_angle(last_servo_angle, SERVO_CENTER_ANGLE)
                            servo.set_angle(new_angle)
                            last_servo_angle = new_angle
                        # "hold" -> ничего не делаем

                # обработка TRIGGER (раньше было FIRE на Arduino)
                if trigger_ids:
                    handled_ids.update(trigger_ids)
                    if current_target_id in trigger_ids:
                        # Здесь можно вставить действие "FIRE" (реле/нож/второй привод)
                        current_target_id = None
                        target_missing_frames = 0

                # показ / выход
                if DEBUG_SHOW:
                    cv2.imshow("OAK + YOLO + Direct Servo (no Arduino)", output)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):  # ESC/q
                        break

        finally:
            if DEBUG_SHOW:
                cv2.destroyAllWindows()
            if servo is not None:
                servo.stop()


if __name__ == "__main__":
    main()
