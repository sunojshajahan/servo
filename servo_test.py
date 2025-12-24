import time
import Jetson.GPIO as GPIO

PIN = 15          # physical pin 15
FREQ = 50         # 50 Hz

def duty_from_pulse_ms(pulse_ms):
    period_ms = 1000 / FREQ
    return (pulse_ms / period_ms) * 100

GPIO.setmode(GPIO.BOARD)
GPIO.setup(PIN, GPIO.OUT)

pwm = GPIO.PWM(PIN, FREQ)
pwm.start(0)

try:
    # center ~1.5ms
    pwm.ChangeDutyCycle(duty_from_pulse_ms(1.5))
    time.sleep(1)

    # left ~1.0ms
    pwm.ChangeDutyCycle(duty_from_pulse_ms(1.0))
    time.sleep(1)

    # right ~2.0ms
    pwm.ChangeDutyCycle(duty_from_pulse_ms(2.0))
    time.sleep(1)

finally:
    pwm.ChangeDutyCycle(0)
    pwm.stop()
    GPIO.cleanup()
