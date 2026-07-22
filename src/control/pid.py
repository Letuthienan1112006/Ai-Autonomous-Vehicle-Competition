"""PID steering and adaptive speed control."""

import time
from collections import deque

import numpy as np

MAX_SPEED = 60
MIN_SPEED = 40

class PIDController:
    def __init__(self, kp, ki, kd, max_output=20):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        self.error_history = deque(maxlen=5)
        self.integral = 0
        self.last_error = 0
        self.last_time = time.time()
        self.last_output = 0

    def compute(self, error):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        if dt < 0.001:
            dt = 0.001

        p_term = error * self.kp

        self.integral += error * dt
        if abs(self.integral) > 8:
            self.integral = 8 * np.sign(self.integral)
        i_term = self.integral * self.ki

        if len(self.error_history) > 0:
            avg_error = (error + self.last_error) / 2
            derivative = (avg_error - self.last_error) / dt
        else:
            derivative = (error - self.last_error) / dt

        self.last_error = error
        d_term = derivative * self.kd

        output = p_term + i_term + d_term

        max_change = 10
        if abs(output - self.last_output) > max_change:
            output = self.last_output + max_change * np.sign(output - self.last_output)

        if abs(output) > self.max_output:
            output = np.sign(output) * self.max_output

        self.error_history.append(error)
        self.last_output = output

        return int(output)


def adaptive_speed_control(error, base_speed=MAX_SPEED, min_speed=MIN_SPEED):
    """
    Điều khiển tốc độ adaptive - TĂNG TỐC ĐỘ ĐỂ LEO DỐC TỐT HƠN
    base_speed: 80->95 (+15)
    min_speed: 60->75 (+15)
    """
    error_abs = abs(error)
    speed_factor = 1 - (error_abs / 280) ** 1.8

    # Nới lỏng penalty khi lệch làn để duy trì tốc độ cao hơn
    if error_abs > 140:  # Tăng ngưỡng từ 120->140
        speed_factor *= 0.8  # Giảm penalty từ 0.75->0.8
    if error_abs > 180:  # Tăng ngưỡng từ 160->180
        speed_factor *= 0.7  # Giảm penalty từ 0.6->0.7

    speed = max(base_speed * speed_factor, min_speed)
    return int(speed)
