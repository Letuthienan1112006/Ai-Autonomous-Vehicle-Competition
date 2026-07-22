"""Multi-frame stability filters used by the navigation controller."""

import time
from collections import deque

class StabilityFilter:
    def __init__(
        self, required_frames=3, confidence_threshold=0.6
    ):  # GIẢM từ 5->3 frames, 0.7->0.6 threshold
        self.required_frames = required_frames
        self.confidence_threshold = confidence_threshold
        self.turn_left_buffer = deque(maxlen=required_frames)
        self.turn_right_buffer = deque(maxlen=required_frames)
        self.confidence_buffer = deque(maxlen=required_frames)

    def update(self, should_turn_left, should_turn_right, confidence):
        self.turn_left_buffer.append(should_turn_left)
        self.turn_right_buffer.append(should_turn_right)
        self.confidence_buffer.append(confidence)

    def get_stable_decision(self):
        if len(self.turn_left_buffer) < self.required_frames:
            return False, False, 0.0

        # Kiểm tra tính nhất quán trong buffer
        left_count = sum(self.turn_left_buffer)
        right_count = sum(self.turn_right_buffer)
        avg_confidence = sum(self.confidence_buffer) / len(self.confidence_buffer)

        # Yêu cầu ít nhất 70% frame đồng ý và độ tin cậy cao - GIẢM NGƯỎNG CHO 60FPS
        stable_left = (
            left_count >= self.required_frames * 0.7  # GIẢM từ 0.8 xuống 0.7
            and avg_confidence >= self.confidence_threshold
        )
        stable_right = (
            right_count >= self.required_frames * 0.7  # GIẢM từ 0.8 xuống 0.7
            and avg_confidence >= self.confidence_threshold
        )

        return stable_left, stable_right, avg_confidence

    def reset(self):
        self.turn_left_buffer.clear()
        self.turn_right_buffer.clear()
        self.confidence_buffer.clear()


# Thêm bộ lọc ổn định cho việc phát hiện biển báo - ĐIỀU CHỈNH CHO 60FPS
class TrafficSignStabilityFilter:
    def __init__(
        self, required_frames=4, disappear_threshold=8
    ):  # GIẢM từ 8->4 và 15->8
        self.required_frames = (
            required_frames  # Số frame cần để xác nhận phát hiện biển báo
        )
        self.disappear_threshold = disappear_threshold  # Số frame liên tiếp mất tín hiệu để xác nhận đã qua biển báo
        self.sign_detection_buffer = deque(
            maxlen=required_frames
        )  # Buffer lưu tín hiệu phát hiện
        self.current_sign_buffer = deque(
            maxlen=required_frames
        )  # Buffer lưu loại biển báo
        self.consecutive_missing = 0  # Đếm số frame liên tiếp không thấy biển báo

    def update(self, sign_detected, current_sign=None):
        """
        Cập nhật trạng thái phát hiện biển báo
        sign_detected: bool - có phát hiện biển báo trong frame này không
        current_sign: str - loại biển báo phát hiện được (nếu có)
        """
        self.sign_detection_buffer.append(sign_detected)
        self.current_sign_buffer.append(current_sign if sign_detected else None)

        if sign_detected:
            self.consecutive_missing = 0
        else:
            self.consecutive_missing += 1

    def is_sign_stably_detected(self):
        """
        Kiểm tra biển báo có được phát hiện ổn định không
        Trả về: (bool, str) - (có phát hiện ổn định, loại biển báo)
        """
        if len(self.sign_detection_buffer) < self.required_frames:
            return False, None

        # Đếm số frame phát hiện biển báo trong buffer
        detection_count = sum(self.sign_detection_buffer)

        # Yêu cầu ít nhất 65% frame phát hiện được biển báo - GIẢM NGƯỎNG CHO 60FPS
        if detection_count >= self.required_frames * 0.65:  # GIẢM từ 0.75 xuống 0.65
            # Tìm biển báo xuất hiện nhiều nhất
            sign_counts = {}
            for sign in self.current_sign_buffer:
                if sign is not None:
                    sign_counts[sign] = sign_counts.get(sign, 0) + 1

            if sign_counts:
                most_common_sign = max(sign_counts, key=sign_counts.get)
                # Yêu cầu biển báo xuất hiện ít nhất 50% trong các frame có phát hiện - GIẢM NGƯỎNG
                if (
                    sign_counts[most_common_sign] >= detection_count * 0.5
                ):  # GIẢM từ 0.6 xuống 0.5
                    return True, most_common_sign

        return False, None

    def has_passed_sign(self):
        """
        Kiểm tra đã đi qua biển báo chưa (mất tín hiệu đủ lâu)
        """
        return self.consecutive_missing >= self.disappear_threshold

    def reset(self):
        """Reset tất cả buffer"""
        self.sign_detection_buffer.clear()
        self.current_sign_buffer.clear()
        self.consecutive_missing = 0


# THÊM: FPS Adapter để tự động điều chỉnh tham số cho 30-60FPS
class FPSAdapter:
    def __init__(self, target_fps=30):
        self.target_fps = target_fps
        self.current_fps = target_fps
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps_check = time.time()

    def update_fps(self):
        self.frame_count += 1
        current_time = time.time()

        # Cập nhật FPS mỗi giây
        if current_time - self.last_fps_check >= 1.0:
            elapsed = current_time - self.start_time
            if elapsed > 0:
                self.current_fps = self.frame_count / elapsed
            self.last_fps_check = current_time

    def get_adapted_frames(self, base_frames_30fps):
        """Chuyển đổi số frame từ 30FPS sang FPS hiện tại"""
        fps_ratio = self.current_fps / 30.0
        return max(1, int(base_frames_30fps * fps_ratio))

    def get_adapted_threshold(self, base_threshold_30fps):
        """Điều chỉnh ngưỡng phần trăm cho FPS cao"""
        if self.current_fps > 50:  # Cho 60FPS
            return base_threshold_30fps * 0.8  # Giảm 20% ngưỡng
        elif self.current_fps > 40:  # Cho 45FPS
            return base_threshold_30fps * 0.9  # Giảm 10% ngưỡng
        return base_threshold_30fps  # Giữ nguyên cho 30FPS


# Shared runtime filter instances.
fps_adapter = FPSAdapter()

def create_adaptive_filters():
    # Các tham số base cho 30FPS
    base_stability_frames = 3
    base_sign_frames = 4
    base_disappear_frames = 6

    # Điều chỉnh dựa trên FPS hiện tại
    adapted_stability_frames = fps_adapter.get_adapted_frames(base_stability_frames)
    adapted_sign_frames = fps_adapter.get_adapted_frames(base_sign_frames)
    adapted_disappear_frames = fps_adapter.get_adapted_frames(base_disappear_frames)

    # Điều chỉnh confidence threshold
    adapted_confidence = fps_adapter.get_adapted_threshold(0.6)
    adapted_sign_threshold = fps_adapter.get_adapted_threshold(0.65)

    stability_filter = StabilityFilter(
        required_frames=adapted_stability_frames,
        confidence_threshold=adapted_confidence,
    )

    traffic_sign_filter = TrafficSignStabilityFilter(
        required_frames=adapted_sign_frames,
        disappear_threshold=adapted_disappear_frames,
    )

    return stability_filter, traffic_sign_filter


stability_filter, traffic_sign_filter = create_adaptive_filters()
