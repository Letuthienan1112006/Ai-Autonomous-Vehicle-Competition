"""Lane segmentation post-processing and intersection logic."""

import numpy as np

CLASS_COLORS = {
    0: [0, 0, 0],
    1: [0, 255, 0],
}

def decode_segmap(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        color_mask[mask == class_id] = color
    return color_mask


def get_lane_midpoint(mask, y=270):
    if y >= mask.shape[0]:
        y = mask.shape[0] - 1
    row = mask[y, :]
    lane_indices = np.where(row == 1)[0]

    if len(lane_indices) < 2:
        return None

    left = lane_indices[0]
    right = lane_indices[-1]

    mid = (left + right) // 2
    return left, right, mid


def detect_intersection_points(mask, scan_y=270):
    """
    Phát hiện các điểm giao lộ bằng cách phân tích nhiều điểm trên ảnh
    Trả về thông tin về các làn đường ở các vị trí khác nhau
    scan_y: Vị trí y để quét (mặc định 270)
    """
    height, width = mask.shape
    intersection_info = {
        "center_left": None,
        "center": None,
        "center_right": None,
        "left": None,
        "right": None,
        "is_intersection": False,
        "lane_count": 0,
        "should_turn_left": False,
        "should_turn_right": False,
        "left_shift": 0,
        "right_shift": 0,
        "confidence": 0.0,  # Thêm độ tin cậy
        "stability": 0.0,  # Thêm độ ổn định
    }

    # Sử dụng scan_y từ tham số thay vì cố định 270
    scan_y = min(scan_y, height - 1)

    # Đầu tiên, tìm ranh giới của toàn bộ vùng làn đường
    row = mask[scan_y, :]
    lane_pixels = np.where(row == 1)[0]

    if len(lane_pixels) < 10:
        return intersection_info

    # Ranh giới ngoài cùng của làn đường
    lane_leftmost = lane_pixels[0]
    lane_rightmost = lane_pixels[-1]
    lane_total_width = lane_rightmost - lane_leftmost

    # Định nghĩa các điểm kiểm tra với điểm trái/phải ở ngoài cùng
    scan_points = {
        "center": width // 2,
        "center_left": lane_leftmost + lane_total_width // 4,
        "center_right": lane_leftmost + 3 * lane_total_width // 4,
        "left": lane_leftmost,  # Điểm trái ở ngoài cùng bên trái
        "right": lane_rightmost,  # Điểm phải ở ngoài cùng bên phải
    }

    # Lưu vị trí ban đầu của điểm trái và phải để so sánh
    if not hasattr(detect_intersection_points, "initial_left"):
        detect_intersection_points.initial_left = lane_leftmost
        detect_intersection_points.initial_right = lane_rightmost
        detect_intersection_points.initial_center = width // 2

    # Phân tích từng điểm
    for point_name, x_pos in scan_points.items():
        if 0 <= x_pos < width:
            if point_name in ["left", "right"]:
                # Đối với điểm trái/phải, kiểm tra chính xác tại vị trí đó
                if mask[scan_y, x_pos] == 1:
                    intersection_info[point_name] = {
                        "detected": True,
                        "start": x_pos,
                        "end": x_pos,
                        "center": x_pos,
                        "width": 1,
                    }
                else:
                    intersection_info[point_name] = {"detected": False}
            else:
                # Đối với các điểm khác, kiểm tra vùng xung quanh
                x_start = max(0, x_pos - 30)
                x_end = min(width, x_pos + 30)

                row_segment = mask[scan_y, x_start:x_end]
                lane_pixels_segment = np.where(row_segment == 1)[0]

                if len(lane_pixels_segment) > 10:
                    lane_start = lane_pixels_segment[0] + x_start
                    lane_end = lane_pixels_segment[-1] + x_start
                    lane_center = (lane_start + lane_end) // 2

                    intersection_info[point_name] = {
                        "detected": True,
                        "start": lane_start,
                        "end": lane_end,
                        "center": lane_center,
                        "width": lane_end - lane_start,
                    }
                else:
                    intersection_info[point_name] = {"detected": False}
        else:
            intersection_info[point_name] = {"detected": False}

    # Đếm số làn phát hiện được trước khi sử dụng
    detected_lanes = sum(
        1
        for info in intersection_info.values()
        if isinstance(info, dict) and info.get("detected", False)
    )

    # Đảm bảo detected_lanes có giá trị hợp lệ
    detected_lanes = max(0, detected_lanes)

    # Tính toán độ dịch chuyển của điểm trái và phải
    if intersection_info["left"].get("detected"):
        intersection_info["left_shift"] = (
            intersection_info["left"]["center"]
            - detect_intersection_points.initial_left
        )

    if intersection_info["right"].get("detected"):
        intersection_info["right_shift"] = (
            intersection_info["right"]["center"]
            - detect_intersection_points.initial_right
        )

    # Ngưỡng động dựa trên tốc độ và điều kiện đường
    base_turn_threshold = 80
    # Điều chỉnh ngưỡng dựa trên số làn đường và độ rộng
    if detected_lanes >= 4:
        turn_threshold = base_turn_threshold * 1.3  # Tăng ngưỡng khi có nhiều làn
    elif detected_lanes == 3:
        turn_threshold = base_turn_threshold * 1.1
    else:
        turn_threshold = base_turn_threshold * 0.9  # Giảm ngưỡng khi ít làn

    # Tính độ tin cậy dựa trên độ nhất quán của shift
    left_shift_abs = abs(intersection_info["left_shift"])
    right_shift_abs = abs(intersection_info["right_shift"])

    # Độ tin cậy tăng khi shift lớn và số làn nhiều
    confidence_score = 0.0
    if detected_lanes >= 3:
        confidence_score += 0.3
    if max(left_shift_abs, right_shift_abs) > turn_threshold:
        confidence_score += 0.4
    if (
        detected_lanes >= 4
        and max(left_shift_abs, right_shift_abs) > turn_threshold * 1.2
    ):
        confidence_score += 0.3

    intersection_info["confidence"] = min(confidence_score, 1.0)

    # Phát hiện khi cần rẽ với điều kiện nghiêm ngặt hơn
    intersection_info["should_turn_left"] = (
        intersection_info["left_shift"] > turn_threshold
        and detected_lanes >= 3  # Tăng yêu cầu số làn
        and intersection_info["confidence"] > 0.6  # Yêu cầu độ tin cậy cao
    )
    intersection_info["should_turn_right"] = (
        intersection_info["right_shift"] < -turn_threshold
        and detected_lanes >= 3  # Tăng yêu cầu số làn
        and intersection_info["confidence"] > 0.6  # Yêu cầu độ tin cậy cao
    )

    intersection_info["lane_count"] = detected_lanes
    intersection_info["is_intersection"] = detected_lanes >= 3

    return intersection_info


def calculate_intersection_steering(intersection_info, traffic_sign_command):
    """
    Tính toán góc lái khi ở giao lộ dựa trên biển báo và độ dịch chuyển của các điểm
    """
    if not intersection_info["is_intersection"]:
        return None

    center_x = 320  # Trung tâm ảnh (width=640)

    # Ưu tiên điều khiển theo độ dịch chuyển của các điểm nếu không có biển báo
    if not traffic_sign_command:
        if intersection_info["should_turn_left"]:
            return -15  # Rẽ trái
        elif intersection_info["should_turn_right"]:
            return 15  # Rẽ phải
        else:
            # Giữ làn trung tâm
            center_lane = intersection_info.get("center")
            if center_lane and center_lane.get("detected"):
                current_center = center_lane["center"]
                steering_angle = (current_center - center_x) * 0.3
                return max(-5, min(5, steering_angle))
            return 0

    # Lấy thông tin làn trung tâm
    center_lane = intersection_info.get("center")
    if not center_lane or not center_lane.get("detected"):
        return None

    current_center = center_lane["center"]

    # Tính toán góc lái dựa trên lệnh biển báo - GIẢM GÓC LÁI ĐỂ TRÁNH VỤT QUA
    if traffic_sign_command == "turn_left":
        # Rẽ trái: sử dụng điểm left hoặc center_left
        if intersection_info["left"] and intersection_info["left"].get("detected"):
            target_x = intersection_info["left"]["center"]
        else:
            target_x = intersection_info.get("center_left", {}).get(
                "center", center_x - 80  # Giảm từ 100 xuống 80
            )

        steering_angle = (target_x - center_x) * 0.4  # Giảm từ 0.6 xuống 0.4
        return max(
            -15, min(-5, steering_angle)
        )  # Giảm góc lái: từ (-20,-8) xuống (-15,-5)

    elif traffic_sign_command == "turn_right":
        # Rẽ phải: sử dụng điểm right hoặc center_right
        if intersection_info["right"] and intersection_info["right"].get("detected"):
            target_x = intersection_info["right"]["center"]
        else:
            target_x = intersection_info.get("center_right", {}).get(
                "center", center_x + 80  # Giảm từ 100 xuống 80
            )

        steering_angle = (target_x - center_x) * 0.4  # Giảm từ 0.6 xuống 0.4
        return max(5, min(15, steering_angle))  # Giảm góc lái: từ (8,20) xuống (5,15)

    elif traffic_sign_command == "straight":
        # Đi thẳng: BỎ QUA hoàn toàn các signal rẽ, CHỈ tập trung giữ làn trung tâm
        # Nếu không có làn trung tâm rõ ràng, sử dụng làn hiện tại
        if center_lane and center_lane.get("detected"):
            steering_angle = (
                current_center - center_x
            ) * 0.25  # Giảm hệ số để ổn định hơn
            return max(-8, min(8, steering_angle))  # Cho phép điều chỉnh nhẹ hơn
        else:
            # Nếu không phát hiện được làn trung tâm, giữ hướng hiện tại
            return 0

    return None


def detect_lane_departure(intersection_info):
    """
    Phát hiện khi xe cần rẽ dựa trên sự thay đổi vị trí các điểm trái/phải
    """
    left_shift = intersection_info.get("left_shift", 0)
    right_shift = intersection_info.get("right_shift", 0)
    lane_count = intersection_info.get("lane_count", 0)

    # Ngưỡng để xác định cần rẽ - GIẢM để xe rẽ sớm hơn
    turn_threshold = 100  # Giảm từ 180 xuống 100

    # Yêu cầu cả độ dịch chuyển lớn VÀ có đủ làn đường (thực sự ở giao lộ)
    if left_shift > turn_threshold and lane_count >= 2:  # Giảm từ 3 xuống 2
        return "turn_left"
    elif right_shift < -turn_threshold and lane_count >= 2:  # Giảm từ 3 xuống 2
        return "turn_right"

    return None

def detect_steep_slope(lane_error, lane_points, prev_lane_points=None):
    """
    Phát hiện dốc cao dựa trên độ thay đổi vị trí làn đường
    Trả về hệ số tăng tốc (1.0 = bình thường, 1.2 = tăng 20%)
    """
    base_boost = 1.0

    # Nếu lệch làn lớn (có thể do dốc) -> tăng tốc để duy trì momentum
    if abs(lane_error) > 100:
        base_boost = 1.15  # Tăng 15%
    elif abs(lane_error) > 50:
        base_boost = 1.1  # Tăng 10%

    # Phát hiện thay đổi đột ngột vị trí làn (có thể do dốc)
    if lane_points and prev_lane_points:
        try:
            prev_left, prev_right, prev_mid = prev_lane_points
            curr_left, curr_right, curr_mid = lane_points

            # Tính độ thay đổi vị trí làn
            mid_change = abs(curr_mid - prev_mid)
            width_change = abs((curr_right - curr_left) - (prev_right - prev_left))

            # Nếu có thay đổi lớn -> có thể là dốc
            if mid_change > 30 or width_change > 20:
                base_boost = max(base_boost, 1.2)  # Tăng tối đa 20%

        except (TypeError, ValueError):
            pass  # Ignore errors in unpacking

    return base_boost
