"""Main autonomous-vehicle controller used for the 2025 competition."""

import base64
import logging
import time

import cv2
import numpy as np
import torch
from .models.segmentation import create_deeplabv3
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
from .communication.simulator_client import SimulatorClient, SimulatorConnectionError
from .control.pid import PIDController, adaptive_speed_control
from .navigation.filters import fps_adapter, stability_filter, traffic_sign_filter
from .vision.lane import (
    calculate_intersection_steering,
    decode_segmap,
    detect_intersection_points,
    detect_lane_departure,
    detect_steep_slope,
    get_lane_midpoint,
)
from .config import (
    OBJECT_DETECTION_WEIGHTS,
    SEGMENTATION_WEIGHTS,
    SIMULATOR_HOST,
    SIMULATOR_PORT,
    TRAFFIC_SIGN_WEIGHTS,
    validate_custom_weights,
)
TRAFFIC_SIGN_CONF = 0.70
VEHICLE_CONF = 0.70
VEHICLE_IOU = 0.50
MIN_BOX_AREA_SIGN = 500
MIN_BOX_AREA_VEHICLE = 900
VEHICLE_STOP_SECONDS = 10.0

# Tắt logging của YOLO
logging.getLogger("ultralytics").setLevel(logging.ERROR)


# YOLOv11 class names
YOLO_CLASSES = [
    "no_straight",  # 0 - cấm đi thẳng
    "no_turn_left",  # 1 - cấm rẽ trái
    "no_turn_right",  # 2 - cấm rẽ phải
    "stop",  # 3 - dừng lại
    "straight",  # 4 - chỉ được đi thẳng
    "turn_left",  # 5 - chỉ được rẽ trái
    "turn_right",  # 6 - chỉ được rẽ phải
    "unknown",  # 7 - không xác định
]
#YOLO_CLASSES += ["vehicle", "pedestrian", "bicycle"]  # Các lớp bổ sung nếu có




img_transform = transforms.Compose(
    [
        transforms.Resize((360, 640)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
yolo_model = None
yolo_obj = None
COCO_NAMES = {}
VEHICLE_SET = {"car", "truck", "bus", "motorcycle", "bicycle", "person"}


def load_runtime_models():
    """Load segmentation, traffic-sign, and obstacle-detection models."""
    global model, yolo_model, yolo_obj, COCO_NAMES

    validate_custom_weights()

    model = create_deeplabv3(
        num_classes=2,
        backbone="resnet50",
        pretrained=False,
    ).to(device)

    checkpoint = torch.load(str(SEGMENTATION_WEIGHTS), map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    yolo_model = YOLO(str(TRAFFIC_SIGN_WEIGHTS))
    yolo_obj = YOLO(OBJECT_DETECTION_WEIGHTS)
    COCO_NAMES = yolo_obj.names

steer_angle = 0
car_speed = 0

# Biến lưu trạng thái giao lộ và biển báo - sử dụng biến cờ thay vì thời gian
intersection_mode = False
traffic_sign_command = None

# Biến cờ để theo dõi trạng thái biển báo
sign_detected = False  # Đã phát hiện biển báo
passed_sign = False  # Đã đi qua biển báo (không còn thấy biển báo)
ready_to_turn = False  # Sẵn sàng rẽ (đã qua biển báo và đến giao lộ)
frames_without_sign = 0  # Đếm số frame không thấy biển báo để xác định đã qua


# Thêm bộ lọc ổn định cho quyết định rẽ - ĐIỀU CHỈNH CHO 60FPS


def main():
    global steer_angle, car_speed
    global intersection_mode, traffic_sign_command
    global sign_detected, passed_sign, ready_to_turn, frames_without_sign

    steer_angle = 0
    car_speed = 0
    intersection_mode = False
    traffic_sign_command = None
    sign_detected = False
    passed_sign = False
    ready_to_turn = False
    frames_without_sign = 0
    stability_filter.reset()
    traffic_sign_filter.reset()

    load_runtime_models()
    client = SimulatorClient(SIMULATOR_HOST, SIMULATOR_PORT)
    client.connect()

    last_midpoint = None
    prev_lane_points = None  # THÊM: Lưu lane points trước đó để phát hiện dốc
    lane_pid = PIDController(kp=0.16, ki=0.001, kd=0.01, max_output=20)

    with torch.no_grad():
        try:
            while True:
                try:
                    recv_data = client.exchange_control(steer_angle, car_speed)

                    jpg_data = base64.b64decode(recv_data["Img"])
                    jpg_array = np.frombuffer(jpg_data, dtype=np.uint8)
                    cam_image = cv2.imdecode(jpg_array, flags=1)
                    if cam_image is None:
                        raise ValueError("Simulator frame could not be decoded as an image.")
                    pil_image = Image.fromarray(
                        cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB)
                    )
                    img_tensor = img_transform(pil_image).unsqueeze(0).to(device)

                    # DeepLabV3 inference
                    output = model(img_tensor)
                    pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

                    # YOLOv11 inference để phát hiện biển báo
                    yolo_results = yolo_model(cam_image, conf=TRAFFIC_SIGN_CONF)
                    detections = []
                    current_traffic_signs = []

                    for result in yolo_results:
                        boxes = result.boxes
                        if boxes is not None:
                            for box in boxes:
                                bbox = box.xyxy[0].cpu().numpy()
                                width = bbox[2] - bbox[0]
                                height = bbox[3] - bbox[1]
                                box_area = width * height

                                if box_area >= MIN_BOX_AREA_SIGN:
                                    class_id = int(box.cls[0])
                                    if not 0 <= class_id < len(YOLO_CLASSES):
                                        continue
                                    detection = {
                                        "class": class_id,
                                        "confidence": float(box.conf[0]),
                                        "bbox": bbox,
                                        "area": box_area,
                                    }
                                    detections.append(detection)
                                    current_traffic_signs.append(YOLO_CLASSES[class_id])
                    #✅ Phát hiện xe/người bằng model COCO
                    veh_results = yolo_obj(
                        cam_image, conf=VEHICLE_CONF, iou=VEHICLE_IOU, classes=[0,1,2,3,5,7]  # person,bicycle,car,motorcycle,bus,truck
                    )
                    vehicle_boxes = []
                    for r in veh_results:
                        if r.boxes is None:
                            continue
                        for b in r.boxes:
                            cls_id = int(b.cls[0])
                            label = COCO_NAMES[cls_id]
                            if label in VEHICLE_SET:
                                x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
                                box_area = max(0, x2 - x1) * max(0, y2 - y1)
                                if box_area < MIN_BOX_AREA_VEHICLE:
                                    continue
                                conf = float(b.conf[0])
                                vehicle_boxes.append((x1, y1, x2, y2, label, conf))


                                    

                    # Phát hiện giao lộ và các điểm làn đường
                    # Đường cho PID (giữ nguyên ở y=270)
                    intersection_info = detect_intersection_points(pred_mask)

                    # Thêm bộ đếm ổn định cho giao lộ (THỰC HIỆN SAU KHI CÓ THÔNG TIN GIAO LỘ)
                    if not hasattr(
                        detect_intersection_points, "intersection_stable_count"
                    ):
                        detect_intersection_points.intersection_stable_count = 0

                    # Đếm số frame liên tiếp phát hiện giao lộ ổn định (ĐIỀU CHỈNH CHO 60FPS)
                    if (
                        intersection_info["is_intersection"]
                        and intersection_info["lane_count"] >= 3  # Giữ nguyên 3
                    ):
                        detect_intersection_points.intersection_stable_count += 1
                    else:
                        detect_intersection_points.intersection_stable_count = 0

                    # GIẢM MẠNH yêu cầu ổn định từ 6 xuống 3 frame cho 60FPS
                    intersection_truly_stable = (
                        detect_intersection_points.intersection_stable_count >= 3
                    )

                    lane_points = get_lane_midpoint(pred_mask, y=270)

                    # Cập nhật bộ lọc ổn định
                    stability_filter.update(
                        intersection_info.get("should_turn_left", False),
                        intersection_info.get("should_turn_right", False),
                        intersection_info.get("confidence", 0.0),
                    )

                    # Lấy quyết định ổn định từ bộ lọc
                    stable_left, stable_right, avg_confidence = (
                        stability_filter.get_stable_decision()
                    )

                    # THÊM: Điểm kiểm tra trước để phát hiện sớm khi đường thẳng sắp hết (y=200)
                    early_check_points = get_lane_midpoint(pred_mask, y=200)

                    # Đường cho hiển thị 5 điểm màu sắc (ở vị trí y=360)
                    intersection_info_display = detect_intersection_points(
                        pred_mask, scan_y=360
                    )

                    lane_error = 0

                    # Chỉ phát hiện lane departure khi có biển báo lệnh
                    suggested_turn = None

                    # Xử lý biển báo và cập nhật trạng thái bằng biến cờ
                    current_sign_found = False
                    has_command_sign = False  # Biến để kiểm tra có biển báo lệnh không
                    has_prohibition_sign = (
                        False  # Biến để kiểm tra có biển báo cấm không
                    )

                    # Xác định biển báo hiện tại để cập nhật vào bộ lọc
                    detected_sign = None
                    if current_traffic_signs:
                        for sign in current_traffic_signs:
                            # Xử lý biển báo LỆNH (cho phép)
                            if sign in ["turn_left", "turn_right", "straight"]:
                                has_command_sign = True
                                detected_sign = sign
                                current_sign_found = True
                                break

                            # Xử lý biển báo CẤM (chuyển đổi thành lệnh rẽ)
                            elif sign in [
                                "no_straight",
                                "no_turn_left",
                                "no_turn_right",
                            ]:
                                has_prohibition_sign = True
                                # Chuyển đổi biển cấm thành lệnh rẽ phù hợp
                                converted_command = None

                                if sign == "no_straight":
                                    # Cấm đi thẳng → kiểm tra trực tiếp sự hiện diện của làn đường
                                    left_lane_available = False
                                    right_lane_available = False

                                    # Kiểm tra làn bên trái
                                    if intersection_info.get(
                                        "left"
                                    ) and intersection_info["left"].get("detected"):
                                        left_lane_available = True
                                    elif intersection_info.get(
                                        "center_left"
                                    ) and intersection_info["center_left"].get(
                                        "detected"
                                    ):
                                        left_lane_available = True

                                    # Kiểm tra làn bên phải
                                    if intersection_info.get(
                                        "right"
                                    ) and intersection_info["right"].get("detected"):
                                        right_lane_available = True
                                    elif intersection_info.get(
                                        "center_right"
                                    ) and intersection_info["center_right"].get(
                                        "detected"
                                    ):
                                        right_lane_available = True

                                    # Quyết định hướng rẽ dựa trên làn đường có sẵn
                                    if left_lane_available and right_lane_available:
                                        # Cả hai hướng đều có đường → ưu tiên rẽ trái
                                        converted_command = "turn_left"
                                    elif left_lane_available:
                                        # Chỉ có đường bên trái
                                        converted_command = "turn_left"
                                    elif right_lane_available:
                                        # Chỉ có đường bên phải
                                        converted_command = "turn_right"
                                    else:
                                        # Không có đường rõ ràng → mặc định rẽ phải
                                        converted_command = "turn_right"

                                elif sign == "no_turn_left":
                                    # Cấm rẽ trái → KIỂM TRA ĐIỂM TRƯỚC để quyết định
                                    # Nếu điểm trước (y=200) vẫn có đường thẳng → đi thẳng
                                    # Nếu điểm trước không có đường thẳng → rẽ phải

                                    can_go_straight_ahead = (
                                        early_check_points is not None
                                    )

                                    if can_go_straight_ahead:
                                        # Vẫn còn đường phía trước → ưu tiên đi thẳng
                                        converted_command = "straight"
                                    else:
                                        # Không còn đường phía trước → phải rẽ
                                        right_lane_available = False

                                        if intersection_info.get(
                                            "right"
                                        ) and intersection_info["right"].get(
                                            "detected"
                                        ):
                                            right_lane_available = True
                                        elif intersection_info.get(
                                            "center_right"
                                        ) and intersection_info["center_right"].get(
                                            "detected"
                                        ):
                                            right_lane_available = True

                                        if (
                                            right_lane_available
                                            and intersection_info.get("lane_count", 0)
                                            >= 3
                                        ):
                                            converted_command = "turn_right"
                                        else:
                                            converted_command = (
                                                "straight"  # Fallback an toàn
                                            )

                                elif sign == "no_turn_right":
                                    # Cấm rẽ phải → KIỂM TRA ĐIỂM TRƯỚC để quyết định
                                    # Nếu điểm trước (y=200) vẫn có đường thẳng → đi thẳng
                                    # Nếu điểm trước không có đường thẳng → rẽ trái

                                    can_go_straight_ahead = (
                                        early_check_points is not None
                                    )

                                    if can_go_straight_ahead:
                                        # Vẫn còn đường phía trước → ưu tiên đi thẳng
                                        converted_command = "straight"
                                    else:
                                        # Không còn đường phía trước → phải rẽ
                                        left_lane_available = False

                                        if intersection_info.get(
                                            "left"
                                        ) and intersection_info["left"].get("detected"):
                                            left_lane_available = True
                                        elif intersection_info.get(
                                            "center_left"
                                        ) and intersection_info["center_left"].get(
                                            "detected"
                                        ):
                                            left_lane_available = True

                                        if (
                                            left_lane_available
                                            and intersection_info.get("lane_count", 0)
                                            >= 3
                                        ):
                                            converted_command = "turn_left"
                                        else:
                                            converted_command = (
                                                "straight"  # Fallback an toàn
                                            )

                                # Sử dụng lệnh đã chuyển đổi làm detected_sign
                                if converted_command:
                                    detected_sign = converted_command
                                    current_sign_found = True
                                break

                    # Cập nhật bộ lọc ổn định cho biển báo
                    traffic_sign_filter.update(current_sign_found, detected_sign)

                    # Kiểm tra biển báo có được phát hiện ổn định không
                    is_stable, stable_sign = (
                        traffic_sign_filter.is_sign_stably_detected()
                    )

                    # Cập nhật trạng thái biển báo dựa trên kết quả ổn định
                    if is_stable and not sign_detected:
                        # Biển báo được phát hiện ổn định, cập nhật trạng thái
                        sign_detected = True
                        traffic_sign_command = stable_sign
                        passed_sign = False
                        ready_to_turn = False
                        frames_without_sign = 0

                    # Kiểm tra đã đi qua biển báo chưa (sử dụng bộ lọc ổn định)
                    if (
                        sign_detected
                        and traffic_sign_filter.has_passed_sign()
                        and not passed_sign
                    ):
                        passed_sign = True
                        frames_without_sign = 0

                    # CHỈ sử dụng detect_lane_departure khi có biển báo lệnh hoặc cấm
                    # NHƯNG BỎ QUA khi có lệnh đi thẳng để tránh bị nhiễu
                    # SỬ DỤNG QUYẾT ĐỊNH ỔN ĐỊNH từ bộ lọc
                    suggested_turn = None
                    if (
                        has_command_sign or has_prohibition_sign or sign_detected
                    ) and traffic_sign_command != "straight":
                        # Sử dụng quyết định ổn định thay vì quyết định tức thời
                        if stable_left and avg_confidence > 0.6:
                            suggested_turn = "turn_left"
                        elif stable_right and avg_confidence > 0.65:
                            suggested_turn = "turn_right"

                    # Kiểm tra đã đi qua biển báo chưa
                    if sign_detected and not current_sign_found:
                        frames_without_sign += 1
                        # SỬ DỤNG FPS ADAPTER để tự động điều chỉnh
                        required_frames = fps_adapter.get_adapted_frames(
                            4
                        )  # Base: 4 frame cho 30FPS
                        if frames_without_sign >= required_frames and not passed_sign:
                            passed_sign = True
                            frames_without_sign = 0
                    else:
                        frames_without_sign = 0

                    # Kiểm tra sẵn sàng rẽ: YÊU CẦU NGHIÊM NGẶT HƠN
                    # CHỈ khi có biển báo lệnh VÀ ĐÃ QUA BIỂN BÁO VÀ đến giao lộ VÀ có đủ làn đường
                    # BỎ logic strong_turn_signal để tránh rẽ sớm

                    # Thêm điều kiện khoảng cách an toàn sau biển báo (SỬ DỤNG FPS ADAPTER)
                    safe_distance_after_sign = True
                    if passed_sign:
                        # SỬ DỤNG FPS ADAPTER: Base 5 frame cho 30FPS
                        if not hasattr(
                            detect_intersection_points, "frames_since_passed"
                        ):
                            detect_intersection_points.frames_since_passed = 0
                        detect_intersection_points.frames_since_passed += 1

                        required_safe_frames = fps_adapter.get_adapted_frames(5)
                        if (
                            detect_intersection_points.frames_since_passed
                            < required_safe_frames
                        ):
                            safe_distance_after_sign = False
                    else:
                        if hasattr(detect_intersection_points, "frames_since_passed"):
                            detect_intersection_points.frames_since_passed = 0

                    # Điều kiện backup: Cho phép rẽ nhanh hơn với tín hiệu mạnh và ổn định
                    strong_and_stable_signal = (
                        passed_sign  # Vẫn yêu cầu đã qua biển báo
                        and intersection_info["is_intersection"]
                        and intersection_info["lane_count"] >= 3
                        and stable_left
                        and avg_confidence > 0.8  # Tín hiệu rẽ trái mạnh
                        and traffic_sign_command == "turn_left"
                    ) or (
                        passed_sign  # Vẫn yêu cầu đã qua biển báo
                        and intersection_info["is_intersection"]
                        and intersection_info["lane_count"] >= 3
                        and stable_right
                        and avg_confidence > 0.8  # Tín hiệu rẽ phải mạnh
                        and traffic_sign_command == "turn_right"
                    )

                    if (
                        passed_sign  # BẮT BUỘC phải đã qua biển báo
                        and safe_distance_after_sign  # VÀ đã có khoảng cách an toàn (GIẢM)
                        and intersection_info["is_intersection"]  # VÀ đang ở giao lộ
                        and intersection_truly_stable  # VÀ giao lộ đã ổn định
                        and intersection_info["lane_count"]
                        >= 3  # GIẢM yêu cầu từ 4 xuống 3 làn đường
                        and not ready_to_turn
                        and traffic_sign_command
                        in ["turn_left", "turn_right", "straight"]
                    ) or (
                        strong_and_stable_signal  # HOẶC có tín hiệu mạnh và ổn định
                        and not ready_to_turn
                    ):
                        ready_to_turn = True
                        intersection_mode = True
                        # Khởi tạo bộ đếm frame intersection
                        if not hasattr(
                            detect_intersection_points, "intersection_frames"
                        ):
                            detect_intersection_points.intersection_frames = 0

                    # Đếm frame trong intersection mode
                    if intersection_mode and ready_to_turn:
                        if not hasattr(
                            detect_intersection_points, "intersection_frames"
                        ):
                            detect_intersection_points.intersection_frames = 0
                        detect_intersection_points.intersection_frames += 1
                    else:
                        if hasattr(detect_intersection_points, "intersection_frames"):
                            detect_intersection_points.intersection_frames = 0

                    # Thoát khỏi chế độ giao lộ khi đã rẽ xong (không còn ở giao lộ)
                    if intersection_mode and ready_to_turn:
                        if not intersection_info["is_intersection"]:
                            # Reset tất cả biến cờ khi thoát giao lộ
                            intersection_mode = False
                            traffic_sign_command = None
                            sign_detected = False
                            passed_sign = False
                            ready_to_turn = False
                            frames_without_sign = 0

                    # LOGIC RESET BỔ SUNG: Reset khi đã hoàn thành lệnh hoặc trong các trường hợp khác
                    reset_needed = False

                    # THÊM: Kiểm tra reset nhanh khi đã có lane rõ ràng sau khi rẽ
                    if intersection_mode and ready_to_turn and lane_points:
                        left_x, right_x, mid_x = lane_points
                        lane_width = right_x - left_x
                        lane_center_error = abs(mid_x - center_x)

                        # Nếu có lane rõ ràng và không còn ở giao lộ phức tạp
                        if (
                            lane_width > 60
                            and lane_center_error < 50
                            and intersection_info["lane_count"] <= 2
                        ):
                            reset_needed = True

                    # Trường hợp 1: RESET NGAY LẬP TỨC khi hoàn thành lệnh rẽ
                    if intersection_mode and ready_to_turn and traffic_sign_command:
                        # Kiểm tra đã hoàn thành lệnh rẽ dựa trên lane stability
                        if lane_points:
                            left_x, right_x, mid_x = lane_points
                            lane_width = right_x - left_x
                            lane_center_error = abs(mid_x - center_x)

                            # Nếu làn đường đã ổn định (error nhỏ và có lane rõ ràng) - NỚI LỎNG ĐỂ RESET DỄ HƠN CHO 60FPS
                            lane_stable = (
                                lane_center_error < 40  # NỚI LỎNG từ 35 lên 40
                                and lane_width > 70  # NỚI LỎNG từ 80 xuống 70
                            )

                            # Kiểm tra đã thực hiện đủ lệnh rẽ - GIẢM NGƯỎNG ĐỂ RESET SỚM HƠN CHO 60FPS
                            command_completed = False

                            if traffic_sign_command == "turn_left":
                                # Đã rẽ trái đủ nếu mid_x đã dịch sang trái - GIẢM NGƯỎNG CHO 60FPS
                                if mid_x < center_x - 30:  # GIẢM từ 40 xuống 30 pixel
                                    command_completed = True
                            elif traffic_sign_command == "turn_right":
                                # Đã rẽ phải đủ nếu mid_x đã dịch sang phải - GIẢM NGƯỎNG CHO 60FPS
                                if mid_x > center_x + 30:  # GIẢM từ 40 xuống 30 pixel
                                    command_completed = True
                            elif traffic_sign_command == "straight":
                                # Lệnh đi thẳng luôn hoàn thành nếu lane stable
                                command_completed = True

                            # Reset ngay nếu đã hoàn thành lệnh và lane ổn định
                            if command_completed and lane_stable:
                                reset_needed = True

                    # Trường hợp 1.5: Timeout trong intersection mode (tránh kẹt) - SỬ DỤNG FPS ADAPTER
                    if intersection_mode and ready_to_turn:
                        intersection_frames = getattr(
                            detect_intersection_points, "intersection_frames", 0
                        )
                        # SỬ DỤNG FPS ADAPTER: Base 25 frame cho 30FPS
                        timeout_frames = fps_adapter.get_adapted_frames(25)
                        if intersection_frames > timeout_frames:
                            reset_needed = True

                    # Trường hợp 2: Đã qua giao lộ (thực hiện xong lệnh)
                    if (
                        sign_detected or passed_sign or ready_to_turn
                    ) and not intersection_info["is_intersection"]:
                        # Nếu đã có trạng thái biển báo nhưng không còn ở giao lộ nữa
                        reset_needed = True

                    # Trường hợp 3: Thấy biển báo mới khác với biển báo cũ
                    if current_sign_found and traffic_sign_command:
                        for sign in current_traffic_signs:
                            if (
                                sign in ["turn_left", "turn_right", "straight"]
                                and sign != traffic_sign_command
                            ):
                                # Biển báo mới khác biển báo cũ
                                reset_needed = True
                                break

                    # Trường hợp 4: Đã qua biển báo quá lâu mà không đến giao lộ
                    if passed_sign and not intersection_info["is_intersection"]:
                        # Có thể thêm bộ đếm frame để reset sau một thời gian
                        if not hasattr(detect_intersection_points, "frames_after_sign"):
                            detect_intersection_points.frames_after_sign = 0
                        detect_intersection_points.frames_after_sign += 1

                        # SỬ DỤNG FPS ADAPTER: Reset nếu đã đi quá base 40 frame mà không thấy giao lộ
                        timeout_after_sign = fps_adapter.get_adapted_frames(40)
                        if (
                            detect_intersection_points.frames_after_sign
                            > timeout_after_sign
                        ):
                            reset_needed = True
                            detect_intersection_points.frames_after_sign = 0
                    else:
                        if hasattr(detect_intersection_points, "frames_after_sign"):
                            detect_intersection_points.frames_after_sign = 0

                    # Thực hiện reset nếu cần
                    if reset_needed:
                        intersection_mode = False
                        traffic_sign_command = None
                        sign_detected = False
                        passed_sign = False
                        ready_to_turn = False
                        frames_without_sign = 0
                        stability_filter.reset()  # Reset bộ lọc ổn định
                        traffic_sign_filter.reset()  # Reset bộ lọc biển báo
                        if hasattr(detect_intersection_points, "frames_after_sign"):
                            detect_intersection_points.frames_after_sign = 0
                        if hasattr(detect_intersection_points, "intersection_frames"):
                            detect_intersection_points.intersection_frames = 0
                        if hasattr(detect_intersection_points, "frames_since_passed"):
                            detect_intersection_points.frames_since_passed = 0
                        if hasattr(
                            detect_intersection_points, "intersection_stable_count"
                        ):
                            detect_intersection_points.intersection_stable_count = 0

                    img_resized = np.array(pil_image.resize((640, 360)))
                    mask_color = decode_segmap(pred_mask)
                    display_img = cv2.addWeighted(img_resized, 0.5, mask_color, 0.5, 0)

                    scan_line_y = 240
                    center_x = display_img.shape[1] // 2

                    # LOGIC ĐIỀU KHIỂN
                    if (
                        intersection_mode
                        and ready_to_turn
                        and traffic_sign_command
                        in ["turn_left", "turn_right", "straight"]
                    ):
                        # Chế độ giao lộ: CHỈ điều khiển theo biển báo lệnh (không phải unknown)
                        intersection_steer = calculate_intersection_steering(
                            intersection_info, traffic_sign_command
                        )
                        if intersection_steer is not None:
                            steer_angle = intersection_steer
                            car_speed = 80  # TĂNG từ 80->90 để có đủ lực leo dốc khi rẽ
                        else:
                            # Fallback về lane following nếu không tính được
                            if lane_points:
                                left_x, right_x, mid_x = lane_points
                                last_midpoint = mid_x
                                lane_error = mid_x - center_x
                            elif last_midpoint is not None:
                                lane_error = last_midpoint - center_x
                            steer_angle = lane_pid.compute(lane_error)
                            # THÊM: Phát hiện dốc và điều chỉnh tốc độ
                            slope_boost = detect_steep_slope(
                                lane_error, lane_points, prev_lane_points
                            )
                            base_speed = adaptive_speed_control(lane_error)
                            car_speed = int(base_speed * slope_boost)
                    else:
                        # Chế độ bình thường: CHỈ theo làn đường màu trắng
                        # (Không có biển báo lệnh, có biển unknown, hoặc chưa sẵn sàng rẽ)
                        if lane_points:
                            left_x, right_x, mid_x = lane_points
                            last_midpoint = mid_x
                            lane_error = mid_x - center_x
                        elif last_midpoint is not None:
                            lane_error = last_midpoint - center_x

                        steer_angle = lane_pid.compute(lane_error)
                        # THÊM: Phát hiện dốc và điều chỉnh tốc độ cho chế độ bình thường
                        slope_boost = detect_steep_slope(
                            lane_error, lane_points, prev_lane_points
                        )
                        base_speed = adaptive_speed_control(lane_error)
                        car_speed = int(base_speed * slope_boost)

                    # === DEBUG VISUALIZATION ===
                    # Tạo cửa sổ debug cho segmentation
                    debug_mask = mask_color.copy()

                    # Vẽ các điểm phát hiện giao lộ
                    scan_y_display = 360
                    for point_name, point_info in intersection_info_display.items():
                        if isinstance(point_info, dict) and point_info.get(
                            "detected", False
                        ):
                            center = point_info["center"]
                            color = {
                                "left": (255, 0, 0),  # Xanh dương cho trái
                                "center_left": (255, 255, 0),  # Cyan cho trái-giữa
                                "center": (0, 255, 0),  # Xanh lá cho giữa
                                "center_right": (0, 255, 255),  # Vàng cho phải-giữa
                                "right": (0, 0, 255),  # Đỏ cho phải
                            }.get(point_name, (255, 255, 255))

                            cv2.circle(
                                debug_mask, (center, scan_y_display), 8, color, -1
                            )
                            cv2.putText(
                                debug_mask,
                                point_name[0].upper(),
                                (center - 5, scan_y_display - 15),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                2,
                            )

                    # Vẽ đường scan cho giao lộ
                    cv2.line(
                        debug_mask,
                        (0, scan_y_display),
                        (640, scan_y_display),
                        (128, 128, 128),
                        2,
                    )
                    cv2.putText(
                        debug_mask,
                        f"Intersection Check Y={scan_y_display}",
                        (10, scan_y_display - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (128, 128, 128),
                        1,
                    )

                    # Vẽ đường scan cho lane following
                    cv2.line(debug_mask, (0, 270), (640, 270), (255, 255, 0), 2)
                    cv2.putText(
                        debug_mask,
                        "Lane Following Y=270",
                        (10, 260),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        1,
                    )

                    # Vẽ đường scan sớm (early check)
                    cv2.line(debug_mask, (0, 200), (640, 200), (0, 255, 0), 2)
                    cv2.putText(
                        debug_mask,
                        "Early Check Y=200",
                        (10, 190),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

                    # Vẽ lane points hiện tại
                    if lane_points:
                        left_x, right_x, mid_x = lane_points
                        cv2.circle(debug_mask, (mid_x, 270), 6, (255, 255, 255), -1)
                        cv2.circle(debug_mask, (left_x, 270), 4, (255, 0, 0), -1)
                        cv2.circle(debug_mask, (right_x, 270), 4, (255, 0, 0), -1)
                        # Vẽ error line
                        cv2.line(
                            debug_mask,
                            (center_x, 270),
                            (mid_x, 270),
                            (255, 255, 255),
                            3,
                        )
                        cv2.putText(
                            debug_mask,
                            f"Error: {mid_x-center_x}",
                            (mid_x + 10, 270),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                        )

                    # Vẽ center line
                    cv2.line(
                        debug_mask, (center_x, 0), (center_x, 360), (128, 128, 128), 1
                    )

                    # === YOLO DEBUG ===
                    cam_debug = cam_image.copy()

                    # Vẽ bounding boxes cho traffic signs
                    for detection in detections:
                        bbox = detection["bbox"]
                        class_id = detection["class"]
                        confidence = detection["confidence"]
                        area = detection["area"]

                        x1, y1, x2, y2 = map(int, bbox)
                        color = (0, 255, 0) if area >= 500 else (0, 0, 255)
                        cv2.rectangle(cam_debug, (x1, y1), (x2, y2), color, 2)

                        label = (
                            f"{YOLO_CLASSES[class_id]} {confidence:.2f} A:{int(area)}"
                        )
                        cv2.putText(
                            cam_debug,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            1,
                        )
                    # ✅ Vẽ khung đỏ lên camera khi thấy xe/người
                    for (x1, y1, x2, y2, label, conf) in vehicle_boxes:
                        cv2.rectangle(cam_debug, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(
                                cam_debug, f"{label.upper()} {conf:.2f}",
                                (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                        )
                    # --- 🛑 STOP khi phát hiện xe khác (vehicle box) ---
                    VEHICLE_STOP_TIME = VEHICLE_STOP_SECONDS  # dừng 3 giây

                    # ✅ đảm bảo biến stop_until tồn tại
                    if "stop_until" not in locals():
                        stop_until = None

                    if len(vehicle_boxes) > 0:  # nếu có xe khác trong khung
                        if stop_until is None or time.time() > stop_until:
                            stop_until = time.time() + VEHICLE_STOP_TIME

                    stop_active = (stop_until is not None and time.time() < stop_until)

                    if stop_active:
                        steer_angle = 0
                        car_speed = 0
                        try:
                            client.send_control(steer_angle, car_speed)
                        except Exception:
                            pass    
                    
        


                    # Thêm thông tin state lên video
                    info_lines = [
                        f"State: {'INTERSECTION' if intersection_mode else 'NORMAL'}",
                        f"Speed: {car_speed}",
                        f"Steer: {steer_angle}",
                        f"Command: {traffic_sign_command or 'None'}",
                        f"Sign Detected: {sign_detected}",
                        f"Passed Sign: {passed_sign}",
                        f"Ready to Turn: {ready_to_turn}",
                        f"Lane Count: {intersection_info.get('lane_count', 0)}",
                        f"Is Intersection: {intersection_info.get('is_intersection', False)}",
                        f"Stable Left: {stable_left}",
                        f"Stable Right: {stable_right}",
                        f"Confidence: {avg_confidence:.2f}",
                        f"Current Signs: {current_traffic_signs}",
                    ]

                    for i, line in enumerate(info_lines):
                        cv2.putText(
                            cam_debug,
                            line,
                            (10, 30 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                        )

                    # Hiển thị cửa sổ debug
                    cv2.imshow("Traffic Signs Detection", cam_debug)
                    cv2.imshow("Lane Segmentation Debug", debug_mask)

                    # Thoát nếu nhấn 'q' hoặc ESC
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:  # 27 là ESC
                        break

                    # Cập nhật prev_lane_points cho frame tiếp theo
                    prev_lane_points = lane_points

                except SimulatorConnectionError:
                    raise
                except Exception as e:
                    print(f"Error in processing: {e}")
                    continue

        except KeyboardInterrupt:
            print("\nStopping...")
        except Exception as e:
            print(f"Connection error: {e}")
        finally:
            cv2.destroyAllWindows()  # Đóng tất cả cửa sổ OpenCV
            client.close()

