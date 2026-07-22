# show_camera_only.py
import socket as sk
import json
import base64
import numpy as np
import cv2

SIM_HOST = "127.0.0.1"
SIM_PORT = 54321

# Gửi lệnh điều khiển mặc định (giữ nguyên/đứng yên)
DEFAULT_STEER = 0
DEFAULT_SPEED = 0   # nếu cần xe lăn bánh để có khung hình, đổi thành >0 (vd 60)

def recv_json_stream(sock, str_buf, max_read=65536):
    """
    Đọc từ TCP stream tới khi parse được 1 JSON hoàn chỉnh.
    Trả về: (obj, new_str_buf)
    """
    decoder = json.JSONDecoder()
    while True:
        s = str_buf.lstrip()
        if s:
            try:
                obj, idx = decoder.raw_decode(s)
                rest = s[idx:]
                return obj, rest
            except json.JSONDecodeError:
                pass
        chunk = sock.recv(max_read)
        if not chunk:
            raise ConnectionError("Socket closed by peer")
        # simulator gửi UTF-8; nếu có byte lẻ thì bỏ qua
        str_buf += chunk.decode("utf-8", errors="ignore")

def main():
    print("Showing front camera only. Press ESC or q to quit.")

    sock = sk.socket(sk.AF_INET, sk.SOCK_STREAM)
    sock.connect((SIM_HOST, SIM_PORT))

    str_buf = ""

    try:
        while True:
            # gửi điều khiển tối thiểu (để simulator trả khung hình)
            msg = f"{DEFAULT_STEER} {DEFAULT_SPEED}"
            sock.sendall(msg.encode("utf-8"))

            # nhận 1 JSON hoàn chỉnh
            recv_data, str_buf = recv_json_stream(sock, str_buf)

            # lấy ảnh base64 -> numpy -> BGR
            jpg_b64 = recv_data.get("Img")
            if not jpg_b64:
                continue
            jpg_data = base64.b64decode(jpg_b64)
            jpg_array = np.frombuffer(jpg_data, dtype=np.uint8)
            frame = cv2.imdecode(jpg_array, flags=1)  # BGR

            if frame is None:
                continue

            cv2.imshow("Front Camera", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # ESC
                break

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        sock.close()

if __name__ == "__main__":
    main()
