# Code Map

This repository was reorganized without deleting the original experimental
work.

| Original file | New location | Purpose |
|---|---|---|
| `test2.py` | `src/competition_controller.py` | Main competition controller |
| `model.py` | `src/models/segmentation.py` | DeepLabV3 model definition |
| `camera.py` | `tools/simulator_camera_viewer.py` | Simulator camera viewer |
| `test3.py` | `tools/c270_webcam_test.py` | Logitech C270 camera test |
| `AGV/test.py` | `experiments/opencv_lane_following_controller.py` | OpenCV lane-following experiment |
| `AGV/client.py` | `experiments/segmentation_obstacle_controller.py` | Segmentation and obstacle experiment |
| `test1_main.py` | `legacy/competition_controller_without_obstacle_detection.py` | Older main controller |

The `legacy/` directory is retained only to preserve project history. New
changes should be made in `src/`.
