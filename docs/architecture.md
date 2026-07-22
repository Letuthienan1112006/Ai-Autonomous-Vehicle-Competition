# System Architecture

```text
Unity Simulator
    │  JSON image frames over TCP
    ▼
Python Competition Controller
    ├── DeepLabV3-ResNet50 road segmentation
    ├── Custom YOLO traffic-sign detection
    ├── YOLOv8 person/vehicle detection
    ├── Intersection and navigation state logic
    ├── PID steering control
    └── Adaptive speed / stop logic
    │  steering_angle speed
    ▼
Unity Simulator Vehicle
```

The main controller exchanges commands and camera frames with the simulator
at `127.0.0.1:54321` by default. The host and port are configurable through
environment variables.
