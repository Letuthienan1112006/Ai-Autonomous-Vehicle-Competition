# Model Weights

The original repository referenced model files outside the project at
`/mnt/big/`. Those hard-coded paths have been removed.

Place the project-specific files here:

```text
weights/
├── segmentation.tar
└── detectsign.pt
```

The object detector defaults to `yolov8n.pt`. Ultralytics may download it
automatically when internet access is available. For offline use, place the
file locally and set `OBJECT_DETECTION_WEIGHTS` in the environment.

Model weights are intentionally excluded from Git because they can be large
and may have separate sharing permissions.
