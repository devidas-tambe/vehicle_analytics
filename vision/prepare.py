"""Create host-specific DeepStream nvinfer configs from SDK paths."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DS = Path(os.getenv("DEEPSTREAM_ROOT", "/opt/nvidia/deepstream/deepstream"))
OUT = ROOT / "vision/generated"


def prepare():
    p = DS / "samples/models/Primary_Detector"
    primary = Path(
        os.getenv("PRIMARY_ONNX", str(p / "resnet18_trafficcamnet_pruned.onnx"))
    ).resolve()
    labels = Path(os.getenv("PRIMARY_LABELS", str(p / "labels.txt"))).resolve()
    color = Path(
        os.getenv("COLOR_ONNX", str(ROOT / "models/vehicle_color.onnx"))
    ).resolve()
    color_labels = Path(
        os.getenv("COLOR_LABELS", str(ROOT / "models/colors.txt"))
    ).resolve()
    tracker = DS / "samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml"
    lib = DS / "lib/libnvds_nvmultiobjecttracker.so"

    for path in [primary, labels, color, color_labels, tracker, lib]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required asset: {path}. See README.md")

    OUT.mkdir(exist_ok=True)
    vehicle_classes = os.getenv("VEHICLE_CLASS_IDS", "0")

    (OUT / "primary.txt").write_text(
        f"""[property]
gpu-id=0
onnx-file={primary}
labelfile-path={labels}
model-engine-file={OUT}/trafficcamnet_b1_fp16.engine
net-scale-factor=0.00392156862745098
model-color-format=0
batch-size=1
network-mode=2
process-mode=1
network-type=0
num-detected-classes=4
interval=0
gie-unique-id=1
cluster-mode=2
[class-attrs-all]
pre-cluster-threshold=0.3
nms-iou-threshold=0.5
topk=100
""",
        encoding="utf-8",
    )

    (OUT / "color.txt").write_text(
        f"""[property]
gpu-id=0
onnx-file={color}
labelfile-path={color_labels}
model-engine-file={OUT}/color_b16_fp16.engine
batch-size=16
network-mode=2
net-scale-factor=0.00392156862745098
model-color-format=0
network-type=1
process-mode=2
gie-unique-id=2
operate-on-gie-id=1
operate-on-class-ids={vehicle_classes}
classifier-async-mode=0
classifier-threshold=0.25
input-object-min-width=20
input-object-min-height=20
output-blob-names=probabilities
""",
        encoding="utf-8",
    )

    return OUT / "primary.txt", OUT / "color.txt", tracker, lib


if __name__ == "__main__":
    print(*prepare(), sep="\n")