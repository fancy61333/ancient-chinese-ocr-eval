#!/usr/bin/env python3
"""Two-stage inference for ancient Chinese character recognition.

Detector: YOLOv8n (bounding boxes + confidence)
Classifier: ResNet18 with a closed 4,105-way head (one class per character category)

This is the inference script of the system reported in

    Graphemic variation and the limits of shape-based identification in
    oracle-bone and bronze inscriptions

released with paths parameterised (the platform-fixed input and output
locations are replaced by arguments) and comments translated. The logic is unchanged:
per image, detect boxes, drop boxes outside a size window, crop with proportional
padding, resize, classify by argmax, discard predictions below the classifier
confidence floor, and write ``{image_id: [{bbox, text}, ...]}`` to JSON.

The three required artefacts are NOT in this repository:
    detection_model.pt, classifier_model.pt, id_to_char.json
They are derived from shared-task data whose licence does not permit redistribution.

Usage inside the shared-task container (default paths are the platform's):
    python run_inference_yolo.py
Local use:
    python run_inference_yolo.py --input /path/to/images \
        --output prediction.json --model-dir /path/to/weights
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torchvision import models
from ultralytics import YOLO

# Values of the submitted run; see config/inference_config.yaml.
DEFAULT_CONF = 0.405           # detector confidence floor
DEFAULT_CROP_PADDING = 0.05    # fraction of box width/height; the reference script ships 0.0
DEFAULT_CLS_CONF = 0.10        # classifier confidence floor
DEFAULT_IMG_SIZE = 64          # classifier input size
DEFAULT_MIN_BOX = 8            # pixels
DEFAULT_MAX_BOX = 500          # pixels
NUM_CLASSES = 4105


def set_seed(seed=2026):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_id_to_char(mapping_path: Path) -> dict:
    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_classifier(checkpoint_path: Path, num_classes: int, device: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def crop_with_padding(img, bbox, padding=0.0):
    x, y, w, h = bbox
    img_h, img_w = img.shape[:2]

    pad_w = int(w * padding)
    pad_h = int(h * padding)

    x_start = max(0, x - pad_w)
    y_start = max(0, y - pad_h)
    x_end = min(img_w, x + w + pad_w)
    y_end = min(img_h, y + h + pad_h)

    return img[y_start:y_end, x_start:x_end]


def parse_args():
    ap = argparse.ArgumentParser(description="YOLOv8n + ResNet18 inference")
    ap.add_argument("--input", default="/path/to/eval/images")
    ap.add_argument("--output", default="./prediction.json")
    ap.add_argument("--model-dir", default=None,
                    help="Directory holding detection_model.pt, classifier_model.pt, "
                         "id_to_char.json. Defaults to ./models next to this script.")
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    ap.add_argument("--crop-padding", type=float, default=DEFAULT_CROP_PADDING)
    ap.add_argument("--cls-conf", type=float, default=DEFAULT_CLS_CONF)
    ap.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    ap.add_argument("--min-box", type=int, default=DEFAULT_MIN_BOX)
    ap.add_argument("--max-box", type=int, default=DEFAULT_MAX_BOX)
    ap.add_argument("--nms-iou", type=float, default=None,
                    help="If unset, the detector library default is used. The submitted "
                         "configuration record lists 0.45, but the submitted call did not "
                         "pass `iou`, so the library default was in effect.")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed()

    input_dir = Path(args.input)
    output_file = Path(args.output)

    if not input_dir.exists():
        print(f"FAIL input directory not found: {input_dir}")
        raise SystemExit(2)

    model_dir = Path(args.model_dir) if args.model_dir else Path(__file__).resolve().parent / "models"
    det_model_path = model_dir / "detection_model.pt"
    cls_model_path = model_dir / "classifier_model.pt"
    char_mapping_path = model_dir / "id_to_char.json"

    print("Ancient character OCR - YOLOv8n + ResNet18")
    print(f"  input            : {input_dir}")
    print(f"  output           : {output_file}")
    print(f"  conf             : {args.conf}")
    print(f"  crop padding     : {args.crop_padding}")
    print(f"  cls conf floor   : {args.cls_conf}")
    print(f"  classifier input : {args.img_size}")
    print(f"  nms iou          : {args.nms_iou if args.nms_iou is not None else 'library default'}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    missing = [str(p) for p in (det_model_path, cls_model_path, char_mapping_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing required model files: " + ", ".join(missing))

    detector = YOLO(str(det_model_path))
    id_to_char = load_id_to_char(char_mapping_path)
    num_classes = max(int(k) for k in id_to_char.keys()) + 1
    num_classes = max(num_classes, NUM_CLASSES)
    classifier = load_classifier(cls_model_path, num_classes, device)

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

    def transform(img):
        tensor = torch.from_numpy(img.transpose(2, 0, 1).copy()).float().div(255.0).unsqueeze(0)
        return (tensor - mean) / std

    det_kwargs = dict(verbose=False, conf=args.conf)
    if args.nms_iou is not None:
        det_kwargs["iou"] = args.nms_iou

    img_paths = sorted(input_dir.rglob("*.png"))
    results = {}
    img_count = 0
    total_chars = 0

    for img_path in img_paths:
        image_id = img_path.stem
        img_count += 1
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: could not read {img_path}")
                results[image_id] = []
                continue

            char_results = []
            for result in detector(img, **det_kwargs):
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    det_conf = float(box.conf[0])
                    w, h = x2 - x1, y2 - y1

                    if w < args.min_box or h < args.min_box:
                        continue
                    if w > args.max_box or h > args.max_box:
                        continue

                    char_img = crop_with_padding(img, [x1, y1, w, h], padding=args.crop_padding)
                    if char_img.size == 0:
                        continue

                    char_img_rgb = cv2.cvtColor(char_img, cv2.COLOR_BGR2RGB)
                    char_img_resized = cv2.resize(
                        char_img_rgb, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR
                    )
                    tensor = transform(char_img_resized).to(device)

                    with torch.no_grad():
                        output = classifier(tensor)
                        prob = nn.functional.softmax(output, dim=1)
                        top1_idx = torch.argmax(prob, dim=1).item()
                        cls_conf = prob[0, top1_idx].item()
                        char_text = id_to_char.get(str(top1_idx), str(top1_idx))

                    if cls_conf < args.cls_conf:
                        continue

                    # The reported system scores confidence as the product of the two
                    # probabilities; the submission schema carries bbox and text only.
                    char_results.append({"bbox": [x1, y1, w, h], "text": char_text,
                                         "confidence": det_conf * cls_conf})

            results[image_id] = [{"bbox": p["bbox"], "text": p["text"]} for p in char_results]
            total_chars += len(char_results)

            if img_count % 50 == 0:
                print(f"Processed {img_count} images, detected {total_chars} characters")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            results[image_id] = []

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Processed {img_count} images; {total_chars} characters written to {output_file}")


if __name__ == "__main__":
    main()
