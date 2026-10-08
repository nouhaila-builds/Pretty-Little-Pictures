"""Pedestrian detector trained on real LLVIP infrared.

The detector is trained once, then frozen. Synthetic infrared is a test
condition, not extra training data. Boxes are read from the LLVIP VOC files.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as TF


def read_person_boxes(annotation_path: Path) -> torch.Tensor:
    """Return person boxes as xyxy float tensor. Missing files yield no boxes."""
    if not annotation_path.is_file():
        return torch.zeros((0, 4), dtype=torch.float32)
    root = ET.parse(annotation_path).getroot()
    boxes = []
    for obj in root.findall("object"):
        if (obj.findtext("name") or "").strip() != "person":
            continue
        box = obj.find("bndbox")
        if box is None:
            continue
        values = [float(box.findtext(key)) for key in ("xmin", "ymin", "xmax", "ymax")]
        if values[2] > values[0] and values[3] > values[1]:
            boxes.append(values)
    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32)
    return torch.tensor(boxes, dtype=torch.float32)


def resize_long_side(image: Image.Image, boxes: torch.Tensor, long_side: int):
    width, height = image.size
    scale = long_side / max(width, height)
    resized = image.resize((max(1, int(round(width * scale))), max(1, int(round(height * scale)))), Image.BILINEAR)
    return resized, boxes * scale


class InfraredDetection(Dataset):
    """Real infrared images. The single channel is repeated to three channels."""

    def __init__(self, root: Path, split: str, names: list[str], long_side: int, train: bool = False):
        self.root = root
        self.split = split
        self.names = list(names)
        self.long_side = long_side
        self.train = train

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        name = self.names[index]
        image = Image.open(self.root / "infrared" / self.split / name).convert("RGB")
        boxes = read_person_boxes(self.root / "Annotations" / f"{Path(name).stem}.xml")
        image, boxes = resize_long_side(image, boxes, self.long_side)
        if self.train and torch.rand(1).item() < 0.5:
            image = TF.hflip(image)
            if len(boxes):
                width = image.size[0]
                flipped = boxes.clone()
                flipped[:, 0] = width - boxes[:, 2]
                flipped[:, 2] = width - boxes[:, 0]
                boxes = flipped
        tensor = TF.to_tensor(image)
        target = {
            "boxes": boxes,
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
            "name": name,
        }
        return tensor, target


def detection_collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_detector(pretrained: bool = True) -> torch.nn.Module:
    """COCO-initialized MobileNetV3 Faster R-CNN, retargeted to background + person."""
    weights = "DEFAULT" if pretrained else None
    # weights=None still downloads an ImageNet backbone. The checkpoint load
    # replaces every parameter, so that download is unnecessary.
    backbone = None if weights is None else "DEFAULT"
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights, weights_backbone=backbone)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    return model


def box_iou(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    if len(first) == 0 or len(second) == 0:
        return torch.zeros((len(first), len(second)))
    area_a = (first[:, 2] - first[:, 0]).clamp(min=0) * (first[:, 3] - first[:, 1]).clamp(min=0)
    area_b = (second[:, 2] - second[:, 0]).clamp(min=0) * (second[:, 3] - second[:, 1]).clamp(min=0)
    lt = torch.maximum(first[:, None, :2], second[None, :, :2])
    rb = torch.minimum(first[:, None, 2:], second[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-8)


def average_precision(rows: list[dict], iou_threshold: float = 0.5) -> dict:
    """VOC-style AP at one IoU threshold. rows hold boxes, scores, and ground truth."""
    total_gt = sum(len(row["gt"]) for row in rows)
    detections = []
    for image_id, row in enumerate(rows):
        for box, score in zip(row["boxes"], row["scores"]):
            detections.append((float(score), image_id, box))
    detections.sort(key=lambda item: item[0], reverse=True)
    matched = {image_id: set() for image_id in range(len(rows))}
    tp = []
    fp = []
    for score, image_id, box in detections:
        gt = rows[image_id]["gt"]
        if len(gt) == 0:
            tp.append(0)
            fp.append(1)
            continue
        ious = box_iou(box.unsqueeze(0), gt)[0]
        best = int(torch.argmax(ious))
        if float(ious[best]) >= iou_threshold and best not in matched[image_id]:
            matched[image_id].add(best)
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    if not detections or total_gt == 0:
        return {"ap50": 0.0, "precision_at_0.5": 0.0, "recall_at_0.5": 0.0, "ground_truth": total_gt}
    tp_cum = torch.tensor(tp, dtype=torch.float32).cumsum(0)
    fp_cum = torch.tensor(fp, dtype=torch.float32).cumsum(0)
    recall = tp_cum / total_gt
    precision = tp_cum / (tp_cum + fp_cum)
    recall_points = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    precision_points = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])
    for index in range(len(precision_points) - 1, 0, -1):
        precision_points[index - 1] = torch.maximum(precision_points[index - 1], precision_points[index])
    changing = recall_points[1:] != recall_points[:-1]
    ap = torch.sum((recall_points[1:] - recall_points[:-1])[changing] * precision_points[1:][changing])
    keep = [index for index, item in enumerate(detections) if item[0] >= 0.5]
    if keep:
        last = keep[-1]
        precision_50 = float(precision[last])
        recall_50 = float(recall[last])
    else:
        precision_50 = 0.0
        recall_50 = 0.0
    return {
        "ap50": float(ap),
        "precision_at_0.5": precision_50,
        "recall_at_0.5": recall_50,
        "ground_truth": total_gt,
        "detections": len(detections),
    }
