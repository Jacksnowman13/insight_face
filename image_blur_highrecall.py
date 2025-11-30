#!/usr/bin/env python3
"""
High-recall face blurring using InsightFace + OpenCV.

- Multi-scale detection + NMS for high recall
- Uses GPU (CUDAExecutionProvider) when available, falls back to CPU
"""

import os
import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # We'll just assume CPU if onnxruntime isn't available


os.environ.setdefault("INSIGHTFACE_HOME", os.path.join(os.getcwd(), "models_cache"))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert image array to uint8 [0, 255] if needed."""
    if image is None:
        return None
    if image.dtype == np.uint8:
        return image

    max_value = float(image.max()) if image.size else 255.0
    if max_value <= 0:
        return image.astype(np.uint8)

    scaled = image.astype(np.float32) * (255.0 / max_value)
    return scaled.clip(0, 255).astype(np.uint8)


def expand_bbox_with_padding(
    bbox: Sequence[float], width: int, height: int, pad_ratio: float = 0.20
) -> Optional[Tuple[int, int, int, int]]:
    """Expand a bounding box by pad_ratio, clamped to image bounds."""
    x1, y1, x2, y2 = map(int, bbox)
    pad_x = int((x2 - x1) * pad_ratio)
    pad_y = int((y2 - y1) * pad_ratio)

    x1 -= pad_x
    y1 -= pad_y
    x2 += pad_x
    y2 += pad_y

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width - 1, x2)
    y2 = min(height - 1, y2)

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def apply_gaussian_blur_to_region(
    image: np.ndarray,
    bbox: Sequence[int],
    kernel_size: int = 99,
    pad_ratio: float = 0.20,
) -> np.ndarray:
    """Apply Gaussian blur to a padded face region within the image."""
    height, width = image.shape[:2]
    padded_bbox = expand_bbox_with_padding(bbox, width, height, pad_ratio=pad_ratio)
    if not padded_bbox:
        return image

    if kernel_size % 2 == 0:
        kernel_size += 1  # Gaussian kernel must be odd

    x1, y1, x2, y2 = padded_bbox
    region_of_interest = image[y1:y2, x1:x2]

    if region_of_interest.size:
        image[y1:y2, x1:x2] = cv2.GaussianBlur(
            region_of_interest,
            (kernel_size, kernel_size),
            0,
        )
    return image


def collect_image_paths(paths: Iterable[str]) -> List[Path]:
    """Collect all image file paths from given files and directories."""
    image_paths: List[Path] = []
    for path_string in paths:
        path = Path(path_string)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(path)
        elif path.is_dir():
            for root, _, files in os.walk(path):
                for filename in files:
                    if Path(filename).suffix.lower() in IMAGE_EXTENSIONS:
                        image_paths.append(Path(root) / filename)
    return sorted(image_paths)


def intersection_over_union(
    box_a: Sequence[int], box_b: Sequence[int]
) -> float:
    """Compute IoU between two bounding boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)
    intersection_area = intersection_width * intersection_height

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union_area = area_a + area_b - intersection_area + 1e-6
    return intersection_area / union_area


def non_max_suppression(
    boxes_with_scores: Sequence[Tuple[int, int, int, int, float]],
    iou_threshold: float = 0.5,
) -> List[Tuple[int, int, int, int, float]]:
    """
    Basic NMS on (x1, y1, x2, y2, score) boxes.
    Keeps highest-score boxes and removes overlapping ones.
    """
    boxes_sorted = sorted(
        boxes_with_scores,
        key=lambda x: x[4],
        reverse=True,
    )
    kept: List[Tuple[int, int, int, int, float]] = []

    while boxes_sorted:
        best = boxes_sorted.pop(0)
        kept.append(best)
        boxes_sorted = [
            box
            for box in boxes_sorted
            if intersection_over_union(best[:4], box[:4]) < iou_threshold
        ]

    return kept


def detect_faces_multiscale(
    image_uint8: np.ndarray,
    face_analyzer: FaceAnalysis,
    scales: Sequence[float],
    min_score: Optional[float] = None,
) -> List[Tuple[int, int, int, int]]:
    """
    Run face detection at multiple scales, merge with NMS, and return bounding boxes.
    """
    height, width = image_uint8.shape[:2]
    all_boxes: List[Tuple[int, int, int, int, float]] = []

    for scale in scales:
        if scale == 1.0:
            upsampled = image_uint8
        else:
            upsampled = cv2.resize(
                image_uint8,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_LINEAR,
            )

        faces = face_analyzer.get(upsampled)
        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)

            # Map back to original resolution
            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            x2 = int(x2 / scale)
            y2 = int(y2 / scale)

            score = float(getattr(face, "det_score", 1.0))
            if (min_score is None) or (score >= min_score):
                all_boxes.append((x1, y1, x2, y2, score))

    if not all_boxes:
        return []

    merged_boxes = non_max_suppression(all_boxes, iou_threshold=0.5)
    return [(x1, y1, x2, y2) for (x1, y1, x2, y2, _) in merged_boxes]


def process_single_image(
    input_path: Path,
    output_path: Path,
    face_analyzer: FaceAnalysis,
    blur_kernel_size: int = 99,
    detection_scales: Sequence[float] = (1.0, 1.5, 2.0),
    pad_ratio: float = 0.20,
    min_score: Optional[float] = None,
) -> bool:
    """Load image, detect faces, blur them, and save output."""
    image_raw = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if image_raw is None:
        print(f"[WARN] cannot read: {input_path}")
        return False

    image_uint8 = normalize_to_uint8(image_raw)
    face_boxes = detect_faces_multiscale(
        image_uint8,
        face_analyzer,
        scales=detection_scales,
        min_score=min_score,
    )

    for box in face_boxes:
        image_uint8 = apply_gaussian_blur_to_region(
            image_uint8,
            box,
            kernel_size=blur_kernel_size,
            pad_ratio=pad_ratio,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image_uint8):
        print(f"[WARN] write failed: {output_path}")
        return False

    print(f"[OK] {input_path.name}: faces blurred = {len(face_boxes)}")
    return True


def choose_execution_providers() -> Tuple[Sequence[str], int]:
    """
    Choose ONNX Runtime providers and ctx_id for InsightFace.

    - Prefer CUDAExecutionProvider if available (GPU)
    - Fall back to CPUExecutionProvider
    """
    if ort is None:
        # No onnxruntime import; safest is CPU
        return ["CPUExecutionProvider"], -1

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        # GPU + CPU fallback
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], 0
    else:
        return ["CPUExecutionProvider"], -1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="High-recall face blurring (TIFF/JPG/PNG). "
        "Multi-scale + NMS. Uses GPU if available."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Image files and/or directories",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output root directory",
    )
    parser.add_argument(
        "--blur",
        type=int,
        default=99,
        help="Gaussian kernel size (odd)",
    )
    parser.add_argument(
        "--det",
        type=int,
        default=960,
        help="Detector size (sets internal model size)",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default="1.0,1.5,2.0",
        help="Comma-separated upsample factors",
    )
    parser.add_argument(
        "--pad",
        type=float,
        default=0.20,
        help="Pad ratio around face bounding box",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Keep boxes with score >= this (if available from detector)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    providers, ctx_id = choose_execution_providers()
    print(f"Using providers: {providers}, ctx_id={ctx_id}")

    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=providers,
    )
    # We let multi-scale resizing handle resolution; det_size is the base size.
    face_app.prepare(ctx_id=ctx_id, det_size=(args.det, args.det))

    detection_scales = tuple(
        float(s) for s in args.scales.split(",") if s.strip()
    )

    image_paths = collect_image_paths(args.input)
    print(f"Found {len(image_paths)} image(s).")

    input_roots = [Path(p) for p in args.input]
    output_root = Path(args.output)

    num_ok, num_fail = 0, 0
    for input_path in image_paths:
        # Preserve relative directory structure
        relative_path = None
        for root in input_roots:
            try:
                relative_path = input_path.relative_to(root)
                break
            except ValueError:
                continue

        if relative_path is None:
            relative_path = input_path.name

        output_path = output_root / relative_path

        success = process_single_image(
            input_path=input_path,
            output_path=output_path,
            face_analyzer=face_app,
            blur_kernel_size=args.blur,
            detection_scales=detection_scales,
            pad_ratio=args.pad,
            min_score=args.min_score,
        )
        if success:
            num_ok += 1
        else:
            num_fail += 1

    print(f"Done. OK={num_ok}, FAIL={num_fail}")


if __name__ == "__main__":
    main()
