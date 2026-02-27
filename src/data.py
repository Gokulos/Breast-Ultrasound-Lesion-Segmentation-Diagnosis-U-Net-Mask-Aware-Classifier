import os
from glob import glob
import cv2
import numpy as np
from typing import Tuple

from .config import CLASS_MAP

def read_gray(path: str, img_size: int) -> np.ndarray | None:
    """Read grayscale image, resize, normalize to [0,1]."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    return img

def make_empty_mask(img_size: int) -> np.ndarray:
    return np.zeros((img_size, img_size), dtype=np.float32)

def load_busi_dataset(root_dir: str, img_size: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads BUSI dataset from class folders.
    Expected folders:
      root/benign/*.png, root/malignant/*.png, root/normal/*.png
    Masks named *_mask.png (usually benign/malignant; often none for normal)

    Returns:
      X: (N,H,W,1) float32
      Y_mask: (N,H,W,1) float32 binary {0,1}  (empty for normals)
      y_class: (N,) int labels 0..2
      has_mask: (N,) bool  True if *_mask.png exists
    """
    X, Y_mask, y_class, has_mask = [], [], [], []

    for cls_name, cls_id in CLASS_MAP.items():
        cls_dir = os.path.join(root_dir, cls_name)
        if not os.path.isdir(cls_dir):
            print(f"[WARN] Missing folder: {cls_dir}")
            continue

        img_paths = sorted(glob(os.path.join(cls_dir, "*.png")))
        img_paths = [p for p in img_paths if not p.endswith("_mask.png")]

        for img_path in img_paths:
            img = read_gray(img_path, img_size)
            if img is None:
                continue

            mask_path = img_path.replace(".png", "_mask.png")
            if os.path.exists(mask_path):
                mask = read_gray(mask_path, img_size)
                if mask is None:
                    continue
                mask = (mask > 0.5).astype(np.float32)  # binarize
                mask_exists = True
            else:
                mask = make_empty_mask(img_size)
                mask_exists = False

            X.append(img[..., None])
            Y_mask.append(mask[..., None])
            y_class.append(cls_id)
            has_mask.append(mask_exists)

    X = np.array(X, dtype=np.float32)
    Y_mask = np.array(Y_mask, dtype=np.float32)
    y_class = np.array(y_class, dtype=np.int32)
    has_mask = np.array(has_mask, dtype=bool)
    return X, Y_mask, y_class, has_mask
