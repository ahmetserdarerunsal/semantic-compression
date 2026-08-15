# -*- coding: utf-8 -*-
"""YOLO instance segmentation ile piksel-seviyesi önem maskesi üretimi.

Pretrained YOLO-seg modeli (eğitim yok) görüntüdeki nesneleri piksel
maskeleriyle tespit eder; önemli sınıfların maskeleri birleştirilip
hafifçe genişletilerek (dilate) tek bir bool önem haritası üretilir.
"""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from config import (
    IMPORTANT_CLASSES,
    MASK_DILATE_PX,
    YOLO_CONF_THRESHOLD,
    YOLO_MODEL,
)


@lru_cache(maxsize=1)
def _load_model():
    """YOLO-seg modelini bir kez yükler (ilk çağrıda ağırlıklar indirilir).

    Ağırlık dosyası proje kökünde varsa oradan okunur; yoksa ultralytics
    çalışma dizinine indirir (indirilen dosya sonraki koşularda köke taşınabilir).
    """
    from ultralytics import YOLO

    from config import PROJECT_ROOT

    local = PROJECT_ROOT / YOLO_MODEL
    return YOLO(str(local) if local.exists() else YOLO_MODEL)


def get_importance_mask(
    image_rgb: np.ndarray,
    conf: float = YOLO_CONF_THRESHOLD,
    important_classes: set[int] | None = IMPORTANT_CLASSES,
    dilate_px: int = MASK_DILATE_PX,
) -> tuple[np.ndarray, list[str]]:
    """Görüntü için bool önem maskesi ve tespit edilen sınıf adlarını döner.

    Hiç nesne bulunamazsa tamamen False bir maske döner (çağıran taraf bu
    durumda baseline'a düşebilir).
    """
    model = _load_model()
    h, w = image_rgb.shape[:2]
    # ultralytics BGR bekler
    results = model(image_rgb[:, :, ::-1], conf=conf, verbose=False)[0]

    mask = np.zeros((h, w), dtype=bool)
    labels: list[str] = []
    if results.masks is None:
        return mask, labels

    classes = results.boxes.cls.cpu().numpy().astype(int)
    for inst_mask, cls_id in zip(results.masks.data.cpu().numpy(), classes):
        if important_classes is not None and cls_id not in important_classes:
            continue
        resized = cv2.resize(
            inst_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        )
        mask |= resized.astype(bool)
        labels.append(results.names[cls_id])

    if dilate_px > 0 and mask.any():
        kernel = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    return mask, labels
