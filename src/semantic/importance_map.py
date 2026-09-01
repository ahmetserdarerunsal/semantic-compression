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


def get_importance_instances(
    image_rgb: np.ndarray,
    conf: float = YOLO_CONF_THRESHOLD,
    important_classes: set[int] | None = IMPORTANT_CLASSES,
) -> list[dict]:
    """Her tespit için AYRI kayıt döner: {"label", "confidence", "mask"}.

    get_importance_mask()'in birleştirilmiş tek maskesinin aksine, bu
    fonksiyon UI'da "tespit edilen nesneler" listesini include/exclude
    edilebilir şekilde göstermek için gereken ham, nesne-başına veriyi
    taşır (mega-spec Part 17). Tek bir YOLO çıkarımı kullanılır — model
    ikinci kez çalıştırılmaz.
    """
    model = _load_model()
    h, w = image_rgb.shape[:2]
    results = model(image_rgb[:, :, ::-1], conf=conf, verbose=False)[0]
    instances: list[dict] = []
    if results.masks is None:
        return instances

    classes = results.boxes.cls.cpu().numpy().astype(int)
    confs = results.boxes.conf.cpu().numpy()
    for inst_mask, cls_id, c in zip(results.masks.data.cpu().numpy(), classes, confs):
        if important_classes is not None and cls_id not in important_classes:
            continue
        resized = cv2.resize(
            inst_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        instances.append(dict(label=results.names[cls_id], confidence=float(c), mask=resized))
    return instances


def fuse_instance_masks(
    instances: list[dict], shape: tuple[int, int], dilate_px: int = MASK_DILATE_PX,
) -> np.ndarray:
    """Seçili örneklerin maskelerini birleştirir (get_importance_mask ile
    AYNI union+dilate mantığı) — hem eski API hem yeni include/exclude UI'ı
    bu TEK fonksiyonu paylaşır, mantık iki yerde tekrar edilmez."""
    mask = np.zeros(shape, dtype=bool)
    for inst in instances:
        mask |= inst["mask"]
    if dilate_px > 0 and mask.any():
        kernel = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    return mask


def get_importance_mask(
    image_rgb: np.ndarray,
    conf: float = YOLO_CONF_THRESHOLD,
    important_classes: set[int] | None = IMPORTANT_CLASSES,
    dilate_px: int = MASK_DILATE_PX,
) -> tuple[np.ndarray, list[str]]:
    """Görüntü için bool önem maskesi ve tespit edilen sınıf adlarını döner.

    Hiç nesne bulunamazsa tamamen False bir maske döner (çağıran taraf bu
    durumda baseline'a düşebilir). Davranış DEĞİŞMEDİ — artık içeride
    get_importance_instances + fuse_instance_masks'i kullanıyor (tek
    kaynaktan, YOLO'yu iki kez çalıştırmadan).
    """
    h, w = image_rgb.shape[:2]
    instances = get_importance_instances(image_rgb, conf, important_classes)
    if not instances:
        return np.zeros((h, w), dtype=bool), []
    mask = fuse_instance_masks(instances, (h, w), dilate_px)
    labels = [inst["label"] for inst in instances]
    return mask, labels
