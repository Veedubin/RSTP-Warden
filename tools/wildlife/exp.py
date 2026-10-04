"""YOLOX-S experiment for the wildlife model: 18 classes, IR-style grayscale augmentation.

Used by YOLOX's own tools (``.yolox/tools/train.py -f exp.py``) and by evaluate.py / export.py.
The dataset lives in data/coco/ as prepare.py writes it (train2017/, val2017/, annotations/).
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for extra in (HERE, HERE / ".yolox"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from wildlife_data import LABELS, grayscale_copy, is_gray  # noqa: E402
from yolox.data import COCODataset, TrainTransform  # noqa: E402
from yolox.exp import Exp as BaseExp  # noqa: E402


class GrayAugCOCODataset(COCODataset):
    """COCODataset whose colour images come back as an IR look-alike part of the time.

    The copy is made in ``load_resized_img``, before mosaic / mixup / HSV, so a mosaic can
    mix grey and colour tiles. Do not use YOLOX's ``--cache``: the cache would freeze one
    random choice per image.
    """

    def __init__(self, *args, gray_prob: float = 0.5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gray_prob = gray_prob
        self._rng = np.random.default_rng()

    def load_resized_img(self, index):
        img = super().load_resized_img(index)
        if self.gray_prob > 0 and random.random() < self.gray_prob and not is_gray(img):
            img = grayscale_copy(img, self._rng)
        return img


class Exp(BaseExp):
    def __init__(self) -> None:
        super().__init__()
        # YOLOX-S
        self.depth = 0.33
        self.width = 0.50
        self.num_classes = len(LABELS)
        self.data_dir = str(HERE / "data" / "coco")
        self.train_ann = "train.json"
        self.val_ann = "val.json"
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.max_epoch = int(os.environ.get("WILDLIFE_EPOCHS", "50"))
        self.no_aug_epochs = 5
        self.warmup_epochs = 2
        self.eval_interval = 5
        self.print_interval = 50
        self.data_num_workers = 6
        self.exp_name = "wildlife_yolox_s"

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        return GrayAugCOCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            img_size=self.input_size,
            preproc=TrainTransform(max_labels=50, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
            cache=cache,
            cache_type=cache_type,
        )
