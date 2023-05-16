import argparse
import os
from pathlib import Path
import time

import cv2
import ray

from method.detect import Detect
from utils import apply_bboxes


def process(
    path: str,
    input: str,
    output: str,
    line_width: int = None,
    color=(200, 128, 128),
    txt_color=(255, 255, 255),
):
    method = Detect(path)

    im = cv2.imread(input)

    bboxes = method.predict(im)

    im = apply_bboxes(im, bboxes, line_width, color, txt_color)

    Path(output).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(os.path.join(output, f"detect-{int(time.time())}.jpeg"), im)
