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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", default="./experts/", help="Путь к файлу/директории с обученной моделью"
    )
    parser.add_argument(
        "-i", "--image", default="data/file.jpeg", help="Пусть к файлу снимка"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="predicted",
        help="Директория для сохранения результатов",
    )

    args = parser.parse_args()

    ray.init(log_to_driver=True, num_gpus=1)
    # ray.init(log_to_driver=True, address='172.25.185.82:6379')
    # ray.init(log_to_driver=True, address='192.168.31.201:10001')
    # ray.init('ray://192.168.31.201:10001', log_to_driver=True)

    process(args.model, args.image, args.output)

    ray.shutdown()
