import os

import numpy as np
from ultralytics import YOLO
import ray

from method.utils import deduplicate_wbboxes


@ray.remote(num_gpus=0.25)
def apply_expert(image: np.ndarray, expert: YOLO) -> np.ndarray:
    results = expert.predict(image)
    return np.array(results[0].boxes.xyxyn)


class Detect:
    experts: list[YOLO] = []

    def __init__(self, path: str, ext: tuple[str, ...] = tuple(".pt")):
        if os.path.isfile(path):
            self.experts += [YOLO(path)]
            return

        fnames = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext)]

        self.experts += [YOLO(f) for f in fnames if os.path.isfile(f)]

    def predict(self, input: np.ndarray) -> np.ndarray:
        ray_ids = [apply_expert.remote(input, model) for model in self.experts]
        experts_res = ray.get(ray_ids)

        wbboxes = [[0.5, bbox] for pred in experts_res for bbox in pred]
        return deduplicate_wbboxes(wbboxes)
