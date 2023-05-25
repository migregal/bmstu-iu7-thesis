import os

import cv2
import numpy as np
from ultralytics import YOLO
import ray

from method.deduplicate import deduplicate_wbboxes


@ray.remote(num_gpus=0.25)
def apply_expert(image: np.ndarray, expert: YOLO):
    r = expert.predict(image)
    return (
        r[0].boxes.xyxyn.detach().cpu().numpy(),
        r[0].boxes.conf.detach().cpu().numpy(),
    )


class Detect:
    experts: list[YOLO] = []
    weights: np.ndarray

    def __init__(self, path: str, ext: tuple[str, ...] = tuple(".pt")):
        if os.path.isfile(path):
            self.experts += [YOLO(path)]
            return

        fnames = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext)]

        self.experts += [YOLO(f) for f in fnames if os.path.isfile(f)]
        self.weights = np.ones(len(self.experts))

    def predict(self, input: cv2.Mat, w: np.ndarray = None, limit: np.float32 = 0.75) -> None | np.ndarray:
        w = w or self.weights

        if len(w) != len(self.experts):
            return None

        input_id = ray.put(input)
        models = [ray.put(model) for model in self.experts]
        ray_ids = [apply_expert.remote(input_id, models[i]) for i in range(len(models))]
        r = ray.get(ray_ids)

        wbboxes = [
            [w[i], np.array(b), np.float32(c)]
            for i, p in enumerate(r)
            for b, c in zip(*p)
        ]

        return deduplicate_wbboxes(wbboxes)
