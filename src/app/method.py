import os

import ray
from ultralytics import YOLO

@ray.remote
def apply_expert(image: str, expert: YOLO):
    r = expert.predict(image)
    return r[0].plot()
    # return r[0].boxes.xyxyn


class Method:
    experts: list[YOLO] = []

    def __init__(self, path: str, ext: tuple[str, ...] = tuple(".pt")):
        if os.path.isfile(path):
            self.experts += [YOLO(path)]
            return

        for fname in os.listdir(path):
            if not fname.endswith(ext):
                continue

            f = os.path.join(path, fname)
            if not os.path.isfile(f):
                continue

            self.experts += [YOLO(f)]

    def predict(self, input: str):
        ray_ids = []
        for model in self.experts:
            ray_ids += [apply_expert.remote(input, model)]

        ret = ray.get(ray_ids)
        # print(ret)
        return ret
