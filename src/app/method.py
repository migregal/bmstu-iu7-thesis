import os

import numpy as np

from ultralytics import YOLO

import ray


@ray.remote
def apply_expert(image: np.ndarray, expert: YOLO) -> np.ndarray:
    r = expert.predict(image)
    return np.array(r[0].boxes.xyxyn)


def get_bbox_area(bbox) -> np.float32:
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def get_intersection_coords(bbox_1: np.ndarray, bbox_2: np.ndarray) -> np.ndarray:
    x_l, y_t = max(bbox_1[0], bbox_2[0]), max(bbox_1[1], bbox_2[1])
    x_r, y_b = min(bbox_1[2], bbox_2[2]), min(bbox_1[3], bbox_2[3])

    return np.array([x_l, y_t, x_r, y_b])


def get_iou(bbox_1: list[float], bbox_2) -> np.float32:
    x_l, y_t, x_r, y_b = get_intersection_coords(bbox_1, bbox_2)
    if x_r < x_l or y_b < y_t:
        return np.float32(0.0)

    intersection_area = (x_r - x_l) * (y_b - y_t)
    bbx1_area, bbx2_area = get_bbox_area(bbox_1), get_bbox_area(bbox_2)

    return np.float32(
        intersection_area / float(bbx1_area + bbx2_area - intersection_area)
    )


def deduplicate_wbboxes(wbboxes, limit: float = 0.75) -> np.ndarray:
    bboxes = []

    for i in range(len(wbboxes)):
        cur, skip = wbboxes[i], False
        for j in range(i + 1, len(wbboxes)):
            other = wbboxes[j]
            iou = get_iou(cur[1], other[1])
            if iou < limit:
                continue

            if cur[0] < other[0]:
                skip = True
                continue

            tmp = get_intersection_coords(cur[1], other[1])
            if get_bbox_area(tmp) < get_bbox_area(cur[1]):
                cur[1] = tmp

        if skip:
            continue

        for j in range(len(bboxes)):
            bbox = bboxes[j]
            if limit < get_iou(cur[1], bbox):
                skip = True
                break

        if not skip:
            bboxes += [cur[1]]

    return np.array(bboxes)


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

    def predict(self, input: np.ndarray) -> np.ndarray:
        ray_ids = []
        for model in self.experts:
            ray_ids += [apply_expert.remote(input, model)]

        experts_res = ray.get(ray_ids)

        wbboxes = []
        for prev in experts_res:
            wbboxes += [[0.5, bbox] for bbox in prev]

        return deduplicate_wbboxes(wbboxes)
