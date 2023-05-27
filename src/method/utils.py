import functools

import numpy as np


def get_bbox_center(bbox):
    return 0.5 * np.array([bbox[0] + bbox[2], bbox[1] + bbox[3]], dtype=np.float32)


def get_bbox_area(bbox) -> np.float32:
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def get_intersection(bbox_1: np.ndarray, bbox_2: np.ndarray) -> np.ndarray:
    if (
        (bbox_1 is None or bbox_2 is None)
        or (bbox_1[0] > bbox_2[2] or bbox_2[0] > bbox_1[2])
        or (bbox_1[3] < bbox_2[1] or bbox_2[3] < bbox_1[1])
    ):
        return None

    x_l, y_t = max(bbox_1[0], bbox_2[0]), min(bbox_1[1], bbox_2[1])
    x_r, y_b = min(bbox_1[2], bbox_2[2]), max(bbox_1[3], bbox_2[3])
    if x_r < x_l or y_b < y_t:
        return None

    return np.array([x_l, y_t, x_r, y_b])


def get_intersection_of_n_bboxes(bboxes: np.ndarray) -> np.ndarray:
    return functools.reduce(get_intersection, bboxes)


def get_mass_center(bboxes: np.ndarray) -> np.ndarray:
    return np.array([get_bbox_center(bbox) for bbox in bboxes]).mean(axis=0)


def get_iou(bbox_1: np.ndarray, bbox_2: np.ndarray) -> np.float32:
    intersection = get_intersection(bbox_1, bbox_2)
    if intersection is None:
        return np.float32(0.0)

    intersection_area = get_bbox_area(intersection)
    bbx1_area, bbx2_area = get_bbox_area(bbox_1), get_bbox_area(bbox_2)

    return np.float32(
        intersection_area / float(bbx1_area + bbx2_area - intersection_area)
    )


def get_bbox_weight(bbox: np.ndarray) -> np.float32:
    return bbox[0] * bbox[2]
