import collections

import numpy as np

from method.utils import (
    get_mass_center,
    get_intersection_of_n_bboxes,
    get_bbox_center,
    get_bbox_weight as weight,
    get_iou as iou,
)


def find_closest_intersection(
    bboxes: np.ndarray, conf: np.ndarray
) -> tuple[np.ndarray, np.float32]:
    res, conf = None, np.float32(0.0)
    cur_c, c = None, None, get_mass_center(bboxes)
