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
    bboxes: np.ndarray, confs: np.ndarray
) -> tuple[np.ndarray, np.float32]:
    res, conf = bboxes[0], confs[0]
    cur_c, c = get_bbox_center(res), get_mass_center(bboxes)

    stack = collections.deque()
    stack.append({"idx": -1, "lst": [], "conf": conf})

    while stack:
        sub = stack.pop()

        if sub["idx"] == len(bboxes):
            continue

        sub["idx"] += 1

        lst = np.array([bboxes[i] for i in sub["lst"]])
        if len(lst) > 0:
            intersection = get_intersection_of_n_bboxes(lst)
            if intersection is None:
                continue

            tmp = get_bbox_center(intersection)
            if np.linalg.norm(c - tmp) < np.linalg.norm(c - cur_c):
                res, conf, cur_c = intersection, sub["conf"], tmp

        if sub["idx"] < len(bboxes):
            stack.append(sub)
            stack.append(
                {
                    "idx": sub["idx"],
                    "lst": sub["lst"] + [sub["idx"]],
                    "conf": max(sub["conf"], confs["idx"]),
                }
            )

    return res, conf


def get_wbboxes_intersection_matrix(wbboxes: list, limit: np.float32 = 0.75) -> dict:
    m = {i: [[], []] for i in range(len(wbboxes))}
    for i in range(len(wbboxes)):
        cur = wbboxes[i]

        lst = []
        for j in range(i + 1, len(wbboxes)):
            if limit <= iou(cur[1], wbboxes[j][1]):
                lst += [j]

            if weight(cur) < weight(wbboxes[j]):
                m.pop(i, None)
                lst = []
                break

        for j in lst:
            if next((True for k in m[j][1] if j in m[k][0]), False):
                continue

            m[i], m[j] = [m[i][0] + [j], m[i][1]], [m[j][0], m[j][1] + [i]]

    return m


def deduplicate_wbboxes(wbboxes: list, limit: np.float32 = 0.75) -> np.ndarray:
    m = get_wbboxes_intersection_matrix(wbboxes, limit)

    bboxes = []
    for i in m.keys():
        cur = wbboxes[i]
        if len(m[i][0]) == 0 and len(m[i][1]) == 0:
            bboxes += [(cur[1], cur[2])]
            continue

        if len(m[i][0]) == 0:
            continue

        lst, skip = [], False
        for j in m[i][0]:
            if weight(cur) == weight(wbboxes[j]):
                lst += [j]

            if weight(cur) < weight(wbboxes[j]):
                skip = False
                break

        if skip:
            continue

        if len(lst) == 0:
            bboxes += [(cur[1], cur[2])]
            continue

        lst = np.array([cur[1]] + [wbboxes[j][1] for j in lst])
        conf = np.array([cur[2]] + [wbboxes[j][2] for j in lst])
        bboxes += [(*find_closest_intersection(lst, conf),)]

    return bboxes
