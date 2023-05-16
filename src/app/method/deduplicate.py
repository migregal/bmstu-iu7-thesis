import collections

import numpy as np

from method.utils import (
    get_mass_center,
    get_intersection_of_n_bboxes,
    get_bbox_center,
    get_bbox_weight,
    get_iou,
)


def find_closest_intersection(bboxes) -> np.ndarray:
    res, res_c, c = None, None, get_mass_center(bboxes)

    stack = collections.deque()

    stack.append(
        {
            "idx": -1,
            "lst": [],
        }
    )

    while stack:
        route = stack.pop()

        if route["idx"] == len(bboxes):
            continue

        route["idx"] += 1

        lst = np.array([bboxes[i] for i in route["lst"]])
        if len(lst) > 0:
            intersection = get_intersection_of_n_bboxes(lst)
            if intersection is None:
                continue

            tmp = get_bbox_center(intersection)
            if res is None or (np.linalg.norm(c - tmp) < np.linalg.norm(c - res_c)):
                res, res_c = intersection, tmp

        if route["idx"] < len(bboxes):
            stack.append(route)
            stack.append(
                {
                    "idx": route["idx"],
                    "lst": route["lst"] + [route["idx"]],
                }
            )

    return res


def deduplicate_wbboxes_2(wbboxes, limit: float = 0.75):
    wbboxes = [wbboxes[2]] + wbboxes[6:]

    m = {i: [[], []] for i in range(len(wbboxes))}
    for i in range(len(wbboxes)):
        for j in range(i + 1, len(wbboxes)):
            # if get_bbox_weight(wbboxes[i]) < get_bbox_weight(wbboxes[j]):
            #     print(' <', i, m.pop(i, None))
            #     break

            if limit > get_iou(wbboxes[i][1], wbboxes[j][1]):
                continue

            if len([1 for k in m[j][1] if j in m[k][0]]) > 0:
                continue

            m[i], m[j] = [m[i][0] + [j], m[i][1]], [m[j][0], m[j][1] + [i]]

    bboxes = []
    for i in range(len(wbboxes)):
        cur = wbboxes[i]
        if len(m[i][0]) == 0 and len(m[i][1]) == 0:
            bboxes += [(cur[1], cur[2])]
            continue

        if len(m[i][0]) == 0:
            continue

        lst = list(
            filter(
                lambda j: get_bbox_weight(cur) <= get_bbox_weight(wbboxes[j]), m[i][0]
            )
        )
        if len(lst) == 0:
            bboxes += [(cur[1], cur[2])]
            continue

        lst.sort(key=lambda j: wbboxes[j][0], reverse=True)
        if get_bbox_weight(wbboxes[lst[0]]) > get_bbox_weight(cur):
            continue

        tmp = np.array([wbboxes[j][1] for j in lst])
        bboxes += [
            (
                find_closest_intersection(tmp),
                np.max([cur[2]] + [wbboxes[j][2] for j in lst]),
            )
        ]

    return bboxes
