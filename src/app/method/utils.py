import numpy as np


def get_bbox_center(bbox):
    return 0.5 * np.array([bbox[0] + bbox[2], bbox[1] + bbox[3]], dtype=np.float32)


def get_bbox_area(bbox) -> np.float32:
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def get_intersection(bbox_1: np.ndarray, bbox_2: np.ndarray) -> None | np.ndarray:
    if bbox_1[0] > bbox_2[2] or bbox_2[0] > bbox_1[2]:
        return None

    if bbox_1[3] < bbox_2[1] or bbox_2[3] < bbox_1[1]:
        return None

    x_l, y_t = max(bbox_1[0], bbox_2[0]), min(bbox_1[1], bbox_2[1])
    x_r, y_b = min(bbox_1[2], bbox_2[2]), max(bbox_1[3], bbox_2[3])
    if x_r < x_l or y_b < y_t:
        return None

    return np.array([x_l, y_t, x_r, y_b])


def get_intersection_of_n_bboxes(bboxes: np.ndarray) -> None | np.ndarray:
    res = bboxes[0]
    for i in range(1, len(bboxes)):
        res = get_intersection(res, bboxes[i])
        if res is None:
            break

    return res


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


def get_bbox_weight(bbox):
    # return bbox[0] * bbox[2]
    return bbox[0]


def deduplicate_wbboxes(wbboxes, limit: float = 0.75):
    bboxes = []

    for i in range(len(wbboxes)):
        cur, skip = wbboxes[i], False
        for j in range(i + 1, len(wbboxes)):
            other = wbboxes[j]
            iou = get_iou(cur[1], other[1])
            if iou < limit:
                continue

            if cur[0] * cur[2] < other[0] * other[2]:
                skip = True
                break

            tmp = get_intersection(cur[1], other[1])
            if get_bbox_area(tmp) < get_bbox_area(cur[1]):
                cur[1] = tmp
                cur[2] = max(cur[2], other[2])

        if skip:
            continue

        for j in range(len(bboxes)):
            bbox = bboxes[j]
            if limit < get_iou(cur[1], bbox[0]):
                skip = True
                break

        if not skip:
            bboxes += [(cur[1], cur[2])]

    return bboxes
