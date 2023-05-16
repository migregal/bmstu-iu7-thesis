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
