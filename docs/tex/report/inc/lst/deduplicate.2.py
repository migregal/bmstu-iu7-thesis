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
