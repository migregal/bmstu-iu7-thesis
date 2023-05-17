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
