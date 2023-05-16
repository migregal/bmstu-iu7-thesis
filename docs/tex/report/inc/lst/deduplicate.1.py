    stack = collections.deque()
    stack.append({"idx": -1, "lst": [], "conf": conf})

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
            if res is None or (np.linalg.norm(c - tmp) < np.linalg.norm(c - cur_c)):
                res, conf, cur_c = intersection, route["conf"], tmp

        if route["idx"] < len(bboxes):
            stack.append(route)
            stack.append(
                {
                    "idx": route["idx"],
                    "lst": route["lst"] + [route["idx"]],
                    "conf": max(route["conf"], [conf["idx"]]),
                }
            )

    return res, conf


def get_wbboxes_intersection_matrix(wbboxes, limit):
    m = {i: [[], []] for i in range(len(wbboxes))}
    for i in range(len(wbboxes)):
        cur = wbboxes[i]
