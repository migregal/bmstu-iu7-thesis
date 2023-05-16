
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
