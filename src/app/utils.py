import cv2


def apply_bboxes(
    im: cv2.Mat,
    bboxes,
    line_width: int = None,
    color=(200, 128, 128),
    txt_color=(255, 255, 255),
) -> cv2.Mat:
    lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)
    x, y = im.shape[1], im.shape[0]

    for b, c in bboxes:
        p1 = (int(x * b[0]), int(y * b[1]))
        p2 = (int(x * b[2]), int(y * b[3]))
        cv2.rectangle(im, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)

        label = f"{c:.2f}"
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[
            0
        ]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            im,
            label,
            (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
            0,
            lw / 3,
            txt_color,
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

    return im
