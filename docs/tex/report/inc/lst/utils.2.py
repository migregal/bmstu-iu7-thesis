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
