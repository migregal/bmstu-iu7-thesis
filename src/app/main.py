import argparse

import cv2
import matplotlib.pyplot as plt

import ray

from method import Method

def process(path: str, input: str, output: str):
    line_width = None
    color=(200, 128, 128)
    # txt_color=(255, 255, 255)

    method = Method(path)

    im = cv2.imread(input)
    lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)

    # TODO: add preprocessing

    bboxes = method.predict(im)

    x, y = im.shape[1], im.shape[0]
    for bbox in bboxes:
        p1, p2 = (int(x*bbox[0]), int(y*bbox[1])), (int(x*bbox[2]), int(y*bbox[3]))
        cv2.rectangle(im, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)

    cv2.imwrite('test.jpeg', im)


    # if label:
    #     tf = max(self.lw - 1, 1)  # font thickness
    #     w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
    #     outside = p1[1] - h >= 3
    #     p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    #     cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
    #     cv2.putText(self.im,
    #                 label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
    #                 0,
    #                 self.lw / 3,
    #                 txt_color,
    #                 thickness=tf,
    #                 lineType=cv2.LINE_AA)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='./experts/', help='Путь к файлу с обученной моделью')
    parser.add_argument('-i', '--image', default='data/file.jpeg', help='Пусть к файлу снимка')
    parser.add_argument('-o', '--output', default='predicted', help='Директория для сохранения результатов')

    args = parser.parse_args()

    ray.init(log_to_driver=False)

    process(args.model, args.image, args.output)

    ray.shutdown()
