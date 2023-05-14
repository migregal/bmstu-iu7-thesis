import argparse

from method import Method

def process(path: str, image: str, output: str):
    method = Method(path)

    # TODO: add preprocessing

    method.predict(image, output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='./experts/', help='Путь к файлу с обученной моделью')
    parser.add_argument('-i', '--image', default='data/file.jpeg', help='Пусть к файлу снимка')
    parser.add_argument('-o', '--output', default='predicted', help='Директория для сохранения результатов')

    args = parser.parse_args()

    process(args.model, args.image, args.output)
