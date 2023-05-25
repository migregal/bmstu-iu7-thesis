import argparse
import os
from pathlib import Path

from method.train import Train


def process(
    path: str,
    epochs: int,
):
    if epochs <= 0:
        print("Invalid epochs count")
        return

    if path is None or not os.path.exists(path) or not os.path.isdir(path):
        print("Can't find dataset")
        return

    method = Train(path)

    method.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data", help="Пусть к набору данных"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        help="Число эпох для обучения",
    )

    args = parser.parse_args()

    process(args.data, args.epochs)
