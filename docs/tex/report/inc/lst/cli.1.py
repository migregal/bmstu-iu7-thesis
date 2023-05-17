if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", default="./experts/", help="Путь к файлу/директории с обученной моделью"
    )
    parser.add_argument(
        "-i", "--image", default="data/file.jpeg", help="Пусть к файлу снимка"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="predicted",
        help="Директория для сохранения результатов",
    )

    args = parser.parse_args()

    ray.init(log_to_driver=False, num_gpus=1)

    process(args.model, args.image, args.output)

    ray.shutdown()
