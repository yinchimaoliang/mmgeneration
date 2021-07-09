import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN model')
    parser.add_argument('--origin-path', help='Train data path.')
    parser.add_argument('--target-path', help='Target data path.')
    parser.add_argument(
        '--folders',
        default=['he_high', 'Maps1_T', 'Maps3_T', 'Maps4_T'],
        help='Target data path.')
    args = parser.parse_args()

    return args


def main():
    pass
    # args = parse_args()
    # origin_path = args.origin_path
    # target_path = args.target_path
    # folders = args.folders

    # cpy_files(origin_path, target_path, folders)


if __name__ == '__main__':
    main()
