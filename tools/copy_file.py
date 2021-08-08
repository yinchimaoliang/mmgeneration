import argparse
import os
import random
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN model')
    parser.add_argument('--origin-path', help='Train data path.')
    parser.add_argument('--target-path', help='Target data path.')
    parser.add_argument(
        '--folders',
        default=['seg01', 'seg02', 'seg04'],
        help='Target data path.')
    args = parser.parse_args()

    return args


def cpy_files(origin_path, target_path, folders):
    names = os.listdir(os.path.join(origin_path, 'voted', 'images'))
    for name in names:
        file_paths = []
        for folder in folders:
            file_paths.append(
                os.path.join(origin_path, folder, 'images', name))
        random.shuffle(file_paths)
        shutil.copyfile(file_paths[0], os.path.join(target_path, 'images',
                                                    name))
        print(f'{name} finished')


def main():
    # pass
    args = parse_args()
    origin_path = args.origin_path
    target_path = args.target_path
    folders = args.folders

    cpy_files(origin_path, target_path, folders)


if __name__ == '__main__':
    main()
