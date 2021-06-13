import argparse
import glob
import os
import random

import mmcv
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN model')
    parser.add_argument('--ori-path', help='Original data path.')
    parser.add_argument('--target-path', help='Target data path.')
    parser.add_argument('--train-ratio', default=0.6)
    parser.add_argument('--valid-ratio', default=0.2)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    ori_path = args.ori_path
    target_path = args.target_path
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    img_paths = glob.glob(os.path.join(ori_path, '*.jpg'))
    random.shuffle(img_paths)
    mmcv.mkdir_or_exist(os.path.join(target_path, 'train'))
    mmcv.mkdir_or_exist(os.path.join(target_path, 'valid'))
    mmcv.mkdir_or_exist(os.path.join(target_path, 'test'))
    train_img_paths = img_paths[:int(len(img_paths) * train_ratio)]
    valid_img_paths = train_img_paths[
        int(len(train_img_paths) *
            train_ratio):int(len(train_img_paths) * train_ratio) +
        int(len(train_img_paths) * valid_ratio)]
    test_img_paths = train_img_paths[int(len(train_img_paths) * train_ratio) +
                                     int(len(train_img_paths) * valid_ratio):]
    for train_img_path in train_img_paths:
        img_a = mmcv.imread(train_img_path)
        img_b = mmcv.imread(train_img_path.replace('jpg', 'png'))
        assert img_a.shape == img_b.shape
        h = img_a.shape[0]
        w = img_a.shape[1]
        img = np.zeros((h, w * 2, 3))
        img[:, :w, :] = img_a
        img[:, w:, :] = img_b
        mmcv.imwrite(
            img,
            os.path.join(target_path, 'train',
                         os.path.split(train_img_path)[-1]))
        print(f'{os.path.split(train_img_path)[-1]} finished')
    for valid_img_path in valid_img_paths:
        img_a = mmcv.imread(valid_img_path)
        img_b = mmcv.imread(valid_img_path.replace('jpg', 'png'))
        assert img_a.shape == img_b.shape
        h = img_a.shape[0]
        w = img_a.shape[1]
        img = np.zeros((h, w * 2, 3))
        img[:, :w, :] = img_a
        img[:, w:, :] = img_b
        mmcv.imwrite(
            img,
            os.path.join(target_path, 'valid',
                         os.path.split(valid_img_path)[-1]))
        print(f'{os.path.split(valid_img_path)[-1]} finished')
    for test_img_path in test_img_paths:
        img_a = mmcv.imread(test_img_path)
        img_b = mmcv.imread(test_img_path.replace('jpg', 'png'))
        assert img_a.shape == img_b.shape
        h = img_a.shape[0]
        w = img_a.shape[1]
        img = np.zeros((h, w * 2, 3))
        img[:, :w, :] = img_a
        img[:, w:, :] = img_b
        mmcv.imwrite(
            img,
            os.path.join(target_path, 'test',
                         os.path.split(test_img_path)[-1]))
        print(f'{os.path.split(test_img_path)[-1]} finished')


if __name__ == '__main__':
    main()
