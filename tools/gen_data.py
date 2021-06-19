import argparse
import os
import os.path as osp

import mmcv
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN model')
    parser.add_argument('--train-path', help='Train data path.')
    parser.add_argument('--valid-path', help='Valid data path.')
    parser.add_argument('--test-path', help='test data path.')
    parser.add_argument('--target-path', help='Target data path.')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    train_path = args.train_path
    valid_path = args.valid_path
    test_path = args.test_path
    target_path = args.target_path
    mmcv.mkdir_or_exist(os.path.join(target_path, 'train'))
    mmcv.mkdir_or_exist(os.path.join(target_path, 'valid'))
    mmcv.mkdir_or_exist(os.path.join(target_path, 'test'))
    train_names = os.listdir(osp.join(train_path, 'images'))
    valid_names = os.listdir(osp.join(valid_path, 'images'))
    test_names = os.listdir(osp.join(test_path, 'images'))
    for train_name in train_names:
        img_a = mmcv.imread(osp.join(train_path, 'images', train_name))
        img_b = mmcv.imread(osp.join(train_path, 'annotations', train_name))
        assert img_a.shape == img_b.shape
        h = img_a.shape[0]
        w = img_a.shape[1]
        img = np.zeros((h, w * 2, 3))
        img[:, :w, :] = img_a
        img[:, w:, :] = img_b
        mmcv.imwrite(img, os.path.join(target_path, 'train', train_name))
        print(f'{train_name} finished')
    for valid_name in valid_names:
        img_a = mmcv.imread(osp.join(valid_path, 'images', valid_name))
        img_b = mmcv.imread(osp.join(valid_path, 'annotations', valid_name))
        assert img_a.shape == img_b.shape
        h = img_a.shape[0]
        w = img_a.shape[1]
        img = np.zeros((h, w * 2, 3))
        img[:, :w, :] = img_a
        img[:, w:, :] = img_b
        mmcv.imwrite(img, os.path.join(target_path, 'valid', valid_name))
        print(f'{valid_name} finished')
    for test_name in test_names:
        img_a = mmcv.imread(osp.join(test_path, 'images', test_name))
        img_b = mmcv.imread(osp.join(test_path, 'annotations', test_name))
        assert img_a.shape == img_b.shape
        h = img_a.shape[0]
        w = img_a.shape[1]
        img = np.zeros((h, w * 2, 3))
        img[:, :w, :] = img_a
        img[:, w:, :] = img_b
        mmcv.imwrite(img, os.path.join(target_path, 'test', test_name))
        print(f'{test_name} finished')


if __name__ == '__main__':
    main()
