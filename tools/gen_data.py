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
    parser.add_argument(
        '--ann-folders',
        type=list,
        default=['Maps1_T', 'Maps3_T', 'Maps4_T', 'he_high'],
        help='Annotation folders')
    parser.add_argument(
        '--num-classes', type=int, default=4, help='Number of classes.')
    parser.add_argument('--target-path', help='Target data path.')
    parser.add_argument('--type', default='label', help='Type of label.')
    args = parser.parse_args()

    return args


def gen_label(train_path, valid_path, test_path, target_path):
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


def gen_label_onehot(train_path, valid_path, test_path, target_path,
                     ann_folders, num_classes):
    mmcv.mkdir_or_exist(os.path.join(target_path, 'train'))
    mmcv.mkdir_or_exist(os.path.join(target_path, 'valid'))
    mmcv.mkdir_or_exist(os.path.join(target_path, 'test'))
    train_names = os.listdir(osp.join(train_path, 'images'))
    valid_names = os.listdir(osp.join(valid_path, 'images'))
    test_names = os.listdir(osp.join(test_path, 'images'))
    for train_name in train_names:
        img_a = mmcv.imread(osp.join(train_path, 'images', train_name))
        img_a = mmcv.imresize(img_a,
                              (img_a.shape[1] // 4, img_a.shape[0] // 4))
        img_b_one_hot = np.zeros((img_a.shape[0], img_a.shape[1], num_classes),
                                 dtype=np.float)
        for ann_folders in ann_folders:
            img_b = mmcv.imread(osp.join(train_path, 'he_high', train_name), 0)
            img_b = mmcv.imresize(img_b,
                                  (img_b.shape[1] // 4, img_b.shape[0] // 4))
            img_b_one_hot += np.eye(4)[img_b]
        mmcv.imwrite(img_a, os.path.join(target_path, 'train', train_name))
        np.save(
            os.path.join(target_path, 'train',
                         train_name.replace('png', 'npy')), img_b_one_hot)
        print(f'{train_name} finished')
    for valid_name in valid_names:
        img_a = mmcv.imread(osp.join(valid_path, 'images', valid_name))
        img_a = mmcv.imresize(img_a,
                              (img_a.shape[1] // 4, img_a.shape[0] // 4))
        img_b_one_hot = np.zeros((img_a.shape[0], img_a.shape[1], num_classes),
                                 dtype=np.float)
        for ann_folders in ann_folders:
            img_b = mmcv.imread(osp.join(valid_path, 'he_high', valid_name), 0)
            img_b = mmcv.imresize(img_b,
                                  (img_b.shape[1] // 4, img_b.shape[0] // 4))
            img_b_one_hot += np.eye(4)[img_b]
        mmcv.imwrite(img_a, os.path.join(target_path, 'valid', valid_name))
        np.save(
            os.path.join(target_path, 'valid',
                         valid_name.replace('png', 'npy')), img_b_one_hot)
        print(f'{valid_name} finished')
    for test_name in test_names:
        img_a = mmcv.imread(osp.join(test_path, 'images', test_name))
        img_a = mmcv.imresize(img_a,
                              (img_a.shape[1] // 4, img_a.shape[0] // 4))
        img_b_one_hot = np.zeros((img_a.shape[0], img_a.shape[1], num_classes),
                                 dtype=np.float)
        for ann_folders in ann_folders:
            img_b = mmcv.imread(osp.join(test_path, 'he_high', test_name), 0)
            img_b = mmcv.imresize(img_b,
                                  (img_b.shape[1] // 4, img_b.shape[0] // 4))
            img_b_one_hot += np.eye(4)[img_b]
        mmcv.imwrite(img_a, os.path.join(target_path, 'test', test_name))
        np.save(
            os.path.join(target_path, 'test', test_name.replace('png', 'npy')),
            img_b_one_hot)
        print(f'{test_name} finished')


def main():
    args = parse_args()
    train_path = args.train_path
    valid_path = args.valid_path
    test_path = args.test_path
    target_path = args.target_path
    ann_folders = args.ann_folders
    num_classes = args.num_classes
    type = args.type
    if type == 'label':
        gen_label(train_path, valid_path, test_path, target_path)
    else:
        gen_label_onehot(train_path, valid_path, test_path, target_path,
                         ann_folders, num_classes)


if __name__ == '__main__':
    main()
