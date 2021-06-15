import argparse
import os
import sys

import mmcv
from mmcv import DictAction
from torchvision import utils

# yapf: disable
sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))  # isort:skip  # noqa

from mmgen.apis import init_model, sample_img2img_model  # isort:skip  # noqa
# yapf: enable


def parse_args():
    parser = argparse.ArgumentParser(description='Translation demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('image_path', help='Image file path')
    parser.add_argument(
        '--save-path',
        type=str,
        default='./work_dirs/demos',
        help='path to save translation sample')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CUDA device id')
    # args for inference/sampling
    parser.add_argument(
        '--sample-cfg',
        nargs='+',
        action=DictAction,
        help='Other customized kwargs for sampling function')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model = init_model(
        args.config, checkpoint=args.checkpoint, device=args.device)

    if args.sample_cfg is None:
        args.sample_cfg = dict()

    img_names = os.listdir(args.image_path)
    for i, img_name in enumerate(img_names):

        results = sample_img2img_model(model,
                                       os.path.join(args.image_path, img_name),
                                       *args.sample_cfg)
        results = (results[:, [2, 1, 0]] + 1.) / 2.
        img = mmcv.imread(os.path.join(args.image_path, img_name))
        w = img.shape[1]
        ann = img[:, w // 2:, 0]
        # save images
        mmcv.mkdir_or_exist(os.path.join(args.save_path, 'images'))
        mmcv.mkdir_or_exist(os.path.join(args.save_path, 'annotations'))
        mmcv.mkdir_or_exist(
            os.path.dirname(os.path.join(args.save_path, 'images', img_name)))
        utils.save_image(
            results, os.path.join(args.save_path, 'images', f'fake_{i}.png'))
        mmcv.imwrite(
            ann,
            os.path.join(args.save_path, 'annotations',
                         f'fake_{i}_manual1.png'))
        print(f'{i} finished.')


if __name__ == '__main__':
    main()
