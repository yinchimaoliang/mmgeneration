import argparse
import os
import sys

import mmcv
import numpy as np
from mmcv import DictAction
from torchvision import utils

# yapf: disable
sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))  # isort:skip  # noqa

from mmgen.apis import init_model, sample_img2img_model  # isort:skip  # noqa
# yapf: enable

PALETTE = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]]
CLASSES = ['benign', 'gleason grade 3', 'gleason grade 4', 'gleason grade 5']


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
    parser.add_argument(
        '--apply-ann',
        action='store_true',
        help='whether to apply ann on the img')

    args = parser.parse_args()
    return args


def show_result(img,
                result,
                palette=None,
                win_name='',
                show=False,
                wait_time=0,
                out_file=None,
                opacity=0.5):
    """Draw `result` over `img`.

    Args:
        img (str or Tensor): The image to be displayed.
        result (Tensor): The semantic segmentation results to draw over
            `img`.
        palette (list[list[int]]] | np.ndarray | None): The palette of
            segmentation map. If None is given, random palette will be
            generated. Default: None
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
            Default: 0.
        show (bool): Whether to show the image.
            Default: False.
        out_file (str or None): The filename to write the image.
            Default: None.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
    Returns:
        img (Tensor): Only if not `show` or `out_file`
    """
    img = mmcv.imread(img)
    img = img.copy()
    seg = result
    if palette is None:
        if PALETTE is None:
            palette = np.random.randint(0, 255, size=(len(CLASSES), 3))
        else:
            palette = PALETTE
    palette = np.array(palette)
    assert palette.shape[0] == len(CLASSES)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    assert 0 < opacity <= 1.0
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    # convert to BGR
    color_seg = color_seg[..., ::-1]

    img = img * (1 - opacity) + color_seg * opacity
    img = img.astype(np.uint8)
    # if out_file specified, do not show image in window
    return img


def main():
    args = parse_args()
    model = init_model(
        args.config, checkpoint=args.checkpoint, device=args.device)

    if args.sample_cfg is None:
        args.sample_cfg = dict()

    img_names = os.listdir(args.image_path)
    # length = len(img_names)
    # aug_names = ['slide003_core092.png', 'slide003_core045.png',
    # 'slide005_core051.png', 'slide002_core073.png', 'slide006_core142.png',
    # 'slide002_core144.png', 'slide007_core005.png', 'slide002_core126.png',
    # 'slide005_core091.png', 'slide007_core006.png', 'slide005_core073.png',
    # 'slide007_core016.png', 'slide001_core145.png']
    for img_name in img_names:
        if 'flip' in img_name:
            continue
        img_whole = mmcv.imread(os.path.join(args.image_path, img_name))
        w = img_whole.shape[1]
        ann = img_whole[:, w // 2:, :]
        img = img_whole[:, :w // 2, :]
        ann_hflip = mmcv.imflip(ann, 'horizontal')
        ann_vflip = mmcv.imflip(ann, 'vertical')
        ann_vhflip = mmcv.imflip(ann_vflip, 'horizontal')
        img_hflip = mmcv.imflip(img, 'horizontal')
        img_vflip = mmcv.imflip(img, 'vertical')
        img_vhflip = mmcv.imflip(img_vflip, 'horizontal')
        img_whole[:, w // 2:, :] = ann_hflip
        img_whole[:, :w // 2, :] = img_hflip
        mmcv.imwrite(
            img_whole,
            os.path.join(args.image_path, img_name.replace('.', '_hflip.')))
        img_whole[:, w // 2:, :] = ann_vflip
        img_whole[:, :w // 2, :] = img_vflip
        mmcv.imwrite(
            img_whole,
            os.path.join(args.image_path, img_name.replace('.', '_vflip.')))
        img_whole[:, w // 2:, :] = ann_vhflip
        img_whole[:, :w // 2, :] = img_vhflip
        mmcv.imwrite(
            img_whole,
            os.path.join(args.image_path, img_name.replace('.', '_vhflip.')))
    #     aug_names.append(img_name.replace('.', '_vflip.'))
    #     aug_names.append(img_name.replace('.', '_hflip.'))
    #     aug_names.append(img_name.replace('.', '_vhflip.'))
    # print(aug_names)
    # img_names = img_names + aug_names
    img_names = os.listdir(args.image_path)
    for i, img_name in enumerate(img_names):
        img = mmcv.imread(os.path.join(args.image_path, img_name))
        w = img.shape[1]
        ann = img[:, w // 2:, 0]
        img = img[:, :w // 2, :]

        results = sample_img2img_model(model,
                                       os.path.join(args.image_path, img_name),
                                       *args.sample_cfg)
        results = (results[:, [2, 1, 0]] + 1.) / 2.
        # save images
        mmcv.mkdir_or_exist(os.path.join(args.save_path, 'images'))
        mmcv.mkdir_or_exist(os.path.join(args.save_path, 'annotations'))
        mmcv.mkdir_or_exist(
            os.path.dirname(os.path.join(args.save_path, 'images', img_name)))
        utils.save_image(
            results, os.path.join(args.save_path, 'images',
                                  f'fake_{img_name}'))
        fake_img = mmcv.imread(
            os.path.join(args.save_path, 'images', f'fake_{img_name}'))
        ann = mmcv.imresize(ann, (fake_img.shape[1], fake_img.shape[0]))
        img = mmcv.imresize(img, (fake_img.shape[1], fake_img.shape[0]))
        mmcv.imwrite(
            ann, os.path.join(args.save_path, 'annotations',
                              f'fake_{img_name}'))
        # mmcv.imwrite(
        #     img,
        #     os.path.join(args.save_path, 'images',
        #                  f'{img_name}'))

        if args.apply_ann:
            # img_vis = show_result(img, ann)
            fake_img_vis = show_result(fake_img, ann)
            # mmcv.imwrite(
            #     img_vis,
            #     os.path.join(args.save_path, 'images',
            #                 f'vis_{img_name}'))
            mmcv.imwrite(
                fake_img_vis,
                os.path.join(args.save_path, 'images', f'fake_vis_{img_name}'))
        print(f'{img_name} finished.')


if __name__ == '__main__':
    main()
