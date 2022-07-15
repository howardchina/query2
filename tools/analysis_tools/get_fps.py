import argparse

import torch
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector

from mmdet.models import build_detector

import time
import mmcv
import torch
import glob

def get_model_fps(model, input_shape):
    img_list = glob.glob('/mnt/home1/workspace2/QueryInst/data/usd514_jpeg_roi/images/*.jpg')
    start = time.perf_counter()
    n = min(len(img_list), 100)
    for i in range(n):
        template_file = img_list[i]
        img = mmcv.imread(template_file)
        resized_img = mmcv.imresize(img, (input_shape[1], input_shape[2]))
        tensor = torch.from_numpy(resized_img.transpose(2, 0, 1))
        if torch.cuda.is_available():
            input = tensor[None, :].float().cuda()
        res = model(input)
        # res = inference_detector(model, resized_img)
    end = time.perf_counter()
    return n / (end - start)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1280, 800],
        help='input image size')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.cfg = cfg
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FPS counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    fps = get_model_fps(model, input_shape)
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'FPS: {fps}\n')


if __name__ == '__main__':
    main()
