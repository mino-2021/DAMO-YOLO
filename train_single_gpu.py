#!/usr/bin/env python
# Single GPU training script for DAMO-YOLO
import argparse
import os
import torch
from loguru import logger

from damo.apis import Trainer
from damo.config.base import parse_config


def make_parser():
    parser = argparse.ArgumentParser('Damo-Yolo single GPU train parser')
    parser.add_argument(
        '-f',
        '--config_file',
        default=None,
        type=str,
        help='config file path',
    )
    parser.add_argument('--tea_config', type=str, default=None)
    parser.add_argument('--tea_ckpt', type=str, default=None)
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main():
    args = make_parser().parse_args()

    # Setup single GPU environment
    args.local_rank = 0
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    # Initialize distributed training with single GPU
    # Use 'gloo' backend for Windows (nccl not supported on Windows)
    torch.cuda.set_device(0)
    backend = 'gloo' if os.name == 'nt' else 'nccl'
    torch.distributed.init_process_group(backend=backend, init_method='env://')

    if args.tea_config is not None:
        tea_config = parse_config(args.tea_config)
    else:
        tea_config = None

    config = parse_config(args.config_file)
    config.merge(args.opts)

    trainer = Trainer(config, args, tea_config)
    trainer.train(args.local_rank)


if __name__ == '__main__':
    main()
