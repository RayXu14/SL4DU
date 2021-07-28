import os

import torch

from util import init_arguments, print_arguments, \
                 check_output_file
from handler import FinetuneHandler

mode = 'test'
      
      
def infer(args):
    if not os.path.exists('infer'):
        os.makedirs('infer')
    check_output_file(os.path.join('infer', args.log_name))
    print_arguments(args)
    handler = FinetuneHandler(args, mode)

    handler.infer()


if __name__ == '__main__':
    args = init_arguments(mode)
    assert args.load_path is not None
    infer(args) 