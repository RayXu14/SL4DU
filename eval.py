import os

import torch

from util import init_arguments, print_arguments, \
                 check_output_file
from handler import FinetuneHandler

mode = 'test'
      
      
def eval(args):
    print_arguments(args)
    handler = FinetuneHandler(args, mode)
    handler.eval()


if __name__ == '__main__':
    args = init_arguments(mode)
    assert args.load_path is not None
    eval(args) 