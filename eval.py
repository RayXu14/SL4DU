import os

import torch

from config import init_arguments, print_arguments
from handler import FinetuneHandler
from util import auto_redirect_std, check_output_dir
      
      
def eval(args):
    check_output_dir(args.log_dir, reserve_file=True)
    auto_redirect_std(args.log_dir, log_name='eval')
    print_arguments(args)
    
    handler = FinetuneHandler(args, mode='eval')
    handler.eval()


if __name__ == '__main__':
    args = init_arguments(mode='eval')
    eval(args) 