import torch

from util import init_arguments, print_arguments, \
                 check_output_dir, auto_redirect_std
from handler import FinetuneHandler

mode = 'train'
      
      
def train(args):
    check_output_dir(args.log_dir)
    auto_redirect_std(args.log_dir)
    print_arguments(args)
    handler = FinetuneHandler(args, mode)
    if args.eval_before_train:
        print('=' * 20 + f'\n\tEvaluation before training\n' + '=' * 20)
        handler.eval()
    for epoch in range(args.max_epoch):
        print('=' * 20 + f'\n\tStart Epoch {epoch}\n' + '=' * 20)
        
        # Auto detect nan
        # will increase the runtime
        # Should only be enabled for debugging
        # with torch.autograd.detect_anomaly():
        
        handler.train_epoch()
        handler.eval()


if __name__ == '__main__':
    args = init_arguments(mode)
    train(args) 