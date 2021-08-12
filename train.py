'''
Enabled for multiGPU debugging.
In general, the error is caused by memory shortage (in container).
e.g. "NCCL Error 2: unhandled system error"
https://stackoverflow.com/questions/61075390
https://www.jianshu.com/p/888c293de469
'''
# import os
# os.environ["NCCL_DEBUG"] = "INFO"

import torch
                 
from config import init_arguments, print_arguments
from handler import FinetuneHandler
from util import auto_redirect_std, check_output_dir
      
      
def train(args):
    check_output_dir(args.log_dir)
    auto_redirect_std(args.log_dir)
    print_arguments(args)
    
    handler = FinetuneHandler(args, mode='train')
    if args.eval_before_train:
        print('=' * 20 + f'\n\tEvaluation before training\n' + '=' * 20)
        handler.eval()
    for epoch in range(args.max_epoch):
        print('=' * 20 + f'\n\tStart Epoch {epoch}\n' + '=' * 20)
        
        '''
        Auto detect nan.
        Will increase the runtime (severely lowing running speed).
        Should only be enabled for debugging.
        '''
        # with torch.autograd.detect_anomaly():
        
        handler.train_epoch()
        handler.eval()


if __name__ == '__main__':
    args = init_arguments(mode='train')
    train(args) 