from argparse import ArgumentParser
import os


def add_task_args(parser):
    parser.add_argument('--task', type=str, required=True,
        choices=['Ubuntu', 'Douban', 'E-commerce',
                 'Daily', 'PersonaChat',
                 'GRADE-Daily', 'USR-PersonaChat', 'FED'],
        help='Decide the method of raw file reading, ')
    parser.add_argument('--data_path', type=str,
        help='The directory where task data is saved.')


def add_raw_data_args(parser):
    parser.add_argument('--raw_train_file', type=str,
        default='train.txt',
        help='The path of the raw file of training set.')
    parser.add_argument('--raw_valid_file', type=str,
        default='valid.txt',
        help='The path of the raw file of validation set.')
    parser.add_argument('--raw_test_file', type=str,
        default='test.txt',
        help='The path of the raw file of test set.')


def add_pkl_data_args(parser):
    parser.add_argument('--pkl_train_file', type=str,
        default='train.pkl',
        help='The path of the preprocessed file of training set.')
    parser.add_argument('--pkl_valid_file', type=str,
        default='valid.pkl',
        help='The path of the preprocessed file of validation set.')
    parser.add_argument('--pkl_test_file', type=str,
        default='test.pkl',
        help='The path of the preprocessed file of test set.')


def add_pretrained_args(parser):
    parser.add_argument('--pretrained_path', type=str,
        default='../../pretrained')
    parser.add_argument('--pretrained_model', type=str,
        help='Name of the pretrained model. (https://huggingface.co/models)')
    parser.add_argument('--add_EOT', action='store_true',
        help='Whether to add EOT token to the pretrained tokenizer and model.')


def add_model_args(parser):
    parser.add_argument('--max_context_len', type=int,
        default=448,
        help='Max number of context tokens.')
    parser.add_argument('--max_response_len', type=int,
        default=64,
        help='Max number of response tokens.')

    parser.add_argument('--dropout_rate', type=float,
        default=0.2)
    parser.add_argument('--freeze_layers', type=int,
        help='Number of freezed model (lower) layers.')
        
    parser.add_argument('--cpu_workers', type=int,
        default=4,
        help='Number of processes that loads data.')
    parser.add_argument('--eval_batch_size', type=int,
        default=100)
    parser.add_argument('--eval_view_every', type=int,
        default=500,
        help='View evaluation process every this many steps.')
        
    parser.add_argument('--load_path', type=str,
        help='Path to load checkpoint if it is needed.')
    parser.add_argument('--not_save_record', action='store_true',
        help='Whether to save evaluation result.')
    parser.add_argument('--eval_before_train', action='store_true',
        help='Whether to evaluate model before training process.')


def add_train_args(parser):
    parser.add_argument('--max_epoch', type=int,
        default=100)
    parser.add_argument('--virtual_batch_size', type=int,
        default=32,
        help='Virtual train batch size, must be integer multiple of train_batch_size.')
    parser.add_argument('--train_batch_size', type=int,
        default=4)
    parser.add_argument('--train_view_every', type=int,
        default=100,
        help='View training process every this many steps.')
    parser.add_argument('--learning_rate', type=float,
        default=3e-5)
    parser.add_argument('--max_gradient_norm', type=float,
        default=5.)
    parser.add_argument('--log_dir', type=str,
        help='Directory to save log and checkpoints.')
    parser.add_argument('--save_ckpt', action='store_true',
        help='Whether to save checkpoints.')
    parser.add_argument('--ckpt_name', type=str,
        default='best')

    ''' Auxiliary Tasks '''
    parser.add_argument('--use_NSP', action='store_true',
        help='Whether to use Next Session Prediction task.')
    parser.add_argument('--use_UR', action='store_true',
        help='Whether to use Utterance Restoration task.')
    parser.add_argument('--use_ID', action='store_true',
        help='Whether to use Incoherence Detection task.')
    parser.add_argument('--use_CD', action='store_true',
        help='Whether to use Consistency Discrimination task.')
    parser.add_argument('--margin', type=float,
        default=0.6,
        help='Margin of MarginRankingLoss.')


def add_infer_args(parser):
    parser.add_argument('--log_name', type=str,
        help='File name to save log.')
    parser.add_argument('--assess', action='store_true',
        help='Whether to assess the correlation between inf & ref.')


def init_arguments(mode):
    parser = ArgumentParser()

    if mode == 'preprocess':
        print('Initializing preprocess arguments...')
        add_task_args(parser)
        add_raw_data_args(parser)
        add_pkl_data_args(parser)
        add_pretrained_args(parser)
    elif mode == 'train':
        print('Initializing train arguments...')
        add_task_args(parser)
        add_pkl_data_args(parser)
        add_pretrained_args(parser)
        add_model_args(parser)
        add_train_args(parser)
    elif mode == 'test':
        print('Initializing test arguments...')
        add_task_args(parser)
        add_pkl_data_args(parser)
        add_pretrained_args(parser)
        add_model_args(parser)
        add_infer_args(parser)
    else:
        raise NotImplementedError('Not supported argument mode.')

    args = parser.parse_args()
    
    ''' Check arguments '''
    if hasattr(args, 'train_view_every'):
        assert args.virtual_batch_size % args.train_batch_size == 0, \
            'Virtual train batch size, must be integer multiple of train_batch_size.'
        assert args.train_view_every % (args.virtual_batch_size // args.train_batch_size) == 0
    
    if hasattr(args, 'log_dir'):
        args.log_dir = os.path.join('result', args.log_dir)
    
    return args


def print_arguments(args):
    for arg in vars(args):
        print(f'{arg}\n\t\t\t\t= {getattr(args, arg)}')
