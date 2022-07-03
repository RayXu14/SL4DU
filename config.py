from argparse import ArgumentParser
import os


def _add_task_args(parser):
    parser.add_argument('--task', type=str, required=True,
        choices=['RS', 'G2R', 'CLS', 'DST', 'RG'],
        help='Decide downstream task: Response Selection / Classification')
    parser.add_argument('--label_name', type=str,
                        choices=['act', 'emotion'],
                        help='The label name of DAS or ERC task.')
    parser.add_argument('--n_class', type=int,
                        help='The number of classes for DAS or ERC task.')
    parser.add_argument('--dataset', type=str, required=True,
        choices=['Ubuntu', 'Douban', 'E-commerce',
                 'Daily', 'SwDA', 'PersonaChat',
                 'GRADEdata', 'USR-PersonaChat', 'FED',
                 'DSTC2'],
        help='Decide the method of raw file reading, ')


def _add_raw_data_args(parser):
    parser.add_argument('--raw_data_path', type=str, required=True,
        help='The directory where raw data is saved.')
    parser.add_argument('--raw_train_file', type=str,
        default='train.txt',
        help='The path of the raw file or directory of training set.')
    parser.add_argument('--raw_valid_file', type=str,
        default='valid.txt',
        help='The path of the raw file or directory of validation set.')
    parser.add_argument('--raw_test_file', type=str,
        default='test.txt',
        help='The path of the raw file or directory of test set.')\
        
        
def _add_genRank_args(parser):
    parser.add_argument('--gen_model', type=str,
        default='gpt2') # gpt2, gpt2-medium, gpt2-large, gpt2-xl
    parser.add_argument('--gen_max_context_length', type=int,
        default=448)
    parser.add_argument('--gen_max_length', type=int,
        default=64)


def _add_pkl_data_args(parser):
    parser.add_argument('--pkl_data_path', type=str, required=True,
        help='The directory where pkl data is saved.')
    parser.add_argument('--pkl_train_file', type=str,
        default='train.pkl',
        help='The path of the preprocessed file of training set.')
    parser.add_argument('--pkl_valid_file', type=str,
        default='valid.pkl',
        help='The path of the preprocessed file of validation set.')
    parser.add_argument('--pkl_test_file', type=str,
        default='test.pkl',
        help='The path of the preprocessed file of test set.')


def _add_pretrainedLM_args(parser):
    parser.add_argument('--pretrained_path', type=str,
        default='../../pretrained')
    parser.add_argument('--pretrained_model', type=str, required=True,
        help='Name of the pretrained model. (https://huggingface.co/models)')
    parser.add_argument('--add_EOT', action='store_true',
        help='Whether to add EOT token to the pretrained tokenizer & model.')


def _add_data_loading_args(parser):
    parser.add_argument('--max_context_len', type=int,
        default=448,
        help='Max number of context tokens.')
    parser.add_argument('--max_response_len', type=int,
        default=64,
        help='Max number of response tokens.')
    parser.add_argument('--cpu_workers', type=int,
        default=4,
        help='Number of processes that loads data.')


def _add_model_args(parser):
    parser.add_argument('--dropout_rate', type=float,
        default=0.2)
    parser.add_argument('--freeze_layers', type=int, required=True,
        help='Number of freezed model (lower) layers.')
    parser.add_argument('--load_path', type=str,
        help='Path to load checkpoint if it is needed.')
        
        
def _add_eval_args(parser):
    parser.add_argument('--eval_batch_size', type=int,
        default=100)
    parser.add_argument('--eval_view_every', type=int,
        default=500,
        help='View evaluation process every this many steps.')
    parser.add_argument('--not_save_record', action='store_true',
        help='Whether to save evaluation result.')
    parser.add_argument('--log_dir', type=str, required=True,
        help='Directory to save log and checkpoints.')


def _add_train_args(parser):
    parser.add_argument('--max_epoch', type=int,
        default=100)
    parser.add_argument('--virtual_batch_size', type=int,
        default=32,
        help='Logical train batch size.')
    parser.add_argument('--train_batch_size', type=int,
        default=4,
        help='GPU train batch size.')
    parser.add_argument('--train_view_every', type=int,
        default=100,
        help='View training process every this many steps.')
    parser.add_argument('--learning_rate', type=float,
        default=3e-5)
    parser.add_argument('--max_gradient_norm', type=float,
        default=5.)
    parser.add_argument('--save_ckpt', action='store_true',
        help='Whether to save checkpoints.')
    parser.add_argument('--ckpt_name', type=str,
        default='best')
    parser.add_argument('--eval_before_train', action='store_true',
        help='Whether to evaluate model before training process.')


def _add_SL_args(parser):
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


def init_arguments(mode):
    parser = ArgumentParser()

    _add_task_args(parser)
    _add_pretrainedLM_args(parser)
    _add_pkl_data_args(parser)
    
    if mode == 'preprocess':
        print('Initializing preprocess arguments...')
        _add_raw_data_args(parser)
        _add_genRank_args(parser)
    elif mode == 'train':
        print('Initializing train arguments...')
        _add_data_loading_args(parser)
        _add_model_args(parser)
        _add_eval_args(parser)
        _add_train_args(parser)
        _add_SL_args(parser)
    elif mode == 'eval':
        print('Initializing evaluation arguments...')
        _add_data_loading_args(parser)
        _add_model_args(parser)
        _add_eval_args(parser)
    else:
        raise NotImplementedError('Not supported argument mode.')

    args = parser.parse_args()
    
    ''' Check arguments '''
    if hasattr(args, 'train_view_every'):
        assert args.virtual_batch_size % args.train_batch_size == 0, \
            'Virtual train batch size, ' + \
            'must be integer multiple of train_batch_size.'
        assert args.train_view_every % \
            (args.virtual_batch_size // args.train_batch_size) == 0, \
            'Ensure fairish visualization of the training process.'
    if mode in ['train', 'eval'] and args.task in ['CLS']:
        assert hasattr(args, 'n_class') and args.n_class is not None, \
            f'Should set n_class for {args.task} task.'
        assert hasattr(args, 'label_name') and args.label_name is not None, \
            f'Should set label_name for {args.task} task.'
    if mode == 'eval':
        assert args.load_path is not None
    
    if hasattr(args, 'log_dir'):
        args.log_dir = os.path.join('result', args.log_dir)
    
    return args


def print_arguments(args):
    for arg in vars(args):
        print(f'{arg}\n\t\t\t\t= {getattr(args, arg)}')
