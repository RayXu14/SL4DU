from data import UbuntuProcessor, DailyProcessor
from util import init_arguments, print_arguments


def prepare_data(args):
    if args.task in ['Ubuntu', 'Douban', 'E-commerce']:
        processor = UbuntuProcessor(args)
    elif args.task in ['Daily']:
        processor = DailyProcessor(args)
    else:
        raise NotImplementedError('Not supported data preprocessing task.')
    processor.process_all()


if __name__ == '__main__':
    args = init_arguments(mode='preprocess')
    print_arguments(args)
    prepare_data(args)