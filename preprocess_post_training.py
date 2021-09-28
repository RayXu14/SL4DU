from data import get_processor
from config import init_arguments, print_arguments


def prepare_data(args):
    print_arguments(args)
    
    processor = get_processor(args)
    processor.make_post_training_corpus()


if __name__ == '__main__':
    args = init_arguments(mode='preprocess')
    prepare_data(args)