import os
import pickle

from data.utils import init_tokenizer
from util import check_output_dir, check_output_file


class BasicProcessor:

    def __init__(self, args):
        self.args = args
        self.tokenizer = init_tokenizer(args)
        check_output_dir(args.pkl_data_path, reserve_file=True)

    def read_raw(self, in_path):
        raise NotImplementedError('Not implemented data preprocessor.')
        
    def write_pkl(self, data, out_file):
        out_path = os.path.join(self.args.pkl_data_path, out_file)
        check_output_file(out_path)
        with open(out_path, 'wb') as f:
            pickle.dump(data, f)
        print(f'==> Saved data to {out_path}.')

    def process_set(self, in_file, out_file):
        in_path = os.path.join(self.args.raw_data_path, in_file)

        dialog_data = self.read_raw(in_path)
        print(f'Processed {len(dialog_data)} cases from {in_path}')
        
        self.write_pkl(dialog_data, out_file)

    def process_all(self):
        self.process_set(self.args.raw_train_file, self.args.pkl_train_file)
        self.process_set(self.args.raw_valid_file, self.args.pkl_valid_file)
        self.process_set(self.args.raw_test_file, self.args.pkl_test_file)