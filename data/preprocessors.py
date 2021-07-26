import os
import pickle
import random

from tqdm import tqdm

from data.utils import init_tokenizer
from util import check_output_file


class BasicProcessor:

    def __init__(self, args):
        self.args = args
        self.tokenizer = init_tokenizer(args)

    def read_file(self, in_path):
        raise NotImplementedError('Not implemented data preprocessor.')

    def process_file(self, in_file, out_file):
        in_path = os.path.join(self.args.data_path, in_file)
        out_path = os.path.join(self.args.data_path, out_file)
        check_output_file(out_path)

        dialog_data = self.read_file(in_path)

        print(f'Processed {len(dialog_data)} cases from {in_path}')
        with open(out_path, 'wb') as f:
            pickle.dump(dialog_data, f)
        print(f'==> Then saved them to {out_path}.')

    def process_all(self):
        self.process_file(self.args.raw_train_file, self.args.pkl_train_file)
        self.process_file(self.args.raw_valid_file, self.args.pkl_valid_file)
        self.process_file(self.args.raw_test_file, self.args.pkl_test_file)


class UbuntuProcessor(BasicProcessor):

    def read_file(self, in_path):
        dialog_data = []
        with open(in_path, encoding='utf8') as f:
            for line in tqdm(f, desc=f'Reading [{in_path}] in Ubuntu style'):
                line = line.strip()
                if len(line) == 0:
                    continue
                data = line.split("\t")
                label = int(data[0].strip())
                dialog = [self.tokenizer.tokenize(turn.strip())
                            for turn in data[1:]]
                dialog_data.append({'label'   : label,
                                    'context' : dialog[:-1],
                                    'response': dialog[-1]})
        return dialog_data


class DailyProcessor(BasicProcessor):

    def read_file(self, in_path):
        dialog_data = []
        with open(in_path, encoding='utf8') as f:
            for line in tqdm(f, desc=f'Reading [{in_path}] in Daily style'):
                line = line.strip()
                if len(line) == 0:
                    continue
                data = line.split('__eou__')
                dialog = [self.tokenizer.tokenize(turn.strip())
                            for turn in data[1:] if len(turn.strip()) > 0]
                if len(dialog) <= 1:
                    continue
                dialog_data.append({'context' : dialog[:-1],
                                    'response': dialog[-1]})
        
        ranking_data = []
        for dialog in dialog_data:
            ranking_data.append({'label'   : 1,
                                'context' : dialog['context'],
                                'response': dialog['response']})
            for _ in range(self.neg_rate):
                neg_response = random.sample(dialog_data, 1)[0]['response']
                ranking_data.append({'label'   : 0,
                                     'context' : dialog['context'],
                                     'response': neg_response})
        return ranking_data

    def process_all(self):
        self.neg_rate = 1
        self.process_file(self.args.raw_train_file, self.args.pkl_train_file)
        self.neg_rate = 9
        self.process_file(self.args.raw_valid_file, self.args.pkl_valid_file)
        self.process_file(self.args.raw_test_file, self.args.pkl_test_file)