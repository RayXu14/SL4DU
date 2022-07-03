import json
import os
import random

import torch
from tqdm import tqdm
from transformers import pipeline, set_seed, AutoTokenizer

from data.preprocessors.basic_processor import BasicProcessor


class UbuntuProcessor(BasicProcessor):

    def read_raw(self, in_path):
        dialog_data = []
        with open(in_path, encoding='utf8') as f:
            for line in tqdm(f, desc=f'Reading [{in_path}] in UbuntuRS style'):
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

        
class UbuntuGen2RankProcessor(BasicProcessor):

    def __init__(self, args):
        super(UbuntuGen2RankProcessor, self).__init__(args)
        self.generator = pipeline('text-generation',
                                  model=self.args.gen_model,
                                  device=0)
        self.gen_tokenizer = AutoTokenizer.from_pretrained(self.args.gen_model)
        # 无法并行 https://github.com/huggingface/transformers/issues/3021
        
    def read_raw(self, in_path):
        dialog_data = []
        pre_g_context = ''
        with open(in_path, encoding='utf8') as f:
            for line in tqdm(f, desc=f'Reading [{in_path}] in UbuntuRS style'):
                line = line.strip()
                if len(line) == 0:
                    continue
                data = line.split("\t")
                
                # tokenize
                label = int(data[0].strip())
                dialog = [self.tokenizer.tokenize(turn.strip())
                            for turn in data[1:]]
                # gen
                g_context = ' '.join(data[1:-1])
                if pre_g_context != g_context:
                    pre_g_context = g_context
                    gen_context = ' '.join(g_context.split()[-self.args.gen_max_context_length:])
                    gen_context_len = len(self.gen_tokenizer(gen_context)['input_ids'])
                    set_seed(42)
                    with torch.no_grad():
                        hints = self.generator(gen_context,
                                      max_length = gen_context_len + self.args.gen_max_length,
                                      num_return_sequences=1) # TODO
                    hint = list(hints[0].values())[0]
                    hint = hint[len(gen_context):]
                    #print('\n\n' + gen_context + '\n\n' + hint + '\n\n')
                dialog_data.append({'label'   : label,
                                    'context' : dialog[:-1],
                                    'hint': hint,
                                    'response': dialog[-1]})
        return dialog_data


class DailyRSProcessor(BasicProcessor):

    def read_raw(self, in_path):
        dialog_data = []
        with open(in_path, encoding='utf8') as f:
            for line in tqdm(f, desc=f'Reading [{in_path}] in DailyRS style'):
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
        self.process_set(self.args.raw_train_file, self.args.pkl_train_file)
        self.neg_rate = 9
        self.process_set(self.args.raw_valid_file, self.args.pkl_valid_file)
        self.process_set(self.args.raw_test_file, self.args.pkl_test_file)


class PersonaChatRSProcessor(BasicProcessor):

    def process_set(self, set_name, neg_rate, pkl_file):        
        dialog_data = []
        data = self.data[set_name]
        for example in data:
            utterances = example['utterances'][-1]['history']
            context = [self.tokenizer.tokenize(turn.strip())
                            for turn in utterances
                            if len(turn.strip()) > 0]
            if len(context) == 0:
                continue
            response = example['utterances'][-1]['candidates'][-1]
            response = self.tokenizer.tokenize(response.strip())
            dialog_data.append({'context' : context,
                                'response': response})
        
        ranking_data = []
        for dialog in dialog_data:
            ranking_data.append({'label'   : 1,
                                 'context' : dialog['context'],
                                 'response': dialog['response']})
            for _ in range(neg_rate):
                neg_response = random.sample(dialog_data, 1)[0]['response']
                ranking_data.append({'label'   : 0,
                                     'context' : dialog['context'],
                                     'response': neg_response})
        
        print(f'Processed {len(ranking_data)} examples for {set_name} set.')
        
        self.write_pkl(ranking_data, pkl_file)
                
    def process_all(self):
        path = os.path.join(self.args.raw_data_path,
                            'personachat_self_original.json')
        with open(path, encoding='utf8') as f:
            self.data = json.load(f)
            
        print('Processing [{in_path}] in PersonaChatRS style.')
        self.process_set('train', 1, self.args.pkl_train_file)
        self.process_set('valid', 9, self.args.pkl_valid_file)
        