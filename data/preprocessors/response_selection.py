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
        
    def read_raw(self, in_path):
        with open(in_path, encoding='utf8') as f:
            n_example = 0
            
            contexts = []
            context_ids = []
            context_lens = []
            
            pre_context = ''
            for line in tqdm(f, desc=f'Reading [{in_path}] in UbuntuG2R style'):
                line = line.strip()
                if len(line) == 0:
                    continue
                context = ' '.join(line.split("\t")[1:-1])
                context = ' '.join(context.split()[-self.args.gen_max_context_length:])
                
                print(context + '\n')
                
                if pre_context == context:
                    assert len(context_ids) > 0
                    context_ids[-1] = context_ids[-1] + [n_example]
                else:
                    pre_context = context
                    contexts.append(context)
                    context_ids.append([n_example])
                    context_len = len(self.gen_tokenizer(context)['input_ids'])
                    context_lens.append(context_len)
                n_example += 1
                if n_example == 10: # TODO
                    break # TODO
        
        assert len(contexts) == len(context_ids) == len(context_lens), f'{len(contexts)} {len(context_ids)} {len(context_lens)}'
        breakpoint()
            
        hints = []
        for i in tqdm(range(0, len(contexts), self.args.gen_batch_size),
                      desc=f'G2R preprocessing...'):
            right_edge = min(i + self.args.gen_batch_size, len(contexts))
            context_batch = contexts[i:right_edge]
            context_id_batch = contexts[i:right_edge]
            context_len_batch = max(context_lens[i:right_edge])
            max_context_len = max(context_len_batch)
            
            set_seed(42)
            with torch.no_grad():
                hint_batch = self.generator(context_batch,
                                       max_length = max_context_len + self.args.gen_max_length,
                                       num_return_sequences=1) # TODO
            for hint, context_id, context_len in zip(hint_batch, context_id_batch, context_len_batch):
                hint = list(hint.values())[0]
                print(hint + '\n')
                hint = hint[context_len:]
                print(hint + '\n')
                hints.extend([hint] * len(context_id))
        breakpoint()
                
        return hints


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
        