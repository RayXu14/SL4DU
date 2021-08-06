import json
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


class GRADEDailyProcessor(BasicProcessor):

    def process_ctx(self, line):
        data = line.split('|||')
        context = [self.tokenizer.tokenize(turn.strip())
                    for turn in data]
        return context

    def process_hyp(self, line):
        return self.tokenizer.tokenize(line)

    def process_score(self, line):
        return float(line)

    def read_file(self, in_path):
        contents = []
        with open(in_path, encoding='utf8') as f:
            for line in tqdm(f, desc=f'Reading [{in_path}]'):
                line = line.strip()
                if len(line) == 0:
                    continue
                contents.append(self.process_line(line))
        return contents
        
    def merge_data(self, contexts, candidates, scores):
        for ctx, cand, score in zip(contexts, candidates, scores):
            self.eval_data.append({'label'   : score,
                                   'context' : ctx,
                                   'response': cand})
                
    def process_all(self):
        self.process_line = self.process_ctx
        ctx_path = os.path.join(self.args.data_path, 
                                'ctx.txt')
        contexts = self.read_file(ctx_path)
        
        self.process_line = self.process_hyp
        generator_hyp_path = os.path.join(self.args.data_path, 
                                          'generator-hyp.txt')
        generator_candidates = self.read_file(generator_hyp_path)
        ranker_hyp_path = os.path.join(self.args.data_path, 
                                       'ranker-hyp.txt')
        ranker_candidates = self.read_file(ranker_hyp_path)
        
        self.process_line = self.process_score
        generator_score_path = os.path.join(self.args.data_path,
                                            'generator-score.txt')
        generator_scores = self.read_file(generator_score_path)
        ranker_score_path = os.path.join(self.args.data_path,
                                         'ranker-score.txt')
        ranker_scores = self.read_file(ranker_score_path)
            
        self.eval_data = []
        self.merge_data(contexts, generator_candidates, generator_scores)
        self.merge_data(contexts, ranker_candidates, ranker_scores)

        print(f'Processed {len(self.eval_data)} examples for GRADE-Daily.')
        out_path = os.path.join(self.args.data_path, 
                                self.args.pkl_test_file)
        with open(out_path, 'wb') as f:
            pickle.dump(self.eval_data, f)
        print(f'==> Then saved them to {out_path}.')


class USRPersonaChatProcessor(BasicProcessor):
                
    def process_all(self):
        with open(os.path.join(self.args.data_path, 
                               self.args.raw_test_file),
                  encoding='utf8') as f:
            data = json.load(f)
        ranking_data = []
        for example in tqdm(data):
            utterances = example['context'].strip()
            utterances = utterances.split('\n')
            context = [self.tokenizer.tokenize(turn.strip())
                        for turn in utterances if len(turn.strip()) > 0]
            if len(context) == 0:
                continue
                
            for response_example in example['responses']:
                if response_example['model'] == 'Original Ground Truth':
                    continue
                response = response_example['response'].strip()
                response = self.tokenizer.tokenize(response)
                scores = response_example['Overall']
                score = sum(scores) / len(scores)
                ranking_data.append({'label'   : score,
                                     'context' : context,
                                     'response': response})
            
        print(f'Processed {len(ranking_data)} examples for USR-PersonaChat.')
        out_path = os.path.join(self.args.data_path, 
                                self.args.pkl_test_file)
        with open(out_path, 'wb') as f:
            pickle.dump(ranking_data, f)
        print(f'==> Then saved them to {out_path}.')
        

class FEDProcessor(BasicProcessor):
                
    def process_all(self):
        with open(os.path.join(self.args.data_path, 
                               self.args.raw_test_file),
                  encoding='utf8') as f:
            data = json.load(f)
        ranking_data = []
        for example in tqdm(data):
            utterances = example['context'].strip()
            utterances = utterances.split('\n')
            context = [self.tokenizer.tokenize(turn.strip())
                        for turn in utterances if len(turn.strip()) > 0]
            if len(context) == 0 or 'response' not in example:
                continue
                
            response = example['response'].strip()
            response = self.tokenizer.tokenize(response)
            scores = example['annotations']['Overall']
            score = sum(scores) / len(scores)
            ranking_data.append({'label'   : score,
                                 'context' : context,
                                 'response': response})
            
        print(f'Processed {len(ranking_data)} examples for FED.')
        out_path = os.path.join(self.args.data_path, 
                                self.args.pkl_test_file)
        with open(out_path, 'wb') as f:
            pickle.dump(ranking_data, f)
        print(f'==> Then saved them to {out_path}.')



class PersonaChatProcessor(BasicProcessor):

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
        out_path = os.path.join(self.args.data_path, pkl_file)
        with open(out_path, 'wb') as f:
            pickle.dump(ranking_data, f)
        print(f'==> Then saved them to {out_path}.')
        return ranking_data
                
    def process_all(self):
        with open(os.path.join(self.args.data_path, 
                               'personachat_self_original.json'),
                  encoding='utf8') as f:
            self.data = json.load(f)
            
        self.process_set('train', 1, self.args.pkl_train_file)
        self.process_set('valid', 9, self.args.pkl_valid_file)