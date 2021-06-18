import os
import pickle
import random

import pyarrow as pa
import torch
from torch.utils.data import Dataset, Sampler

from data.utils import init_tokenizer


class CRMatchingDataset(Dataset):

    def __init__(self, args, filename, tokenizer, is_shuffle):
        super().__init__()
        self.args = args
        self.is_shuffle = is_shuffle

        data_path = os.path.join(args.data_path, filename)
        with open(data_path, 'rb') as f:
            self.dialog_data = pickle.load(f)
        print(f'Read {len(self.dialog_data)} examples from {data_path}.')
        
        ''' Whether to shuffle dialog data '''
        if is_shuffle:
            keys = [sum([len(t) for t in x['context']]) + len(x['response'])
                        for x in self.dialog_data]
            self.id_map =  [i[0] for i in sorted(enumerate(keys), key=lambda x: x[1])]
            print('And shuffled.')
        else:
            self.id_map =  list(range(len(self.dialog_data)))

        self.tokenizer = tokenizer
        self.mask_id, self.unk_id = tokenizer.convert_tokens_to_ids(['[MASK]','[UNK]'])
        
        '''
        pytorch的DataLoader多线程内存泄漏的简单的解决办法
        (https://github.com/pytorch/pytorch/issues/13246#issuecomment-823930907)
        '''
        self.dialog_data = pa.array(self.dialog_data)
        self.id_map = pa.array(self.id_map)
        
    def __len__(self):
        return len(self.dialog_data)
        
    def __getitem__(self, index):
        remapped_index = self.id_map[index].as_py()
        dialog_data = self.dialog_data[remapped_index].as_py()
        data_dict = {}
        
        ''' build main task (response selection) data'''
        data_dict["crm_token_ids"], \
        data_dict["crm_segment_ids"], \
        data_dict["crm_attention_mask"] \
            = self._build_cr_matching(dialog_data)
        data_dict["crm_label"] = dialog_data['label']
        
        return data_dict

    def _build_cr_matching(self, dialog):
        context = self._concat_utterances(dialog['context'])
        return self._auto_pack(context, dialog['response'])
        
    def _concat_utterances(self, utterances):
        context = []
        for utt in utterances:
            context.extend(utt)
            context.append('[EOT]')
        return context

    def _auto_pack(self, context, response):
        context, response = self._seq_len_trim(context, response)

        context = ["[CLS]"] + context + ["[SEP]"]
        segment_ids = [0] * len(context)
        attention_mask = [1] * len(context)
        #assert len(context) == len(segment_ids) == len(attention_mask)

        response = response + ["[SEP]"]
        segment_ids.extend([1] * len(response))
        attention_mask.extend([1] * len(response))
        dialog = context + response
        assert len(dialog) == len(segment_ids) == len(attention_mask)
        
        token_ids = self.tokenizer.convert_tokens_to_ids(dialog)
        return token_ids, segment_ids, attention_mask

    def _seq_len_trim(self, context, response):
        context = context[-(self.args.max_context_len - 2):]
        response = response[:self.args.max_response_len - 1]
        return context, response


class ChunkedRandomSampler(Sampler):
    ''' 
    Accelerate model training
    '''
    def __init__(self, data: Dataset, batch_size):
        super().__init__(data)
        self.data = data
        self.batch_size = batch_size

    def __iter__(self):
        indices = list(range(len(self.data)))
        chunks = [indices[i:i + self.batch_size] for i in
                     range(0, len(self.data), self.batch_size)]
        random.shuffle(chunks)
        new_indices = [e for piece in chunks for e in piece]
        return iter(new_indices)

    def __len__(self):
        return len(self.data)


def collate_fn(data_dict_batch):
    batch_dict = {}
    cr_matching_collate(data_dict_batch, batch_dict, 'crm')
    return batch_dict


def cr_matching_collate(data_dict_batch, batch_dict, task):
    tids_key = f'{task}_token_ids'
    sids_key = f'{task}_segment_ids'
    amask_key = f'{task}_attention_mask'
    label_key = f'{task}_label'

    ''' Ignore empty batch'''
    # TODO, is this needed?
    non_empty_batch = []
    for sample in data_dict_batch:
        if tids_key in sample:
            non_empty_batch.append(sample)
    if len(non_empty_batch) == 0:
        print('++++++++++\nEmpty case of task RS!')
        print(data_dict_batch)
        print('++++++++++')
        return

    ''' Padding '''
    max_len = max([len(e[tids_key]) for e in non_empty_batch])

    batch = dict()
    batch[tids_key]  = []
    batch[sids_key]  = []
    batch[amask_key] = []
    batch[label_key] = []

    for sample in non_empty_batch:
        cur_len = len(sample[tids_key])
        sample[tids_key].extend([0] * (max_len - cur_len))
        sample[sids_key].extend([0] * (max_len - cur_len))
        sample[amask_key].extend([0] * (max_len - cur_len))
        batch[tids_key].append(sample[tids_key])
        batch[sids_key].append(sample[sids_key])
        batch[amask_key].append(sample[amask_key])
        batch[label_key].append(sample[label_key])

    ''' Tensorize '''
    batch_dict[tids_key]  = torch.LongTensor(batch[tids_key])
    batch_dict[sids_key]  = torch.LongTensor(batch[sids_key])
    batch_dict[amask_key] = torch.LongTensor(batch[amask_key])
    batch_dict[label_key] = torch.FloatTensor(batch[label_key])