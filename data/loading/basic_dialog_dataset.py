from copy import deepcopy
import os
import pickle
import random

import pyarrow as pa
from torch.utils.data import Dataset

from data.utils import init_tokenizer
from util import fetch_pyarrow


class BasicDialogDataset(Dataset):

    def __init__(self, args, filename, tokenizer, is_train):
        super().__init__()
        self.args = args
        self.is_train = is_train

        data_path = os.path.join(args.pkl_data_path, filename)
        with open(data_path, 'rb') as f:
            samples = pickle.load(f)
        print(f'Read {len(samples)} examples from {data_path}.')
        
        samples = self._process_samples(samples)

        self.tokenizer = tokenizer
        self.mask_id, self.unk_id = \
            tokenizer.convert_tokens_to_ids(['[MASK]','[UNK]'])
        
        if is_train:
            keys = [sum([len(t) for t in x['context']]) + len(x['response'])
                    for x in samples]
            id_map =  [i[0] for i in sorted(enumerate(keys), key=lambda x: x[1])]
            print('And shuffled.')
        else:
            id_map = list(range(len(samples)))
            print('Not shuffled.')
            
        '''
        Simple solution for memory leak of pytorch's DataLoader
        (https://github.com/pytorch/pytorch/issues/13246#issuecomment-823930907)
        '''
        self.samples = pa.array(samples)
        self.id_map = pa.array(id_map)

    def _process_samples(self, samples):
        return samples
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, index):
        remapped_index = fetch_pyarrow(self.id_map, index)
        data_dict = {}
        
        self._get_main_task_item(data_dict, remapped_index)
        
        ''' build auxiliary task data'''
        if self.is_train:
            if hasattr(self.args, 'use_NSP') and self.args.use_NSP:
                nsp_sample = self._build_NSP(remapped_index)
                if nsp_sample is not None:
                    data_dict['nsp_label'], \
                    (data_dict['nsp_token_ids'], 
                     data_dict['nsp_segment_ids'], 
                     data_dict['nsp_attention_mask']) \
                        = nsp_sample
                    
            if hasattr(self.args, 'use_UR') and self.args.use_UR:
                ur_sample = self._build_UR(remapped_index,
                                           data_dict['main_token_ids'])
                if ur_sample is not None:
                    data_dict['ur_positions'], \
                    data_dict['ur_labels'], \
                    data_dict['ur_token_ids'] \
                        = ur_sample
                    assert len(data_dict['ur_token_ids']) \
                        == len(data_dict['main_token_ids'])
                    data_dict['ur_segment_ids'] \
                        = deepcopy(data_dict['main_segment_ids'])
                    data_dict['ur_attention_mask'] \
                        = deepcopy(data_dict['main_attention_mask'])
                    assert len(data_dict['ur_token_ids']) \
                        == len(data_dict['ur_segment_ids']) \
                        == len(data_dict['ur_attention_mask'])
                            
            if hasattr(self.args, 'use_ID') and self.args.use_ID:
                id_sample = self._build_ID(remapped_index)
                if id_sample is not None:
                    data_dict['id_label'], \
                    data_dict['id_token_ids'], \
                    data_dict['id_segment_ids'], \
                    data_dict['id_attention_mask'], \
                    data_dict['id_locations'] \
                        = id_sample
                    
            if hasattr(self.args, 'use_CD') and self.args.use_CD:
                cd_sample = self._build_CD(remapped_index)
                if cd_sample is not None:
                    (data_dict['cd_pos_token_ids'], 
                     data_dict['cd_pos_segment_ids'], 
                     data_dict['cd_pos_attention_mask']), \
                    (data_dict['cd_neg_token_ids'], 
                     data_dict['cd_neg_segment_ids'], 
                     data_dict['cd_neg_attention_mask']) \
                        = cd_sample
        
        return data_dict
        
    def _get_main_task_item(self, data_dict, remapped_index):
        sample = self._get_sample(remapped_index)
        context = self._concat_utterances(sample['context'])
        data_dict['main_label'] = sample['label']
        data_dict['main_token_ids'], \
        data_dict['main_segment_ids'], \
        data_dict['main_attention_mask'] \
            = self._auto_pack(context, sample['response'])
        
    def _get_sample(self, remapped_index = None):
        if remapped_index is None:
            if hasattr(self, 'proper_ids'):
                remapped_index = random.sample(self.proper_ids, 1)[0]
            else:
                remapped_index = random.randint(0, len(self) - 1)
        sample = fetch_pyarrow(self.samples, remapped_index)
        return deepcopy(sample)
        
    def _concat_utterances(self, utterances):
        context = []
        for utt in utterances:
            context.extend(utt)
            context.append('[EOT]')
        return context

    def _auto_pack(self, context, response=None):
        if response is None:
            context = self._seq_len_trim(context)
        else:
            context, response = self._seq_len_trim(context, response)

        dialog = ["[CLS]"] + context + ["[SEP]"]
        segment_ids = [0] * len(dialog)
        attention_mask = [1] * len(dialog)

        if response is not None:
            response = response + ["[SEP]"]
            segment_ids.extend([1] * len(response))
            attention_mask.extend([1] * len(response))
            
            dialog = dialog + response
            
        assert len(dialog) == len(segment_ids) == len(attention_mask)
        
        token_ids = self.tokenizer.convert_tokens_to_ids(dialog)
        return token_ids, segment_ids, attention_mask

    def _seq_len_trim(self, context, response=None):
        if response is None:
            assert False, 'Please make sure not all context are needed.'
        else:
            context = context[-(self.args.max_context_len - 2):]
            response = response[:self.args.max_response_len - 1]
            return context, response

    def _ensure_dialog_length(self, utterances, remapped_index, \
                                    hard_limit = 1, lower_bound = 2):
        if len(utterances) < hard_limit:
            return False
        elif len(utterances) < lower_bound:
            positive_sample = self._get_positive(remapped_index)
            utterances.append(positive_sample['response'])
            return True
        else:
            return True
    '''
    Auxiliary Tasks
    '''
    def _build_NSP(self, remapped_index):
        '''
        Next Session Prediction
        '''
        utterances = self._get_sample(remapped_index)['context']
        if not self._ensure_dialog_length(utterances, remapped_index):
            return None
      
        if random.random() > 0.5:
            label = 1
            left_loc = random.randint(1, len(utterances) - 1)
            left_part = utterances[:left_loc]
            right_part = utterances[left_loc:]
        else:
            label = 0
            random_utterances = self._get_sample()['context']
            negative_utterances = deepcopy(random_utterances)
            self._ensure_dialog_length(negative_utterances, remapped_index)
                
            if random.random() > 0.5:
                left_loc = random.randint(1, len(utterances) - 1)
                left_part = utterances[:left_loc]
                right_loc = random.randint(1, len(negative_utterances) - 1)
                right_part = negative_utterances[right_loc:]
            else:
                left_loc = random.randint(1, len(negative_utterances) - 1)
                left_part = negative_utterances[:left_loc]
                right_loc = random.randint(1, len(utterances) - 1)
                right_part = utterances[right_loc:]
      
        left_context = self._concat_utterances(left_part)
        right_context = self._concat_utterances(right_part)
        return label, self._auto_pack(left_context, right_context)

    def _build_UR(self, remapped_index, token_ids):
        '''
        Utterance Restoration
        '''
        token_ids = deepcopy(token_ids)
        
        utterances = self._get_sample(remapped_index)['context']
        if len(utterances) == 0:
            return None
            
        utt_lens = [len(utt) for utt in utterances]
        trimed_context_len = sum(utt_lens) + len(utt_lens) + 2
        while trimed_context_len > self.args.max_context_len:
            extra_length = trimed_context_len - self.args.max_context_len
            if utt_lens[0] + 1 > extra_length:
                utt_lens[0] -= extra_length
                break
            else:
                trimed_context_len -= (utt_lens.pop(0) + 1)
        
        # Why this is needed?
        candidates = []
        for ix, utt_len in enumerate(utt_lens):
            if utt_len > 0:
                candidates.append(ix)
        if len(candidates) == 0:
            return None
      
        selected = random.sample(candidates, 1)[0]
        left_ix = sum(utt_lens[:selected]) + selected + 1
        right_ix = left_ix + utt_lens[selected]
      
        labels = []
        positions = []
        for ix in range(left_ix, right_ix):
            if token_ids[ix] != self.unk_id:
                labels.append(token_ids[ix])
                positions.append(ix)
                token_ids[ix] = self.mask_id
        if len(labels) == 0:
            return None
          
        return positions, labels, token_ids
      
    def _build_ID(self, remapped_index):
        '''
        Incoherence Detection
        '''
        utterances = self._get_sample(remapped_index)['context']
        
        # Why this is needed?
        solid_utterances = []
        for utt in utterances:
            if len(utt) > 0:
                solid_utterances.append(utt)
        utterances = solid_utterances
      
        if not self._ensure_dialog_length(utterances, remapped_index):
            return None
      
        context_len \
            = sum([len(utt) for utt in utterances]) + len(utterances) + 2
        max_context_len \
            = self.args.max_context_len + self.args.max_response_len
        while context_len > max_context_len:
            if context_len - max_context_len < len(utterances[0]):
                utterances[0] = utterances[0][context_len - max_context_len:]
                break
            else:
                context_len -= (len(utterances.pop(0)) + 1)
      
        random_utterances = self._get_sample()['context']
        random_utt_ix = random.randint(0, len(random_utterances) - 1)
        random_utt = random_utterances[random_utt_ix]
        label = random.randint(0, len(utterances) - 1)
        utterances[label] = random_utt
      
        # get utterance location
        cur_loc = 1
        locations = []
        for utt in utterances:
            right_loc = min(cur_loc + len(utt) + 1, max_context_len - 1)
            locations.append((cur_loc, right_loc))
            cur_loc = right_loc
            if cur_loc >= max_context_len - 1:
                break
        
        context = self._concat_utterances(utterances)[:max_context_len - 2]
        context = ["[CLS]"] + context + ["[SEP]"]
        token_ids = self.tokenizer.convert_tokens_to_ids(context)
        segment_ids = [0] * len(token_ids)
        attention_mask = [1] * len(token_ids)
      
        return label, token_ids, segment_ids, attention_mask, locations
        
    def _build_CD(self, remapped_index):
        '''
        Consistency Discrimination
        '''
        utterances = self._get_sample(remapped_index)['context']
        if len(utterances) < 3:  # TODO
            return None
            
        if len(utterances) == 3 or random.random() > 0.5:
            speaker = 0
        else:
            speaker = 1
        utterances = utterances[speaker::2]

        utt_pair = random.sample(utterances, 2)
        base_utt = [utt_pair[0]]
        positive_utt = [utt_pair[1]]
        
        random_utterances = self._get_sample()['context']
        random_utt_ix = random.randint(0, len(random_utterances) - 1)
        negative_utt = [random_utterances[random_utt_ix]]

        base_utt = self._concat_utterances(base_utt)
        positive_utt = self._concat_utterances(positive_utt)
        negative_utt = self._concat_utterances(negative_utt)

        return self._auto_pack(base_utt, positive_utt), \
               self._auto_pack(base_utt, negative_utt)