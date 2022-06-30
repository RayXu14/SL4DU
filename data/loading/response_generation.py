from tqdm import tqdm

from data.loading.basic_dialog_dataset import BasicDialogDataset
               

class ResponseGenerationDataset(BasicDialogDataset):

    def _process_samples(self, samples):
        dialogs = []
        for sample in tqdm(samples, desc='Process RG samples...'):
            if self.is_train:
                dialog = sample['dialog']
                for i in range(1, len(dialog)):
                    dialogs.append({'context': dialog[:i],
                                    'response': dialog[i]})
            else:
                dialogs.append({'context': sample['dialog'][:-1],
                                'response': sample['dialog'][-1]})
        return dialogs
        
    def _get_main_task_item(self, data_dict, remapped_index):
        sample = self._get_sample(remapped_index)
        
        context = self._concat_utterances(sample['context'])
        response = sample['response']
    
        token_ids, segment_ids, attention_mask = self._auto_pack(context, response)
        
        generation_len = min(len(response) + 1, self.args.max_response_len)
        right_ix = len(token_ids)
        left_ix = right_ix - generation_len
        assert left_ix > 0
        labels = []
        positions = []
        for ix in range(left_ix, right_ix):
            if token_ids[ix] != self.unk_id:
                labels.append(token_ids[ix])
                positions.append(ix - 1)
        
        ''' create UniLM mask '''
        UniLMmask = [[1] * left_ix + [0] * generation_len for _ in range(left_ix)]
        for mid in range(left_ix + 1, right_ix + 1):
            UniLMmask.append([1] * mid + [0] * (right_ix - mid))
                
        data_dict['main_token_ids'] = token_ids
        data_dict['main_segment_ids'] = segment_ids
        data_dict['main_attention_mask'] = segment_ids
        data_dict['main_positions'] = positions
        data_dict['main_labels'] = labels
        data_dict['main_UniLMmask'] = UniLMmask
        if not self.is_train:
            token_ids, segment_ids, attention_mask = \
                self._auto_pack(context, response)
            
            data_dict['main_gen_token_ids'] = token_ids
            data_dict['main_gen_segment_ids'] = segment_ids
            data_dict['main_gen_attention_mask'] = attention_mask
            data_dict['main_gen_init_generation'] = len(token_ids) - 1
            data_dict['main_gen_response'] = \
                self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(response))
