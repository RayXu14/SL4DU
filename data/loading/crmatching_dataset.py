from data.loading.basic_dialog_dataset import BasicDialogDataset
from util import fetch_pyarrow
               

class CRMatchingDataset(BasicDialogDataset):
    
    def _get_positive(self, remapped_index):
        return fetch_pyarrow(self.samples, (remapped_index // 2) * 2)
               

class G2RDataset(BasicDialogDataset):
        
    def _get_main_task_item(self, data_dict, remapped_index):
        sample = self._get_sample(remapped_index)
        context = self._concat_utterances(sample['context'])
        context = self.tokenizer.tokenize(sample['hint'][:self.args.max_hint_len]) + ['[SEP]'] + context
        data_dict['main_label'] = sample['label']
        data_dict['main_token_ids'], \
        data_dict['main_segment_ids'], \
        data_dict['main_attention_mask'] \
            = self._auto_pack(context, sample['response'])
    
    def _get_positive(self, remapped_index):
        return fetch_pyarrow(self.samples, (remapped_index // 2) * 2)
