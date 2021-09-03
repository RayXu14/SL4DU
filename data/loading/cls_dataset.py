from tqdm import tqdm

from data.loading.basic_dialog_dataset import BasicDialogDataset
from util import fetch_pyarrow
               

class ClassificationDataset(BasicDialogDataset):

    def _process_samples(self, samples):
        full_samples = []
        self.proper_ids = []
        for sample in tqdm(samples):
            for i in range(len(sample['dialog'])):
                if i > 0:
                    self.proper_ids.append(len(full_samples))
                full_samples.append({'context': sample['dialog'][:i],
                                     'response': sample['dialog'][i],
                                     'label': sample[self.args.label_name + 's'][i]})
        assert len(self.proper_ids) > 0, 'No sample has context.'
        return full_samples
    
    def _get_positive(self, remapped_index):
        return fetch_pyarrow(self.samples, remapped_index)