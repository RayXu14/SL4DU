from tqdm import tqdm

from data.loading.basic_dialog_dataset import BasicDialogDataset
               

class ClassificationDataset(BasicDialogDataset):

    def _process_samples(self, samples):
        full_samples = []
        self.proper_ids = []
        for sample in tqdm(samples, desc='Process CLS samples...'):
            for i in range(len(sample['dialog'])):
                label = sample[self.args.label_name + 's'][i]
                if i > 0 and label is not None:
                    self.proper_ids.append(len(full_samples))
                    full_samples.append({'context': sample['dialog'][:i],
                                         'response': sample['dialog'][i],
                                         'label': label})
        assert len(self.proper_ids) > 0, 'No sample has context.'
        return full_samples