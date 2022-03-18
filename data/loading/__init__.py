import random

from torch.utils.data import Dataset, Sampler

from data.loading.collate import collate_fn
        
        
def get_dataset(args, filename, tokenizer, is_train):
    if args.task == 'CLS':
        from data.loading.cls_dataset import ClassificationDataset
        return ClassificationDataset(args, filename, tokenizer, is_train)
    elif args.task == 'RS':
        from data.loading.crmatching_dataset import CRMatchingDataset
        return CRMatchingDataset(args, filename, tokenizer, is_train)
    elif args.task == 'RG':
        from data.loading.response_generation import ResponseGenerationDataset
        return ResponseGenerationDataset(args, filename, tokenizer, is_train)
    else:
        raise NotImplementedError('Not supported task.')


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
