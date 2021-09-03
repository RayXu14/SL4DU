from data.loading.basic_dialog_dataset import BasicDialogDataset
from util import fetch_pyarrow
               

class CRMatchingDataset(BasicDialogDataset):
    
    def _get_positive(self, remapped_index):
        return fetch_pyarrow(self.samples, (remapped_index // 2) * 2)
