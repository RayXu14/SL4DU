import torch.nn as nn

from model.basic_dialog_model import BasicDialogModel


class CRMatchingModel(BasicDialogModel):

    def _init_main_task(self):
        pass   

    def main_forward(self, batch):
        return self.matching_forward(batch['main_token_ids'],
                                     batch['main_segment_ids'],
                                     batch['main_attention_mask'],
                                     batch['main_label'])
