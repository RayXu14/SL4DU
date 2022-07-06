import torch.nn as nn

from model.basic_dialog_model import BasicDialogModel


class CRMatchingModel(BasicDialogModel):

    def _init_main_task(self):
        self.matching_cls = nn.Sequential(nn.Dropout(p=self.args.dropout_rate),
                                nn.Linear(self.model_config.hidden_size, 1))
        self.matching_loss_fct = nn.BCEWithLogitsLoss() 

    def main_forward(self, batch):
        return self.matching_forward(batch['main_token_ids'],
                                     batch['main_segment_ids'],
                                     batch['main_attention_mask'],
                                     batch['main_label'])
