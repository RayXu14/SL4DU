import os

import torch.nn as nn
from transformers import AutoModel


class CRMatchingModel(nn.Module):

    def __init__(self, args, new_tokenizer_len=None):
        super().__init__()

        ''' Initialize pretrained model '''
        path = os.path.join(args.pretrained_path, args.pretrained_model)
        self.model = AutoModel.from_pretrained(path)
        if new_tokenizer_len is not None:
            self.model.resize_token_embeddings(new_tokenizer_len)
        if args.freeze_layers > 0:
            for p in self.model.embeddings.parameters():
                p.requires_grad = False
            for layerid in range(args.freeze_layers):
                for p in self.model.encoder.layer[layerid].parameters():
                    p.requires_grad = False

        ''' Add heads '''
        self.crmatching_cls = nn.Sequential(nn.Dropout(p=args.dropout_rate),
                                 nn.Linear(self.model.config.hidden_size, 1))
        ''' Add heads '''
        self.matching_loss_fct = nn.BCEWithLogitsLoss()

    def matching_forward(self, token_ids, segment_ids, attention_mask, label, cls):
        outputs = self.model(input_ids=token_ids,
                             attention_mask=attention_mask,
                             token_type_ids=segment_ids)
        cls_logits = outputs.last_hidden_state[:, 0, :] # [batch_size, output_size]
        logits = cls(cls_logits).squeeze(-1) # [batch_size,]
        loss = self.matching_loss_fct(logits, label)
        return logits, loss

    def forward(self, batch):
        loss_dict = dict()
        
        logits, loss = self.matching_forward(batch['crm_token_ids'],
                                             batch['crm_segment_ids'],
                                             batch['crm_attention_mask'],
                                             batch['crm_label'],
                                             self.crmatching_cls)
        loss_dict['crmatching'] = loss

        return logits, loss_dict
