from abc import abstractmethod
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead


class BasicDialogModel(nn.Module):

    def __init__(self, args, new_tokenizer_len=None):
        super().__init__()
        self.args = args

        ''' Initialize pretrained model '''
        path = os.path.join(args.pretrained_path, args.pretrained_model)
        self.model_config = AutoConfig.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path)
        if new_tokenizer_len is not None:
            self.model.resize_token_embeddings(new_tokenizer_len)
        if args.freeze_layers > 0:
            for p in self.model.embeddings.parameters():
                p.requires_grad = False
            for layerid in range(args.freeze_layers):
                for p in self.model.encoder.layer[layerid].parameters():
                    p.requires_grad = False

        ''' Add heads and loss_fct '''
        self._init_main_task()
        if hasattr(args, 'use_NSP') and args.use_NSP:
            self.matching_cls = nn.Sequential(nn.Dropout(p=self.args.dropout_rate),
                                    nn.Linear(self.model_config.hidden_size, 1))
            self.matching_loss_fct = nn.BCEWithLogitsLoss() 
        
        if hasattr(args, 'use_UR') and args.use_UR:
            self.UR_cls = BertOnlyMLMHead(self.model_config)
            self.UR_loss_fct = nn.CrossEntropyLoss()
            
        if hasattr(args, 'use_ID') and args.use_ID:
            self.ID_cls = nn.Sequential(nn.Dropout(p=args.dropout_rate),
                             nn.Linear(self.model_config.hidden_size * 2, 1))
            self.ID_loss_fct = nn.CrossEntropyLoss()
            
        if hasattr(args, 'use_CD') and args.use_CD:
            self.CD_cls = nn.Sequential(nn.Dropout(p=args.dropout_rate),
                                 nn.Linear(self.model_config.hidden_size, 1))
            self.CD_loss_fct = nn.BCEWithLogitsLoss()        

    def _init_main_task(self):
        raise NotImplementedError('No main task pointed.')
    
    @abstractmethod
    def main_forward(self):
        pass
    
    def matching_forward(self, token_ids, segment_ids, attention_mask, label):
        outputs = self.model(input_ids=token_ids,
                             attention_mask=attention_mask,
                             token_type_ids=segment_ids)
        cls_hidden = outputs.last_hidden_state[:, 0, :] # [batch_size, hidden]
        logits = self.matching_cls(cls_hidden).squeeze(-1) # [batch_size, ]
        loss = self.matching_loss_fct(logits, label)
        return logits, loss
    
    def UR_forward(self, token_ids, segment_ids, attention_mask, \
                   positions, labels):
        outputs = self.model(input_ids=token_ids,
                             attention_mask=attention_mask,
                             token_type_ids=segment_ids)
        outputs = outputs.last_hidden_state # [batch_size, seq_len, hidden]
        outputs = outputs.reshape(-1, outputs.shape[-1])
        positions = positions.unsqueeze(-1).expand(-1, outputs.shape[-1])
        outputs = torch.gather(outputs, 0, positions)
        probability = self.UR_cls(outputs)
        loss = self.UR_loss_fct(probability, labels)
        return loss
  
    def ID_forward(self, token_ids, segment_ids, attention_mask,
                         label, locations):
        outputs = self.model(input_ids=token_ids,
                             attention_mask=attention_mask,
                             token_type_ids=segment_ids)
        outputs = outputs.last_hidden_state
        losses = []
        for i, (sample_hiddens, loc_tuple_list) \
            in enumerate(zip(outputs, locations)):
            utt_reps = []
            for loc_a, loc_b in loc_tuple_list:
                utt_rep = sample_hiddens[loc_a:loc_b]
                max_rep = torch.max(utt_rep, dim=0).values
                mean_rep = torch.mean(utt_rep, dim=0)
                fused_rep = torch.cat([max_rep, mean_rep], dim=-1)
                utt_reps.append(fused_rep)
            utt_reps = torch.stack(utt_reps, dim=0)
            logits = self.ID_cls(utt_reps).squeeze(-1)
            loss = self.ID_loss_fct(logits.unsqueeze(0),
                                    label[i].unsqueeze(0).long())
            losses.append(loss)
        return sum(losses) / len(losses)
    
    def CD_forward(self, label, token_ids, segment_ids, attention_mask):
        outputs = self.model(input_ids=token_ids,
                             attention_mask=attention_mask,
                             token_type_ids=segment_ids)
        cls_hidden = outputs.last_hidden_state[:, 0, :] # [batch_size, hidden]
        # TODO same or different to/from main matching task?
        logits = self.CD_cls(cls_hidden).squeeze(-1) # [batch_size,]
        loss = self.CD_loss_fct(logits, label)
        return loss

    def forward(self, batch):
        loss_dict = dict()
        
        logits, loss = self.main_forward(batch)
        loss_dict['main'] = loss

        if 'nsp_token_ids' in batch:
            _, loss_dict['NSP'] = self.matching_forward(
                                            batch['nsp_token_ids'],
                                            batch['nsp_segment_ids'],
                                            batch['nsp_attention_mask'],
                                            batch['nsp_label'])

        if 'ur_token_ids' in batch:
            loss_dict['UR'] = self.UR_forward(batch['ur_token_ids'],
                                              batch['ur_segment_ids'],
                                              batch['ur_attention_mask'],
                                              batch['ur_positions'],
                                              batch['ur_labels'])
      
        if 'id_token_ids' in batch:
            loss_dict['ID'] = self.ID_forward(batch['id_token_ids'],
                                           batch['id_segment_ids'],
                                           batch['id_attention_mask'],
                                           batch['id_label'],
                                           batch['id_locations'])
        
        if 'cd_token_ids' in batch:
            loss_dict['CD'] = self.CD_forward(batch['cd_label'],
                                              batch['cd_token_ids'],
                                              batch['cd_segment_ids'],
                                              batch['cd_attention_mask'])

        return logits, loss_dict
