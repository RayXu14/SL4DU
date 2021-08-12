import os

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead


class CRMatchingModel(nn.Module):

    def __init__(self, args, new_tokenizer_len=None):
        super().__init__()

        ''' Initialize pretrained model '''
        path = os.path.join(args.pretrained_path, args.pretrained_model)
        model_config = AutoConfig.from_pretrained(path)
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
        self.crmatching_cls = nn.Sequential(nn.Dropout(p=args.dropout_rate),
                                     nn.Linear(model_config.hidden_size, 1))
        self.matching_loss_fct = nn.BCEWithLogitsLoss()
        
        if hasattr(args, 'use_UR') and args.use_UR:
            self.UR_cls = BertOnlyMLMHead(model_config)
            self.UR_loss_fct = nn.CrossEntropyLoss()
            
        if hasattr(args, 'use_ID') and args.use_ID:
            self.ID_cls = nn.Sequential(nn.Dropout(p=args.dropout_rate),
                             nn.Linear(model_config.hidden_size * 2, 1))
            self.ID_loss_fct = nn.CrossEntropyLoss()
            
        if hasattr(args, 'use_CD') and args.use_CD:
            self.CD_cls = nn.Sequential(nn.Dropout(p=args.dropout_rate),
                                 nn.Linear(model_config.hidden_size, 1))
            self.CD_loss_fct = nn.MarginRankingLoss(args.margin)            

    def matching_forward(self, token_ids, segment_ids, attention_mask, label):
        outputs = self.model(input_ids=token_ids,
                             attention_mask=attention_mask,
                             token_type_ids=segment_ids)
        cls_hidden = outputs.last_hidden_state[:, 0, :] # [batch_size, hidden]
        logits = self.crmatching_cls(cls_hidden).squeeze(-1) # [batch_size,]
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
            in enumerate(zip(outputs, locations.numpy())):
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
    
    def CD_forward(self, pos_token_ids, pos_segment_ids, pos_attention_mask,
                         neg_token_ids, neg_segment_ids, neg_attention_mask):
        pos_outputs = self.model(input_ids=pos_token_ids,
                             attention_mask=pos_segment_ids,
                             token_type_ids=pos_attention_mask)
        pos_cls_hidden = pos_outputs.last_hidden_state[:, 0, :]
        pos_logits = self.CD_cls(pos_cls_hidden).squeeze(-1)
        pos_pred = torch.sigmoid(pos_logits)
        
        neg_outputs = self.model(input_ids=neg_token_ids,
                                attention_mask=neg_segment_ids,
                                token_type_ids=neg_attention_mask)
        neg_cls_hidden = neg_outputs.last_hidden_state[:, 0, :]
        neg_logits = self.CD_cls(neg_cls_hidden).squeeze(-1)
        neg_pred = torch.sigmoid(neg_logits)
        
        loss = self.CD_loss_fct(pos_pred, neg_pred,
                            torch.ones(pos_pred.shape[0]).to(pos_pred.device))
        return loss

    def forward(self, batch):
        loss_dict = dict()
        
        logits, loss = self.matching_forward(batch['crm_token_ids'],
                                             batch['crm_segment_ids'],
                                             batch['crm_attention_mask'],
                                             batch['crm_label'])
        loss_dict['crmatching'] = loss

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
        
        if 'cd_pos_token_ids' in batch:
            loss_dict['CD'] = self.CD_forward(batch['cd_pos_token_ids'],
                                           batch['cd_pos_segment_ids'],
                                           batch['cd_pos_attention_mask'],
                                           batch['cd_neg_token_ids'],
                                           batch['cd_neg_segment_ids'],
                                           batch['cd_neg_attention_mask'])

        return logits, loss_dict
