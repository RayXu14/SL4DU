import numpy as np
import torch
import torch.nn as nn

from data.utils import init_tokenizer
from model.basic_dialog_model import BasicDialogModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from util import tensor2np, tensor2list


class ResponseGenerationModel(BasicDialogModel):

    def _init_main_task(self):
        self.RG_cls = BertOnlyMLMHead(self.model_config)
        self.RG_loss_fct = nn.CrossEntropyLoss()  
        tokenizer = init_tokenizer(self.args)
        self.sep_id, self.pad_id = \
            tokenizer.convert_tokens_to_ids(['[SEP]', '[PAD]'])

    def main_forward(self, batch):
        outputs = self.model(input_ids=batch['main_token_ids'],
                             attention_mask=batch['main_UniLMmask'],
                             token_type_ids=batch['main_segment_ids'])
        outputs = outputs.last_hidden_state # [batch_size, seq_len, hidden]
        outputs = outputs.reshape(-1, outputs.shape[-1])
        positions = batch['main_positions'].unsqueeze(-1).expand(-1, outputs.shape[-1])
        outputs = torch.gather(outputs, 0, positions)
        probability = self.RG_cls(outputs)
        loss = self.RG_loss_fct(probability, batch['main_labels'])
        pred_responses = None
        if not self.training:
            device = batch['main_gen_token_ids'].device
            gen_pos = batch['main_gen_init_generation']
            batch_size = len(gen_pos)
            batch_seq = list(range(batch_size))
            reach_eos = np.array([False] * len(gen_pos))
            pred_responses = [[] for _ in range(batch_size)]
            max_len = self.args.max_context_len + self.args.max_response_len
            long_addition = np.array([[1] for _ in range(batch_size)])
            float_addition = np.array([[1.] for _ in range(batch_size)])
            
            input_ids = tensor2np(batch['main_gen_token_ids'])
            attention_mask = tensor2np(batch['main_gen_attention_mask'])
            segment_ids = tensor2np(batch['main_gen_segment_ids'])
            
            while not reach_eos.all():
                tensor_input_ids = torch.LongTensor(input_ids).to(device)
                tensor_segment_ids = torch.LongTensor(segment_ids).to(device)
                tensor_attention_mask = torch.FloatTensor(attention_mask).to(device)
            
                # prediction
                outputs = self.model(input_ids=tensor_input_ids,
                                     attention_mask=tensor_attention_mask,
                                     token_type_ids=tensor_segment_ids)
                outputs = outputs.last_hidden_state # [batch_size, seq_len, hidden]
                outputs = outputs[batch_seq, gen_pos] # [batch_size, hidden]
                probability = self.RG_cls(outputs) # [batch_size, vocab_size]
                pred_tokens = torch.argmax(probability, dim=-1)
                
                # update
                pred_tokens = tensor2np(pred_tokens)
                reach_eos |= (pred_tokens == self.sep_id)
                
                if len(input_ids[0]) < max_len:
                    input_ids = np.concatenate((input_ids, long_addition * self.pad_id), axis=-1)
                    segment_ids = np.concatenate((segment_ids, long_addition), axis=-1)
                    attention_mask = np.concatenate((attention_mask, float_addition), axis=-1)
                
                for i in range(batch_size):
                    if reach_eos[i]:
                        continue
                    pred_responses[i].append(pred_tokens[i])
                    if pred_tokens[i] == self.sep_id:
                        reach_eos[i] = True
                    if gen_pos[i] < max_len - 1:
                        gen_pos[i] += 1
                        input_ids[i][gen_pos[i]] = pred_tokens[i]
                    else:
                        reach_eos[i] = True
        return pred_responses, loss