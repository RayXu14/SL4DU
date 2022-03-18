import os

import numpy as np
import torch

from handler.basic_finetune import BasicFinetuneHandler
from util import bleu_metric, normalize_answer, distinct_metric


class RGHandler(BasicFinetuneHandler):
    
    def get_label_from_batch(self, batch):
        return batch['main_gen_response']
        
    def logits2preds(self, batch_rids):
        #pred_responses = [self.tokenizer.convert_ids_to_tokens(rids) \
        #                    for rids in batch_rids]
        pred_responses = self.tokenizer.batch_decode(batch_rids)
        return pred_responses
    
    def report(self, preds, labels, losses):
        assert len(preds) == len(labels)
        
        if (not self.args.not_save_record) and hasattr(self, 'epoch'):
            record_path = os.path.join(self.args.log_dir,
                                       f'epoch-{self.epoch}')
            with open(record_path, 'w') as f:
                for l, p in zip(labels, preds):
                    f.write(f'{l}\n{p}\n\n')
            print(f'Record of epoch {self.epoch} is recorded.')

        ''' Print result '''
        perplexity = np.mean(np.exp(losses))
        b1, b2, b3, b4 = bleu_metric(preds, labels)
        d1, d2 = distinct_metric(preds)
        print('\n'.join(['=' * 10,
                         f'Evaluation result of epoch {self.epoch}.',
                         f'Perplexity = {perplexity}',
                         f'BLEU = {b1}\t{b2}\t{b3}\t{b4}',
                         f'Distinct-1 = {d1}',
                         f'Distinct-2 = {d2}',
                         '=' * 10]))
        return (b1 + b2 + b3 + b4) / 4