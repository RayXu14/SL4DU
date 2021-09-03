import os

import torch

from handler.basic_finetune import BasicFinetuneHandler
from util import auto_report_RS


class RSHandler(BasicFinetuneHandler):

    def logits2preds(self, logits):
        return torch.sigmoid(logits)
    
    def report(self, preds, labels, losses):
        assert len(preds) == len(labels)
        
        if (not self.args.not_save_record) and hasattr(self, 'epoch'):
            record_path = os.path.join(self.args.log_dir,
                                       f'epoch-{self.epoch}')
            with open(record_path, 'w') as f:
                for l, p in zip(labels, preds):
                    l = int(l)
                    f.write(f'{l}\t{p}\n')
            print(f'Record of epoch {self.epoch} is recorded.')

        ''' Print result '''
        mean_loss = sum(losses) / len(losses)
        report, main_metric = auto_report_RS(labels, preds, self.args.dataset)
        print('\n'.join(['=' * 10,
                         f'Evaluation result of epoch {self.epoch}.',
                         f'\tMean loss = {mean_loss}',
                         report,
                         '=' * 10]))
        return main_metric