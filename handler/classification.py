from sklearn.metrics import f1_score, precision_score, recall_score
import torch

from handler.basic_finetune import BasicFinetuneHandler


class CLSHandler(BasicFinetuneHandler):

    def logits2preds(self, logits):
        max_idx = torch.max(logits, dim=-1).indices
        return max_idx
    
    def report(self, preds, labels, losses):
        assert len(preds) == len(labels)

        report = []
        if self.args.label_name == 'emotion' and self.args.dataset == 'Daily':
            micro_f1 = f1_score(labels, preds, average='micro',
                                labels=[0, 2, 3, 4, 5, 6])
            report.append(f'Micro F1\t= {micro_f1}')
        macro_f1 = f1_score(labels, preds, average='macro')
        report.append(f'Macro F1\t= {macro_f1}')
        macro_precision = precision_score(labels, preds, average='macro') 
        report.append(f'Macro P\t= {macro_precision}')
        macro_recall = recall_score(labels, preds, average='macro')    
        report.append(f'Macro R\t= {macro_recall}')
        
        mean_loss = sum(losses) / len(losses)
        print('\n'.join(['=' * 10,
                         f'Evaluation result of epoch {self.epoch}.',
                         f'\tMean loss = {mean_loss}',
                         '\n'.join(report),
                         '=' * 10]))
        return macro_f1
