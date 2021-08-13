import os
import time

import scipy.stats
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import CRMatchingDataset, ChunkedRandomSampler, collate_fn, \
                 init_tokenizer
from model import CRMatchingModel
from util import batch2cuda, tensor2list, \
                 visualize_model, auto_report_metrics, smart_model_loading

class FinetuneHandler:

    def __init__(self, args, mode):
        self.args = args
        self.mode = mode

        self.tokenizer = init_tokenizer(args)
        if args.add_EOT:
            new_tokenizer_len = len(self.tokenizer)
        else:
            new_tokenizer_len = None

        ''' Build model '''
        time_start = time.time()
        self.model = CRMatchingModel(args, new_tokenizer_len)
        if args.load_path is not None:
            checkpoint = torch.load(args.load_path)
            self.model = smart_model_loading(self.model, checkpoint)
            self.epoch = checkpoint['epoch']
        else:
            checkpoint = None
            self.epoch = -1
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            self.parallel = True
        else:
            self.parallel = False
        self.model = self.model.cuda()
        visualize_model(self.model)
        print(f'Builded Model in {time.time() - time_start}s.')

        if mode == 'train':
            ''' Initialize optimization '''
            self.best_epoch = -1
            self.best_metric = 0.
            trainable_parameters = list(filter(lambda p: p.requires_grad,
                                               self.model.parameters()))
            self.optimizer = optim.Adam(trainable_parameters,
                                        lr=args.learning_rate)
            if checkpoint is not None:
                self.best_epoch = checkpoint['epoch']
                self.best_metric = checkpoint['metric']
                self.optimizer.load_state_dict(
                    checkpoint['optimizer_state_dict'])

            ''' Build data '''
            self.train_loader = self.build_dataloader(args.pkl_train_file,
                                                      args.train_batch_size,
                                                      is_shuffle=True)
            self.eval_loader = self.build_dataloader(args.pkl_valid_file,
                                                     args.eval_batch_size,
                                                     is_shuffle=False)
        elif mode == 'eval':
            self.eval_loader = self.build_dataloader(args.pkl_test_file,
                                                     args.eval_batch_size,
                                                     is_shuffle=False)
        else:
            raise NotImplementedError('Not supported handler mode.')

    def build_dataloader(self, filename, batch_size, is_shuffle):
        time_start = time.time()
        dataset = CRMatchingDataset(self.args,
                                    filename,
                                    self.tokenizer,
                                    is_shuffle=is_shuffle)
        chunkedSampler = ChunkedRandomSampler(dataset, batch_size)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=self.args.cpu_workers,
                                sampler=chunkedSampler,
                                drop_last=False,
                                collate_fn=collate_fn)
        print(f'Builded data from {filename} in {time.time() - time_start}s.')
        return dataloader

    def run_model(self, batch):
        '''
        Officially, it is not recommended to
        move tensors to CUDA device in collate_fn
        (https://pytorch.org/docs/stable/data.html#multi-process-data-loading)
        So move tensors to GPU after fetch them from Dataloader.
        '''
        cuda_batch = batch2cuda(batch)
        logits, losses = self.model(cuda_batch)
        if self.parallel:
            for task in losses:
                losses[task] = losses[task].mean()
        return logits, losses

    def eval(self):
        time_start = time.time()
        print('=' * 10
                + f'\nStart evaluation of epoch {self.epoch}.\n'
                + '=' * 10)
        all_losses = []
        all_preds = []
        all_labels = []
        '''
        Let dropout & batchnorm layer be in eval mode
        and require the model do not calculate gradient
        to reduce GPU calculation amount & memory usage.
        (https://blog.csdn.net/songyu0120/article/details/103884586)
        '''
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.eval_loader):
                logits, losses = self.run_model(batch)
                pred = tensor2list(torch.sigmoid(logits))

                all_losses.append(losses['crmatching'].item())
                all_preds.extend(pred)
                all_labels.extend(tensor2list(batch["crm_label"]))

                if (batch_idx + 1) % self.args.eval_view_every == 0:
                    print(f'Evaluated {batch_idx + 1}-th batches.')

        ''' Save record '''
        if (not self.args.not_save_record) and hasattr(self, 'epoch'):
            record_path = os.path.join(self.args.log_dir,
                                       f'epoch-{self.epoch}')
            with open(record_path, 'w') as f:
                for l, p in zip(all_labels, all_preds):
                    l = int(l)
                    f.write(f'{l}\t{p}\n')
            print(f'Record of epoch {self.epoch} is recorded.')

        ''' Print result '''
        mean_loss = sum(all_losses) / len(all_losses)
        report, main_metric = auto_report_metrics(all_labels,
                                                  all_preds,
                                                  self.args.task)
        print('\n'.join(['=' * 10,
                         f'Evaluation result of epoch {self.epoch}.',
                         f'\tMean loss = {mean_loss}',
                         report,
                         '=' * 10]))

        ''' Assess correlation betweem predictions and labels '''
        if self.args.assess:
            s_score = scipy.stats.spearmanr(all_labels, all_preds)
            p_score = scipy.stats.pearsonr(all_labels, all_preds)
            report = '\n'.join(['=' * 10,
                                f'Assessment result:',
                                f'Spearman score = {s_score}',
                                f'Pearson score  = {p_score}',
                                '=' * 10])
            print(report)

        if self.mode == 'train' and main_metric > self.best_metric:
            ''' Save checkpoint '''
            print(f'Last best: {self.best_metric} epoch: {self.best_epoch}')
            print(f'NEW BEST RESULTS!')
            self.best_epoch = self.epoch
            self.best_metric = main_metric
            if self.args.save_ckpt:
                save_path = os.path.join(self.args.log_dir,
                                         self.args.ckpt_name)
                if isinstance(self.model, nn.DataParallel):
                    model_state_dict = self.model.module.state_dict()
                else:
                    model_state_dict = self.model.state_dict()
                torch.save({'epoch': self.best_epoch,
                            'metric': self.best_metric,
                            'model_state_dict': model_state_dict,
                            'optimizer_state_dict':
                                self.optimizer.state_dict()},
                           save_path)
                print(f'Ckeckpoint of epoch {self.epoch}'
                    + f' is saved to {save_path}.')
            else:
                print('Discard checkpoint.')
        else:
            print('NOT a better checkpoint.')

        torch.cuda.empty_cache()
        print(f'This Evaluation take {(time.time() - time_start) / 3600}h.')

    def infer(self):
        time_start = time.time()
        print('=' * 10 + f'\nStart inferencing.\n' + '=' * 10)
        all_preds = []
        all_labels = []

        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.eval_loader):
                logits, _ = self.run_model(batch)
                pred = tensor2list(torch.sigmoid(logits))

                all_preds.extend(pred)
                all_labels.extend(tensor2list(batch["crm_label"]))

        torch.cuda.empty_cache()
        print(f'This Inferencing take {(time.time() - time_start) / 3600}h.')

    def train_epoch(self):
        time_start = time.time()
        self.epoch += 1
        assert self.train_loader is not None, 'Not training mode.'
        print(f'Start training of epoch {self.epoch}.')

        ''' Initialize training '''
        self.model.train()

        virtual_batch_losses = dict()
        virtual_batch_losses['crmatching'] = 0.
        accumulate_batch = 0

        if self.args.use_NSP:
            virtual_batch_losses['NSP'] = 0.
        if self.args.use_UR:
            virtual_batch_losses['UR'] = 0.
        if self.args.use_ID:
            virtual_batch_losses['ID'] = 0.
        if self.args.use_CD:
            virtual_batch_losses['CD'] = 0.

        for batch_idx, batch in enumerate(self.train_loader):
            _, losses = self.run_model(batch)
            ''' Calculate losses '''
            train_loss = 0.
            for task in losses:
                losses[task] *= self.args.train_batch_size \
                                    / self.args.virtual_batch_size
                train_loss += losses[task]
                virtual_batch_losses[task] += losses[task].item()
            accumulate_batch += batch['crm_label'].shape[0]

            ''' Update '''
            train_loss.backward()
            if accumulate_batch == self.args.virtual_batch_size or \
                batch_idx + 1 == len(self.train_loader): # last batch
                self.optimizer.step()
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.args.max_gradient_norm)
                self.optimizer.zero_grad()

                ''' View training process '''
                if (batch_idx + 1) % self.args.train_view_every == 0:
                    report = [
                        '[Epoch: {:3d}][Iter: {:6d}/{:6d}][lr: {:7f}]'.format(
                            self.epoch, batch_idx + 1, len(self.train_loader),
                            self.optimizer.param_groups[0]['lr'])]
                    for task, task_loss in virtual_batch_losses.items():
                        report.append(f'\t{task} Loss = {task_loss}')
                    print('\n'.join(report))

                ''' reset virtual batch accumulation '''
                accumulate_batch = 0
                for task in virtual_batch_losses:
                    virtual_batch_losses[task] = 0.

        torch.cuda.empty_cache()
        print(f'This train epoch take {(time.time() - time_start) / 3600}h.')