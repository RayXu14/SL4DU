import os

from tqdm import tqdm

from data.preprocessors.basic_processor import BasicProcessor


class DailyProcessor(BasicProcessor):

    def read_raw(self, in_dir):
        set_name = in_dir.split('/')[-1]
        text_path = os.path.join(in_dir, f'dialogues_{set_name}.txt')
        act_path = os.path.join(in_dir, f'dialogues_act_{set_name}.txt')
        emo_path = os.path.join(in_dir, f'dialogues_emotion_{set_name}.txt')
    
        dialog_data = []
        with open(text_path, encoding='utf8') as ft, \
             open(act_path, encoding='utf8') as fa, \
             open(emo_path, encoding='utf8') as fe:
            for tline, aline, eline in tqdm(zip(ft, fa, fe), 
                    desc=f'Reading [{in_dir}] in Daily style'):
                tline = tline.strip()
                if len(tline) == 0:
                    continue
                dialog_raw = tline.split('__eou__')[:-1]
                dialog = [self.tokenizer.tokenize(turn.strip())
                            for turn in dialog_raw]
                if len(dialog) <= 1:
                    continue
                
                acts = [int(act) - 1 for act in aline.strip().split()]
                emos = [int(emo) for emo in eline.strip().split()]
                assert len(dialog) == len(acts) == len(emos)
                dialog_data.append({'dialog': dialog,
                                    'acts': acts,
                                    'emotions': emos})
        return dialog_data
    