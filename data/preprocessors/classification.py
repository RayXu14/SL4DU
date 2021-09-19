import os
import re

from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from data.preprocessors.basic_processor import BasicProcessor
from data.preprocessors.swda.swda import CorpusReader


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


class SwDAProcessor(BasicProcessor):
    def read_raw(self, in_dir):
        corpus = CorpusReader(in_dir)
        dialog_data = []
        all_acts = []
        for trans in corpus.iter_transcripts():
            dialog = []
            acts = []
            for utt in trans.utterances:
                clean_words = utt.text_words(filter_disfluency=True)
                text = ' '.join(clean_words)
                
                # Remove punctuation
                text = re.sub('[(|)|#|.]', '', text)
                # Remove dashes and words in angle brackets (e.g. "<Laughter>")
                text = re.sub('\W-+\W|<\w+>', ' ', text)
                
                # Remove extra spaces
                text = re.sub('\s+', ' ', text)
                # Remove data rows that end up empty after cleaning
                if text == ' ':
                    continue
                    
                text = self.tokenizer.tokenize(text.strip())
                
                act = utt.damsl_act_tag()
                if act == '+':
                    act = None
                else:
                    all_acts.append(act)
                
                dialog.append(text)
                acts.append(act)
            
            dialog_data.append({'dialog': dialog,
                                'acts': acts,})
    
        label_encoder = LabelEncoder()
        label_encoder.fit(all_acts)
        for sample in dialog_data:
            acts = sample['acts']
            for i in range(len(acts)):
                if acts[i] is not None:
                    acts[i] = label_encoder.transform([acts[i]])[0]
            
        return dialog_data, label_encoder.classes_

    def process_set(self, in_dir, out_file):
        in_dir = os.path.join(self.args.raw_data_path, in_dir)

        dialog_data, classes = self.read_raw(in_dir)
        print(f'Processed {len(dialog_data)} cases from {in_dir}')
        
        self.write_pkl(dialog_data, out_file)
        class_path = os.path.join(self.args.pkl_data_path, out_file + '.classes')
        with open(class_path, 'w') as f:
            for cls in classes:
                f.write(cls + '\n')
        print(f'==> Saved classes to {class_path}.')