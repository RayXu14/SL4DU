import os
import re

from sklearn.preprocessing import LabelEncoder
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


class SwDAProcessor(BasicProcessor):
    def read_raw(self, in_dir):

        from data.data_rep.swda.swda import CorpusReader

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
                
                text = text.strip()
                if self.tokenization:
                    text = self.tokenizer.tokenize(text)
                
                act = utt.damsl_act_tag()
                if act == '+':
                    act = None
                else:
                    all_acts.append(act)
                
                dialog.append(text)
                acts.append(act)
            
            dialog_data.append({'dialog': dialog,
                                'acts': acts,})
    
        if not hasattr(self, 'label_encoder'):
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(all_acts)
            classes = self.label_encoder.classes_
        else:
            classes = None
        for sample in dialog_data:
            acts = sample['acts']
            for i in range(len(acts)):
                if acts[i] is not None:
                    acts[i] = self.label_encoder.transform([acts[i]])[0]
            
        return dialog_data, classes

    def process_set(self, in_dir, out_file):
        in_dir = os.path.join(self.args.raw_data_path, in_dir)

        self.tokenization = True
        dialog_data, classes = self.read_raw(in_dir)
        print(f'Read {len(dialog_data)} cases from {in_dir}')
        
        self.write_pkl(dialog_data, out_file)
        if classes is not None:
            class_path = os.path.join(self.args.pkl_data_path, out_file + '.classes')
            with open(class_path, 'w') as f:
                for cls in classes:
                    f.write(cls + '\n')
            print(f'==> Saved classes to {class_path}.')
            
    def make_post_training_corpus(self):
        in_dir = os.path.join(self.args.raw_data_path, self.args.raw_train_file)

        self.tokenization = False
        dialog_data, _ = self.read_raw(in_dir)
        print(f'Read {len(dialog_data)} cases from {in_dir}')
        
        post_file = os.path.join(self.args.pkl_data_path,
                                 self.args.raw_train_file + '.post')
        with open(post_file, 'w') as f:
            for example in dialog_data:
                dialog = example['dialog']
                for utt in dialog:
                    f.write(utt + '\n')
                f.write('\n')
