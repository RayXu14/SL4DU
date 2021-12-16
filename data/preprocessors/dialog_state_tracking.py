import json
import os

from data.preprocessors.basic_processor import BasicProcessor


class DSTC2Processor(BasicProcessor):

    def read_ontology(self):
        path = os.path.join(self.args.raw_data_path,
                            'scripts',
                            'config',
                            f'ontology_dstc2.json')
        with open(path, encoding='utf8') as f:
            ontology = json.load(f)
        return ontology
        
    def get_paths(self, split):
        path = os.path.join(self.args.raw_data_path,
                            'scripts',
                            'config',
                            f'dstc2_{split}.flist')
        with open(path, encoding='utf8') as f:
            flist = f.readlines()
        flist = map(lambda s: s.strip(), flist)
        flist = filter(lambda s: s != '', flist)
        return flist
        
    def read_log(self, in_dir):
        path = os.path.join(in_dir, 'log.json')
        with open(path, encoding='utf8') as f:
            log = json.load(f)
        steward_utts = []
        targets = []
        assert len(log['turns']) > 0
        for turn in log['turns']:
            utt = turn['output']['transcript'].strip()
            assert len(utt) > 0, str(in_dir) + '\n\n' + str(turn)
            steward_utts.append(self.tokenizer.tokenize(utt))
            '''
            slots = turn['output']['dialog-acts']
            if len(slots) != 1:
                targets.append(None)
                continue
            slots = slots[0]['slots']
            if len(slots) != 1:
                targets.append(None)
                continue
            if slots[0][0] == 'slot':
                target = slots[0][1]
            else:
                 target = slots[0][0]
            targets.append(target)
            '''
        return steward_utts
        
    def read_label(self, in_dir):
        path = os.path.join(in_dir, 'label.json')
        with open(path, encoding='utf8') as f:
            label = json.load(f)
        customer_utts, joint_goals, turn_labels = [], [], []
        assert len(label['turns']) > 0
        previous_goal = dict()
        for turn in label['turns']:
            goal = dict()
            for k, v in turn['goal-labels'].items():
                goal[k] = v
                
            utt = self.tokenizer.tokenize(turn['transcription'].strip())
            assert len(utt) > 0
            semantics = turn['semantics']['json']
            
            turn_label = []
            for k, v in goal.items():
                if k not in previous_goal:
                    turn_label.append(('inform', k, v))
                elif v != previous_goal[k]:
                    turn_label.append(('inform', k, v))
            for r in turn['requested-slots']:
                turn_label.append(('request', r))
            '''
            turn_label = []
            for e in semantics:
                act = e['act']
                if len(e['slots']) == 0:
                    continue
                slots = e['slots'][0]
                if act == 'inform':
                    value = slots[1]
                    if slots[0] == 'this':
                        assert target is not None
                        key = target
                        if not (in_dir == '../../data/DSTC2/data/Mar13_S1A0/voip-4c0d36762a-20130328_205236' and key == 'food'):
                        #if key not in joint_goals_acc:
                            joint_goals_acc[key] = value
                    else:
                        key = slots[0]
                        joint_goals_acc[key] = value
                    
                    turn_label.append(('inform', key, value))
                elif act != 'request':
                    assert len(slots) == 2
                    assert slots[0] == 'slot'
                    turn_label.append(('request', slots[1]))
                else:
                    continue
            '''
            
            customer_utts.append(utt)
            joint_goals.append(goal)
            turn_labels.append(turn_label)
            
            previous_goal = goal
            
        return customer_utts, joint_goals, turn_labels
        
        
    def process_set(self, split, out_file):
        ontology = self.read_ontology()
    
        in_paths = self.get_paths(split)
        dialogs = []
        for s in in_paths:
            in_dir = os.path.join(self.args.raw_data_path, 'data', s)
            steward_utts = self.read_log(in_dir)
            customer_utts, joint_goals, turn_labels = self.read_label(in_dir)
            assert len(steward_utts) == len(customer_utts) \
                                     == len(joint_goals) \
                                     == len(turn_labels)
            dialogs.append({'steward_utts': steward_utts,
                            'customer_utts': customer_utts,
                            'joint_goals': joint_goals,
                            'turn_labels': turn_labels})
        print(f'Processed {len(dialogs)} dialogs')
        
        self.write_pkl(dialogs, out_file)