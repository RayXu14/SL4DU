import torch

whangpth_path = '../BERT-ResSel/results/bert_ubuntu_pt/post_training/20210926-225427/checkpoints/checkpoint_20.pth'
bin_path = '../../pretrained/post-SwDA-bert-base-uncased/pytorch_model.bin'

state_dict = torch.load(whangpth_path, map_location='cpu')
new_state_dict = dict()

'''
for k in state_dict['model'].keys():
    if k.startswith('_bert_model.bert.encoder'):
        continue
    print(k, state_dict['model'][k].shape)
assert False
'''

for state_key in state_dict['model'].keys():
    if state_key.startswith('_bert_model.'):
        new_state_dict[state_key[len('_bert_model.'):]] = state_dict['model'][state_key]
        
torch.save(new_state_dict, bin_path)
