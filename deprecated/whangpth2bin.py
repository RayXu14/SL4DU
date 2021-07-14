import torch

whangpth_path = '/workspace/pretrained/post-bert-base-uncased/pytorch_model.pth'
bin_path = '/workspace/pretrained/post-bert-base-uncased/pytorch_model.bin'

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