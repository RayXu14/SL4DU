def data_stat(filename):
    min_turn = 10
    max_turn = 0
    all_turns = []
    all_utt_lens = []
    all_resp_lens = []
    with open(filename) as f:
        for line in f:
            eles = line.strip().split('\t')
            turns = len(eles) - 2
            if turns < min_turn:
                min_turn = turns
            elif turns > max_turn:
                max_turn = turns
            all_turns.append(turns)
            all_utt_lens.extend([len(e.split()) for e in eles[:-1]])
            all_resp_lens.append(len(eles[-1].split()))
                
    print(f'{filename} min turn number = {min_turn} max turn number = {max_turn}')
    print(f'\tavg turn = {sum(all_turns)/len(all_turns)}')
    print(f'\tavg tokens/per dialog = {sum(all_utt_lens)/len(all_utt_lens)}')
    print(f'\tavg resp tokens = {sum(all_resp_lens)/len(all_resp_lens)}')


data_dir = '/share/xurj/AAAI-2020-selfSpvMatching/data/ecd/'

data_stat(data_dir + 'train.txt')
data_stat(data_dir + 'dev.txt')
data_stat(data_dir + 'test.txt')