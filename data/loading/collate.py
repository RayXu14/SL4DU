import torch

def collate_fn(data_dict_batch):
    tensor_batch = {}
    collate_task(data_dict_batch, tensor_batch, 'main')
    collate_task(data_dict_batch, tensor_batch, 'nsp')
    collate_task(data_dict_batch, tensor_batch, 'ur')
    collate_task(data_dict_batch, tensor_batch, 'id')
    collate_task(data_dict_batch, tensor_batch, 'cd')
    return tensor_batch


def collate_task(data_dict_batch, tensor_batch, task):
    tids_key = f'{task}_token_ids'
    sids_key = f'{task}_segment_ids'
    amask_key = f'{task}_attention_mask'
    label_key = f'{task}_label'
    positions_key = f'{task}_positions' # UR
    labels_key = f'{task}_labels' # UR
    locations_key = f'{task}_locations' # ID

    ''' Ignore empty batch'''
    non_empty_batch = []
    for data_dict in data_dict_batch:
        if tids_key in data_dict:
            non_empty_batch.append(data_dict)
    if len(non_empty_batch) == 0:
        return
    data_dict_batch = non_empty_batch

    ''' Padding '''
    max_len = max([len(e[tids_key]) for e in data_dict_batch])

    batch = dict()
    batch[tids_key]  = []
    batch[sids_key]  = []
    batch[amask_key] = []
    batch[label_key] = []

    for data_dict in data_dict_batch:
        cur_len = len(data_dict[tids_key])
        data_dict[tids_key].extend([0] * (max_len - cur_len))
        data_dict[sids_key].extend([0] * (max_len - cur_len))
        data_dict[amask_key].extend([0] * (max_len - cur_len))
        batch[tids_key].append(data_dict[tids_key])
        batch[sids_key].append(data_dict[sids_key])
        batch[amask_key].append(data_dict[amask_key])

    ''' Tensorize '''
    tensor_batch[tids_key]  = torch.LongTensor(batch[tids_key])
    tensor_batch[sids_key]  = torch.LongTensor(batch[sids_key])
    tensor_batch[amask_key] = torch.LongTensor(batch[amask_key])
    
    ''' Optional features'''
    if label_key in data_dict_batch[0]:
        batch[label_key]  = []
        for data_dict in data_dict_batch:
            batch[label_key].append(data_dict[label_key])
        tensor_batch[label_key] = torch.FloatTensor(batch[label_key])

    if labels_key in data_dict_batch[0]:
        batch[labels_key]  = []
        for data_dict in data_dict_batch:
            batch[labels_key].extend(data_dict[labels_key])
        tensor_batch[labels_key] = torch.LongTensor(batch[labels_key])

    if positions_key in data_dict_batch[0]:
        batch[positions_key]  = []
        for ix, data_dict in enumerate(data_dict_batch):
            for pos in data_dict[positions_key]:
                batch[positions_key].append(ix * max_len + pos)
        tensor_batch[positions_key] = torch.LongTensor(batch[positions_key])

    if locations_key in data_dict_batch[0]:
        tensor_batch[locations_key] = []
        for data_dict in data_dict_batch:
            tensor_batch[locations_key].append(data_dict[locations_key])
            