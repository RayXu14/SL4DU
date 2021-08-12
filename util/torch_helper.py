from collections import OrderedDict


def visualize_model(model):
    model_spec = []
    for name, param in model.named_parameters():
        model_spec.append(f'Trainable={param.requires_grad}: {name}')
        model_spec.append(f'\t{param.type()}\t{list(param.size())}')
    model_spec = '\n'.join(model_spec)
    print(f'\n{model}\n{model_spec}\n')
    
    
def batch2cuda(batch):
    cuda_batch = {}
    for key in batch:
        try:
            cuda_batch[key] = batch[key].cuda()
        except: # There may be non-tensor data
            '''
            WARNING: Non-tensor data cannot be loaded by nn.DataParallel.
            '''
            cuda_batch[key] = batch[key]
            pass
    return cuda_batch
    
    
def tensor2np(tensor):
    return tensor.detach().cpu().numpy()


def tensor2list(tensor):
    return tensor2np(tensor).tolist()
    

def smart_model_loading(model, checkpoint):
    print('=== Smart Model Loading module reported. ===')
    state_dict = OrderedDict()
    for k, v in checkpoint['model_state_dict'].items():
        if k in model.state_dict():
            state_dict[k] = v
        else:
            print(f'SMART MODEL LOADING: {k} is ignored!')
    model.load_state_dict(state_dict)
    return model