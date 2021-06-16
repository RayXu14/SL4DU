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
            pass
    return cuda_batch
    
    
def tensor2np(tensor):
    return tensor.detach().cpu().numpy()


def tensor2list(tensor):
    return tensor2np(tensor).tolist()