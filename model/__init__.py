def get_model(args, new_tokenizer_len):
    if args.task == 'CLS':
        from model.classification import ClassificationModel
        return ClassificationModel(args, new_tokenizer_len)
    elif args.task == 'RS':
        from model.crmatching import CRMatchingModel
        return CRMatchingModel(args, new_tokenizer_len)
    raise NotImplementedError('Not supported task.')
    