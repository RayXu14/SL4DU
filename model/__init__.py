def get_model(args, new_tokenizer_len):
    if args.task == 'CLS':
        from model.classification import ClassificationModel
        return ClassificationModel(args, new_tokenizer_len)
    elif args.task == 'RS' or args.task == 'G2R':
        from model.crmatching import CRMatchingModel
        return CRMatchingModel(args, new_tokenizer_len)
    elif args.task == 'RG':
        from model.response_generation import ResponseGenerationModel
        return ResponseGenerationModel(args, new_tokenizer_len)
    raise NotImplementedError('Not supported task.')
    