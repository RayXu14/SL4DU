def get_handler(args, mode):
    if args.task == 'CLS':
        from handler.classification import CLSHandler
        return CLSHandler(args, mode)
    elif args.task == 'RS':
        from handler.response_selection import RSHandler
        return RSHandler(args, mode)
    raise NotImplementedError('Not supported task.')
    