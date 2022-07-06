def get_handler(args, mode):
    if args.task == 'CLS':
        from handler.classification import CLSHandler
        return CLSHandler(args, mode)
    elif args.task == 'RS' or args.task == 'G2R':
        from handler.response_selection import RSHandler
        return RSHandler(args, mode)
    elif args.task == 'RG':
        from handler.response_generation import RGHandler
        return RGHandler(args, mode)
    raise NotImplementedError('Not supported task.')
    