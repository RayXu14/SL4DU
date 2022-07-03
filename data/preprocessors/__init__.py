from data.preprocessors.classification import DailyProcessor, \
                                              SwDAProcessor
from data.preprocessors.response_selection import UbuntuProcessor, \
                                                  UbuntuGenRankProcessor, \
                                                  DailyRSProcessor, \
                                                  PersonaChatRSProcessor
from data.preprocessors.dialog_state_tracking import DSTC2Processor
        
        
def get_processor(args):
    if args.task == 'RS':
        if args.dataset in ['Ubuntu', 'Douban', 'E-commerce']:
            return UbuntuProcessor(args)
        elif args.dataset in ['Daily']:
            return DailyRSProcessor(args)
        elif args.dataset in ['PersonaChat']:
            return PersonaChatRSProcessor(args)
    if args.task == 'GR':
        if args.dataset in ['Ubuntu', 'Douban', 'E-commerce']:
            return UbuntuGenRankProcessor(args)
    elif args.task == 'CLS':
        if args.dataset in ['Daily']:
            return DailyProcessor(args)
        if args.dataset in ['SwDA']:
            return SwDAProcessor(args)
    elif args.task == 'DST':
        if args.dataset in ['DSTC2']:
            return DSTC2Processor(args)
    raise NotImplementedError('Not supported task-dataset combination.')
    