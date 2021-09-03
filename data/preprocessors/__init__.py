from data.preprocessors.classification import DailyProcessor
from data.preprocessors.response_selection import UbuntuProcessor, \
                                                  DailyRSProcessor, \
                                                  PersonaChatRSProcessor
        
        
def get_processor(args):
    if args.task == 'RS':
        if args.dataset in ['Ubuntu', 'Douban', 'E-commerce']:
            return UbuntuProcessor(args)
        elif args.dataset in ['Daily']:
            return DailyRSProcessor(args)
        elif args.dataset in ['PersonaChat']:
            return PersonaChatRSProcessor(args)
    elif args.task == 'CLS':
        if args.dataset in ['Daily']:
            return DailyProcessor(args)
    raise NotImplementedError('Not supported task-dataset combination.')
    