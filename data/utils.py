import os

from transformers import AutoTokenizer
                         
                         
def init_tokenizer(args):
    path = os.path.join(args.pretrained_path, args.pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(path)
    print(f'Initialized tokenizer from {path}')
    if args.add_EOT:
        tokenizer.add_tokens(["[EOT]"])
    return tokenizer