from datetime import datetime
import os
import sys


class Redirect:
    def __init__(self, target, path='log'):
        self.terminal = target
        self.log = open(path, 'a')

    def write(self, message):
        if len(message.strip()) > 0:
            timestamp = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
            prefix = timestamp + ' | '
        else:
            prefix = ''
        self.terminal.write(prefix + message)
        self.log.write(prefix + message)

    def flush(self):
        pass


def redirect_std(path: str):
    sys.stdout = Redirect(sys.stdout, path + '.out')
    sys.stderr = Redirect(sys.stderr, path + '.err')

    
def auto_redirect_std(dir_path):
    path = os.path.join(dir_path, 'log')
    redirect_std(path)


def check_output_file(path):
    if os.path.exists(path):
        input(f"Output target [{path}] already exists. Cover it? >")


def check_output_dir(path: str):
    if os.path.exists(path) and os.listdir(path):
        print(f'Output directory [{path}] already exists and is not empty.')
        input('Will cover it. Continue? >')
        for p in os.listdir(path):
            try:
                print(f'Removing {path}/{p}...')
                os.remove(f'{path}/{p}')
                print('\t=> And succeeded.')
            except:
                print('\t=> But failed.')
                pass
    elif not os.path.exists(path):
        os.makedirs(path)
        print(f'Created {path}.')
    else:
        print(f'{path} exist but is just empty. Will write to it.')


def check_output_dirs(paths):
    for path in paths:
        check_output_dir(path)