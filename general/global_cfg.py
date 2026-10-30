import os
import yaml
from general.config import load_config


def _wsl_to_windows_path(datas_path):
    if os.name != 'nt':
        return None
    norm = str(datas_path).replace('\\', '/')
    prefix = '/mnt/'
    if not norm.lower().startswith(prefix) or len(norm) < len(prefix) + 2:
        return None
    drive = norm[len(prefix)]
    if norm[len(prefix) + 1] != '/':
        return None
    rest = norm[len(prefix) + 2:].replace('/', os.sep)
    return f'{drive.upper()}:\\{rest}'

def get_global_path(datas_path):
    home_path = os.path.expanduser("~")
    global_name = os.path.join(home_path,'global.yaml')
    if os.path.exists(global_name):
        with open(global_name, "r") as f:
            config = yaml.safe_load(f)
        if 'datas_path' in config:
            after_datas = datas_path.split("datas/", 1)[-1]
            return os.path.join(config['datas_path'],after_datas)
        else:
            print(f'\033[91m No datas_path in {global_name} \033[0m')
            return None
    else:
        return None

def replace_path(datas_path):
    if not os.path.exists(datas_path):
        print(f'\033[91m{datas_path}\033[0m', end="")
        global_path = get_global_path(datas_path)
        if global_path and os.path.exists(global_path):
            print(f'-->\033[92m{global_path}\033[0m')
            return global_path

        win_path = _wsl_to_windows_path(datas_path)
        if win_path and os.path.exists(win_path):
            print(f'-->\033[92m{win_path}\033[0m')
            return win_path

        print(f'-->\033[91m{global_path or win_path}\033[0m')
        return datas_path
    else:
        return datas_path

def get_machine_info():
    home_path = os.path.expanduser("~")
    global_name = os.path.join(home_path,'global.yaml')
    if os.path.exists(global_name):
        with open(global_name, "r") as f:
            config = yaml.safe_load(f)
        if 'machine_info' in config:
            return config['machine_info']
        else:
            print(f'\033[91m No datas_path in machine_info \033[0m')
            return os.name
    else:
        return os.name
