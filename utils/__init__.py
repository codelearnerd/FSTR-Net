
"""
utils/initialization
"""


def notebook_init():

    print('Checking setup...')
    from IPython import display

    from utils.general import emojis
    from utils.torch_utils import select_device

    display.clear_output()
    select_device(newline=False)
    print(emojis('Setup complete '))
    return display
