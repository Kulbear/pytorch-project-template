import os
import json
import argparse
from bunch import Bunch


def read_json_config(f_path):
    """ Read configuration from the given json file.

    Parameters
    ----------
    f_path : str
        The path of the configuration file, must have a ".json" postfix.

    Returns
    -------
    config : bunch dict
        A config file of Bunch dictionary format that supports 
        attribute-style access.
    """

    with open(f_path, 'r') as f:
        config = Bunch(json.load(f))

    return config


def mkdir(dir):
    """ Similar to "mkdir" in bash.
    
    Create a directory with path 'dir' if it does not exist.
     """
    if not os.path.exists(dir):
        os.makedirs(dir)

def get_args():
    """ Create an argument parser. """
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c',
        '--config',
        metavar='C',
        default='None',
        help='The path to the configuration file')
    args = argparser.parse_args()
    return args