import os


def mkdir(dir):
    """ Similar to "mkdir" in bash.
    
    Create a directory with path 'dir' if it does not exist.
     """
    if not os.path.exists(dir):
        os.makedirs(dir)