from .ioUtils import *

def resolve_to_environ(config):
    import os
    os.environ['MEDSEG_MODEL'] = config.model.name
    os.environ['MEDSEG_DATA']  = config.data.name