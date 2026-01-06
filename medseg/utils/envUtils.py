import os

__all__ = ['resolve_to_environ']

def resolve_to_environ(config):
    os.environ['MEDSEG_MODEL'] = config.model.name
    os.environ['MEDSEG_DATA']  = config.data.name