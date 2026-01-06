import os
import sys
import time
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger('MedSeg')
logger.setLevel(logging.INFO)

medseg_home = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(medseg_home)
os.environ['MEDSEG_HOME'] = medseg_home
os.environ['MEDSEG_MODEL'] = 'Default'
os.environ['MEDSEG_DATA'] = 'Default'
os.environ['MEDSEG_TIME'] = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())

from . import models
from . import utils
from . import data
from . import loss

__all__ = ['models', 'utils', 'data', 'loss', 'logger'] 