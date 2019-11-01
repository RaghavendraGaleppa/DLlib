''' The init file may contain confings and stuff '''
import os
import sys
sys.path.append(os.path.dirname(__file__))

import utils
from .train import train
