import time
import logging

from functools import reduce
from collections import namedtuple, deque

import pandas as pd
from scipy.sparse import lil_matrix

logging.basicConfig(level=logging.INFO,
    format="%(levelname)s - %(asctime)s - %(msg)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("log_knapsack.log"), # For debug
        logging.StreamHandler(),
    ])
