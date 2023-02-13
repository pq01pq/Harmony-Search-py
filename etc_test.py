import os
import typing
import warnings
from sys import float_info
import math
from typing import Iterable
import random

import matplotlib
import numpy as np
import pandas as pd
from numpy.random import RandomState
import seaborn as sns
# from sklearn.model_selection import train_test_split
# from keras.models import Sequential
# from keras.layers.core import Dense
# from keras.layers import activation
# from keras import optimizers


# class NonTest:
#     def f(self, **kwagrs):
#         print(kwagrs.get('test1'))
#         print(kwagrs.get('test2'))

print(__name__)


def dimension_of(domain, n_dim: int = 0):
    if not isinstance(domain, Iterable):
        return n_dim
    elif len(domain) == 0:
        return n_dim + 1

    max_sub_dim = 0
    for sub_domain in domain:
        sub_dim = dimension_of(sub_domain, n_dim + 1)
        if max_sub_dim < sub_dim:
            max_sub_dim = sub_dim
    return max_sub_dim

print(dimension_of([
    np.linspace(0, 1, 10)
]))

print(issubclass(np.ndarray, Iterable))
# seq = Sequential()
# lay = Dense()
# optimizers.Adam()
# seq.compile(keras.metrics)

from seaborn.palettes import SEABORN_PALETTES, QUAL_PALETTES, get_colormap


