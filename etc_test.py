import os
import typing
import warnings
from sys import float_info
import math
import random
import numpy as np
import pandas as pd
from numpy.random import RandomState
# from sklearn.model_selection import train_test_split
# from keras.models import Sequential
# from keras.layers.core import Dense
# from keras.layers import activation
# from keras import optimizers


# class NonTest:
#     def f(self, **kwagrs):
#         print(kwagrs.get('test1'))
#         print(kwagrs.get('test2'))


def __domain_dimension(domain, n_dim):
    if not hasattr(domain, '__iter__'):
        return n_dim

    max_sub_dim = 0
    for sub_domain in domain:
        sub_dim = __domain_dimension(sub_domain, n_dim + 1)
        if max_sub_dim < sub_dim:
            max_sub_dim = sub_dim
    return max_sub_dim


s = pd.Series([1, 2])
d = pd.DataFrame(
    data=[
        [1, 2, 3, np.NAN],
        [1, np.NAN, 2, 3]
    ],
    columns=['1', '2', '3', '4']
)

boool = pd.DataFrame(
    data=[
        [True, True, True, False],
        [True, True, False, False],
        [True, False, False, False],
    ],
    columns=['1', '2', '3', '4']
)
print(boool[['1', '2']])

# seq = Sequential()
# lay = Dense()
# optimizers.Adam()
# seq.compile(keras.metrics)

