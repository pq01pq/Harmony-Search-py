import os
import warnings
from sys import float_info
import math
import random
import numpy as np
from numpy.random import RandomState


# class NonTest:
#     def f(self, **kwagrs):
#         print(kwagrs.get('test1'))
#         print(kwagrs.get('test2'))

class Duple(tuple):

    _dtype = tuple

    def __init__(self, *args):
        super().__init__(self=args)


def f(i, j=1, **kwargs):
    print(kwargs)
    print(*kwargs)
    print(type(**kwargs))


class Kwarg:
    def __init__(self, a=1, **kwargs):
        self.a = a
        self.b = kwargs['b']

    def __getitem__(self, *args, **kwargs):
        print(args)
        print(kwargs)

    def __setitem__(self, *args, **kwargs):
        print(args)
        print(kwargs)


k = Kwarg(b=1, a=10)
print(k.a, k.b)
a = k[1, 2, 3]
a = k[:, :, :]
k[1, 2, 3] = 4
k[:, :, :] = 4
a = k[(1, 2)]

# class MyRand:
#     def __init__(self, seed):
#         self.__seed = seed
#         self.__rand = RandomState(seed)
#
#     def rand(self):
#         return self.__rand.rand()
#
#     @property
#     def seed(self):
#         return self.__seed
#
#     @seed.setter
#     def seed(self, seed):
#         self.__seed = seed
#         self.__rand.seed(seed)

# d = {'1': 1}
# print(d.get('2'))
# print(d['2'])

# li1 = [1.0, 2]
# li2 = [3, 4]
# li3 = [5, 6]
# li = np.vstack([li1, li2, li3])
# print(np.array([_li for _li in li[0:2, 0]]))
# print(issubclass(np.int_, int))

# arr = np.array([[1, 2], [3, 4]])
# arr.__setitem__((0, 0), 4, 2)
# print(arr)

# m = np.array([
#     [1 for _ in range(2)] for _ in range(3)
# ])
# print(m[[0, 0]])

# tup = (
#     (1, 2),
#     (3, 4)
# )
