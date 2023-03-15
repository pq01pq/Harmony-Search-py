import os
import time
import numpy as np
from harmony import *

from typing import Iterable


def rosenbrock(x: Iterable) -> float:
    return 100 * (x[0]**2 - x[1])**2 + (1 - x[0])**2


def wave(x: Iterable) -> float:
    return np.cos(x[0]) + np.cos(x[1])


# def constraint(var):
#     return abs(var[0]) <= 2.048 and abs(var[1]) <= 2.048


obj_func = wave

low, high = 0, 4 * np.pi
discrete_domain = split_interval(low, high, 10000)
continuous_domain = [low, high]

n_test = 10

mem_size = 100
epoch = 10000
hmcr = 0.9
par = 0.2
seed = None

multiple = 1
multiple = False if multiple == 0 else True

domain = [
    continuous_domain,
    continuous_domain
]

for i_test in range(n_test):
    t0 = time.time()
    harmony = ContinuousHarmonizer(domain=domain, mem_size=mem_size, obj_func=obj_func)
    harmony.set(hmcr=hmcr, par=par, epoch=epoch, seed=seed)
    solution, min_costs = harmony.multiple_search() if multiple else harmony.search()
    t1 = time.time()
    print(f'Δt : {(t1 - t0) * 1000:.3f} ms')
    print(f'solution :\n{solution}')
    if multiple:
        print(f'cost :\n'
              f'{np.array([obj_func(solution[i_sol]) for i_sol in range(len(solution))]).reshape(-1, 1)}')
    else:
        print(f'cost :\n{obj_func(solution)}')
    print()
print()
print('=================================================')
print()
print()
domain = [
    discrete_domain,
    discrete_domain
]
for i_test in range(n_test):
    t0 = time.time()
    harmony = DiscreteHarmonizer(domain=domain, mem_size=mem_size, obj_func=obj_func)
    harmony.set(hmcr=hmcr, par=par, epoch=epoch, seed=seed)
    solution, min_costs = harmony.multiple_search() if multiple else harmony.search()
    t1 = time.time()
    print(f'Δt : {(t1 - t0) * 1000:.3f} ms')
    print(f'solution :\n{solution}')
    if multiple:
        print(f'cost :\n'
              f'{np.array([obj_func(solution[i_sol]) for i_sol in range(len(solution))]).reshape(-1, 1)}')
    else:
        print(f'cost :\n{obj_func(solution)}')
    print()