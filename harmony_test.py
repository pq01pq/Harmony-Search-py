import os
import time
import numpy as np
from harmony import *


def rosenbrock(var):
    return 100 * (var[0]**2 - var[1])**2 + (1 - var[0])**2


def wave(var):
    return np.cos(var[0]) + np.cos(var[1])


# def constraint(var):
#     return abs(var[0]) <= 2.048 and abs(var[1]) <= 2.048


obj_func = wave

low, high = 0, 4 * np.pi
discrete_domain = split_interval(low, high, 10000)
continuous_domain = [low, high]

n_test = 10

mem_size = 100
n_iter = 10000
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
    harmony = ContinuousHarmonizer()
    harmony.set(domain=domain, mem_size=mem_size, obj_func=obj_func,
                hmcr=hmcr, par=par, n_iter=n_iter, seed=seed)
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
    harmony = DiscreteHarmonizer()
    harmony.set(domain=domain, mem_size=mem_size, obj_func=obj_func,
                hmcr=hmcr, par=par, n_iter=n_iter, seed=seed)
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