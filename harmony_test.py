import os
import time
import numpy as np
from harmony.stage import *


def rosenbrock(var):
    return 100 * (var[0]**2 - var[1])**2 + (1 - var[0])**2


def wave(var):
    return np.cos(var[0]) + np.cos(var[1])

# def constraint(var):
#     return abs(var[0]) <= 2.048 and abs(var[1]) <= 2.048


cur_dir = os.getcwd()

obj_func = wave
mem_sizes = [25, 50, 100]
n_iters = [1000, 10000]
# n_interval = 10000
hmcrs = [0.5, 0.75, 0.9, 0.95]
pars = [0.2, 0.4, 0.6, 0.8]

low, high = 0, 4 * np.pi
discrete_domain = split_interval(low, high, 10000)
continuous_domain = [low, high]

n_test = 10

mem_size = 100
n_iter = 10000
seed = None

discrete = 0
multiple = 1
discrete = False if discrete == 0 else True
multiple = False if multiple == 0 else True

# print(np.isclose(3.141935, 3.141816, rtol=np.finfo(np.float32).eps * 1000))

print(f'HMCR test : {hmcrs}')
for i_test in range(len(hmcrs)):
    hmcr = hmcrs[i_test]
    print('-------------------------------------------------')
    print(f'HMCR : {hmcr}')
    print('-------------------------------------------------')
    if discrete:
        domain = [
            discrete_domain,
            discrete_domain
        ]
    else:
        domain = [
            continuous_domain,
            continuous_domain
        ]
    for _ in range(n_test):
        t0 = time.time()
        harmony = DiscreteHarmonizer() if discrete else ContinuousHarmonizer()
        harmony.set(domain=domain, mem_size=mem_size, obj_func=obj_func,
                    hmcr=hmcr, n_iter=n_iter, seed=seed)
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
print(f'PAR test : {pars}')
for i_test in range(len(pars)):
    par = pars[i_test]
    print('-------------------------------------------------')
    print(f'PAR : {par}')
    print('-------------------------------------------------')
    if discrete:
        domain = [
            discrete_domain,
            discrete_domain
        ]
    else:
        domain = [
            continuous_domain,
            continuous_domain
        ]
    for _ in range(n_test):
        t0 = time.time()
        harmony = DiscreteHarmonizer() if discrete else ContinuousHarmonizer()
        harmony.set(domain=domain, mem_size=mem_size, obj_func=obj_func,
                    par=par, n_iter=n_iter, seed=seed)
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
