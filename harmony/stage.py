from __future__ import annotations

import pandas as pd

from ._chamber import (
    # Pre-imported libraries
    Callable,
    np,

    # Classes
    _Harmonizer

    # Functions
)


class ContinuousHarmonizer(_Harmonizer):
    def __init__(self, domain: list[list] = None, mem_size: int = 0,
                 obj_func: Callable[[list], float] = None,
                 constraint_func: Callable[[list], bool] = None, **kwargs):
        super().__init__(obj_func=obj_func, constraint_func=constraint_func, **kwargs)

        self.__memory = None

        self.__init(domain=domain, mem_size=mem_size)

    def __init(self, domain, mem_size):
        self.__memory = self.Memory(domain=self.Domain(domain=domain), size=mem_size)

        if self.__memory.size < 1 or self.domain.n_var < 1 or self.obj_func is None:
            return

        # Randomized composition of variables
        i_mem = 0
        while i_mem < self.__memory.size:
            new_members = np.array(
                [self._random.uniform(self.domain[i_var, 0], self.domain[i_var, 1])
                 for i_var in range(self.domain.n_var)],
                dtype=self.Memory.dtype)

            # Check constraint (optional)
            if self._violate_constraint(new_members):
                continue

            self.__memory[i_mem] = new_members
            self.__memory.cost[i_mem] = self._calc_cost(new_members)
            i_mem += 1

        # Sort in order by cost : O(lg(hm_size))
        self.__memory.sort(maximize=self.maximize)

    def set(self, hmcr: float = None, par: float = None, n_iter: int = None,
            seed: int = None, maximize: bool = None, **kwargs):
        super().set(hmcr=hmcr, par=par, n_iter=n_iter, seed=seed, maximize=maximize, **kwargs)

    def _calc_cost(self, members):
        return self.obj_func(members)

    def _violate_constraint(self, members):
        return self.constraint_func is not None and self.constraint_func(members)

    # Randomized new composition of variables
    def _select_members(self):
        new_vars = []
        for i_var in range(self.domain.n_var):  # θ(n_var) : Not considerable
            # HarmonySearch memory consideration
            if self._random.uniform() < self.hmcr:
                random_value = self.__memory[self._random.randint(0, self.__memory.size), i_var]
            else:
                random_value = self._random.uniform(self.domain[i_var, 0], self.domain[i_var, 1])
            # Pitch adjusting (optional)
            if self.par is not None and self._random.uniform() < self.par:
                bandwidth = self.__get_bandwidth(i_var)

                increased_value = random_value + bandwidth
                decreased_value = random_value - bandwidth
                # If adjusted value get out of domain, adjust opposite way
                if self._random.uniform() < 0.5:
                    # Usually decreased
                    random_value = increased_value if decreased_value < self.domain[i_var, 0] \
                        else decreased_value
                else:
                    # Usually increased
                    random_value = decreased_value if increased_value > self.domain[i_var, 1] \
                        else increased_value

            new_vars.append(random_value)

        return new_vars

    def __get_bandwidth(self, var_idx):
        min_delta = np.finfo(np.float32).max
        for i_mem in range(1, self.__memory.size):
            cur_delta = abs(self.__memory[i_mem, var_idx] - self.__memory[i_mem, 0])
            if cur_delta < min_delta:
                min_delta = cur_delta

        return min_delta / 2

    @property
    def memory(self):
        return self.__memory

    @property
    def domain(self):
        return self.__memory.domain

    def __getitem__(self, *args):
        if isinstance(args[0], int):
            return self.__memory[args[0]]
        indices = args[0]
        if indices[1] == self.__memory.n_var:
            return self.__memory.cost[indices[1]]
        return self.__memory[indices[0], indices[1]]

    def __str__(self):
        return str(
            pd.DataFrame(
                data=np.append(self.__memory, self.__memory.cost, axis=1),
                columns=[f'x{i + 1}' for i in range(self.domain.n_var)] + ['cost']
            )
        )

    class Memory(_Harmonizer.Memory):
        __dtype = np.float32

        @classmethod
        @property
        def dtype(cls):
            return cls.__dtype

        def __init__(self, domain: ContinuousHarmonizer.Domain, size: int):
            super().__init__(size=size, n_var=domain.n_var, dtype=self.__dtype)
            self.__domain = domain

        @property
        def domain(self):
            return self.__domain

    class Domain(_Harmonizer.Domain):
        def __init__(self, domain):
            super().__init__(domain=domain)


class DiscreteHarmonizer(_Harmonizer):
    def __init__(self, domain: list[list] = None, mem_size: int = 50,
                 obj_func: Callable[[list], float] = None,
                 constraint_func: Callable[[list], bool] = None, **kwargs):
        super().__init__(obj_func=obj_func, constraint_func=constraint_func, **kwargs)
        self.__memory = None

        self.__init(domain=domain, mem_size=mem_size)

    def __init(self, domain, mem_size):
        self.__memory = self.Memory(domain=self.Domain(domain=domain), size=mem_size)

        if self.__memory.size < 1 or self.domain.n_var < 1 or self.obj_func is None:
            return

        # Randomized composition of variables
        i_mem = 0
        while i_mem < self.__memory.size:
            new_var_indices = np.array(
                [self._random.randint(0, self.domain.length(i_var)) for i_var in range(self.domain.n_var)],
                dtype=self.Memory.dtype)

            # Check constraint (optional)
            if self._violate_constraint(new_var_indices):
                continue

            self.__memory[i_mem] = new_var_indices
            self.__memory.cost[i_mem] = self._calc_cost(new_var_indices)
            i_mem += 1

        # Sort in order by cost : O(lg(hm_size))
        self.__memory.sort(maximize=self.maximize)

    def set(self, hmcr: float = None, par: float = None, n_iter: int = None,
            seed: int = None, maximize: bool = None, **kwargs):
        super().set(hmcr=hmcr, par=par, n_iter=n_iter, seed=seed, maximize=maximize, **kwargs)

    def _calc_cost(self, members):
        return self.obj_func(self.__memory.vars_by_indices(members))

    def _violate_constraint(self, members):
        return self.constraint_func is not None and not self.constraint_func(self.__memory.vars_by_indices(members))

    # Randomized new composition of variables
    def _select_members(self):
        new_var_indices = []
        for i_var in range(self.domain.n_var):  # θ(n_var) : Not considerable
            # HarmonySearch memory consideration
            if self._random.uniform() < self.hmcr:
                random_index = self.__memory[self._random.randint(0, self.__memory.size), i_var]
            else:
                random_index = self._random.randint(0, self.domain.length(i_var))

            # Pitch adjusting (optional)
            if self.par is not None and self._random.uniform() < self.par:
                increased_index = random_index + 1
                decreased_index = random_index - 1
                # If adjusted value get out of domain, adjust opposite way
                if self._random.uniform() < 0.5:
                    # Usually decreased
                    random_index = increased_index if decreased_index < 0 \
                        else decreased_index
                else:
                    # Usually increased
                    random_index = decreased_index if increased_index >= self.domain.length(i_var) \
                        else increased_index

            new_var_indices.append(random_index)

        return new_var_indices

    @property
    def memory(self):
        return self.__memory

    @property
    def domain(self):
        return self.__memory.domain

    def __getitem__(self, *args):
        if isinstance(args[0], int):
            return self.__memory.vars_by_indices(self.__memory[args[0]])
        indices = args[0]
        if indices[1] == self.__memory.n_var:
            return self.__memory.cost[indices[1]]
        return self.__memory.var_by_index(self.__memory[indices[0], indices[1]])

    def __str__(self):
        var_memory = np.array(
            [self.__memory.vars_by_indices(self.__memory[i_mem]) for i_mem in range(self.__memory.size)],
            dtype=ContinuousHarmonizer.Memory.dtype
        )
        cost_memory = self.__memory.cost.reshape(-1, 1)

        return str(
            pd.DataFrame(
                data=np.append(var_memory, cost_memory, axis=1),
                columns=[f'x{i + 1}' for i in range(self.domain.n_var)] + ['cost']
            )
        )

    class Memory(_Harmonizer.Memory):
        __dtype = np.int32

        @classmethod
        @property
        def dtype(cls):
            return cls.__dtype

        def __init__(self, domain: DiscreteHarmonizer.Domain, size: int):
            super().__init__(size=size, n_var=domain.n_var, dtype=self.__dtype)
            self.__domain = domain

        def vars_by_indices(self, domain_indices: list | np.ndarray):
            return np.array(
                [self.__domain[i_var, domain_indices[i_var]] for i_var in range(self.__domain.n_var)],
                dtype=ContinuousHarmonizer.Memory.dtype)

        def var_by_index(self, var_idx, domain_idx):
            return self.__domain[var_idx, domain_idx]

        @property
        def domain(self):
            return self.__domain

    class Domain(_Harmonizer.Domain):
        def __init__(self, domain):
            super().__init__(domain=domain)
            self.__domain_lengths = [len(self[i_var]) for i_var in range(self.n_var)]

        @property
        def lengths(self):
            return self.__domain_lengths

        def length(self, i_var):
            return self.__domain_lengths[i_var]


def split_interval(low, high, n_interval):
    return np.linspace(low, high, n_interval + 1)
