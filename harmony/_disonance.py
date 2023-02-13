class MemoryException(Exception):
    def __init__(self):
        pass

    def __str__(self):
        return 'Size of memory is smaller than 1.'


class DomainException(Exception):
    def __init__(self):
        pass

    def __str__(self):
        pass


class DimensionException(DomainException):
    def __init__(self, assigned_dim):
        self.__assigned_dim = assigned_dim

    def __str__(self):
        return f'The dimension of domain must be 2D, but {self.__assigned_dim}D is given.'


class DegreeException(DomainException):
    def __init__(self):
        pass

    def __str__(self):
        return 'The degree of domain must larger than 0.'
