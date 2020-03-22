import numpy as np

class NumPyRVG:
    '''
    The NumPy Random Value Generator.
    Generates random scalars (if `length` is not given) or 1D arrays of a certain type.
    The type can be either a primitive one, a scalar or a struct.
    '''
    def __init__(self, limit):
        if limit <= 0:
            raise ValueError('argument `limit` must be a number greater than 0')
        self.limit = limit

    def scalar_or_array(self, rvg, args, dtype, length):

        is_primitive_type = len(args) == 2

        if is_primitive_type:
            args = (args[0] * self.limit, args[1] * self.limit)

        # scalar requested
        if not length:
            val = rvg(*args)
            try:
                return dtype(val)
            except TypeError:
                return val

        # array requested
        vals = [rvg(*args) for _ in range(length)]
        try:
            return np.array(map(dtype, vals), dtype=dtype)
        except TypeError:
            return np.array(vals, dtype=dtype)

    def rand_val_for_non_primitive_type(self, dtype):
        retvals = []
        for field_name in dtype.names:
            field_dtype = dtype[field_name] if hasattr(dtype[field_name], 'names') and dtype[field_name].names else dtype[field_name].type
            retvals.append(self(field_dtype))
        return tuple(retvals)

    def __call__(self, dtype, length=0):

        if length < 0:
            raise ValueError('argument `length` must be a number greater or equal to 0')

        # primitive types
        if np.issubdtype(dtype, np.signedinteger):
            return self.scalar_or_array(np.random.randint, (-1, 1), dtype, length)

        if np.issubdtype(dtype, np.unsignedinteger):
            return self.scalar_or_array(np.random.randint, (0, 1), dtype, length)

        if np.issubdtype(dtype, np.floating):
            return self.scalar_or_array(np.random.uniform, (-1, 1), dtype, length)

        # custom types, treated all the same way (i.e. as tuples)
        if hasattr(dtype, 'fields'):
            return self.scalar_or_array(self.rand_val_for_non_primitive_type, (dtype,), dtype, length)

        if dtype == np.void:
            raise TypeError('HOLY SHIT')

        raise NotImplementedError('dtype ' + str(dtype) + ' is not supported')
