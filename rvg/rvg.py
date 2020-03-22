import numpy as np
import warnings

class NumPyRVG:
    '''
    The NumPy Random Value Generator.
    Generates random scalars (if `length` is not given) or 1D arrays of a certain type.
    The type can be either a primitive one, a scalar or a struct.
    '''
    def __init__(self, a, b=None):

        # case were 1 integer was given
        if b is None:
            if a <= 0:
                raise ValueError('argument `limit` must be a number greater than 0')
            self.a, self.b = -a, a

        # case were 2 integers were given
        else:
            if a >= b:
                raise ValueError('argument `a` must be less than `b`')
            self.a, self.b = a, b
            if b < 0:
                warnings.warn(
                    f'value {b} for argument `b` will cause a runtime error if generation of values of unsigned type is attempted',
                    Warning,
                    stacklevel=2
                )

    def scalar_or_array(self, rvg, args, dtype, length):

        # scalar requested
        if not length:
            try:
                val = rvg(*args)
                return dtype(val)
            except TypeError:
                return val
            except ValueError:
                raise ValueError(f'Can not generate unsigned value with upper bound equal to {self.b}')

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
            return self.scalar_or_array(np.random.randint, (self.a, self.b), dtype, length)

        if np.issubdtype(dtype, np.unsignedinteger):
            return self.scalar_or_array(np.random.randint, (max(self.a, 0), self.b), dtype, length)

        if np.issubdtype(dtype, np.floating):
            return self.scalar_or_array(np.random.uniform, (self.a, self.b), dtype, length)

        # custom types, treated all the same way (i.e. as tuples)
        if hasattr(dtype, 'fields'):
            return self.scalar_or_array(self.rand_val_for_non_primitive_type, (dtype,), dtype, length)

        raise NotImplementedError('dtype ' + str(dtype) + ' is not supported')
