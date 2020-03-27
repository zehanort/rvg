import numpy as np
from numpy.lib.recfunctions import _get_fieldspec as fields
from .utils import isscalar, issubarray, isstruct, to_tuple, maybe_dict_get, uniform_dist
import warnings

class NumPyRVG:
    '''
    The NumPy Random Value Generator.
    Generates random scalars (if `samples` is not given) or arrays of a certain type.
    The type can be either a primitive one, a scalar or a struct.
    '''
    def __init__(self, **kwargs):
        '''
        kwargs can contain exactly one of the following:
            dtype:  A dtype for the generated values
            limit:  An integer that will be used as the
                    numerical limits of the generated values (-limit, limit)
            limits: An iterable with two integers, a and b,
                    that will be used as the numerical limits of the generated values (a, b)
        '''

        kwargs_error_msg = 'exactly one of the following arguments is needed: `dtype`, `limit`, `limits`'

        if len(kwargs) != 1:
            raise AttributeError(kwargs_error_msg)

        dtype = kwargs.get('dtype')
        limit = kwargs.get('limit')
        limits = kwargs.get('limits')

        if all(x is None for x in [dtype, limit, limits]):
            raise AttributeError(kwargs_error_msg)

        ### 3 mutually exclusive cases follow (mutual exclusiveness has just been checked) ###

        # case 1: `limit` was given
        if limit:
            if not isinstance(limit, int):
                raise TypeError('argument `limit` must be an integer greater than 0')
            if limit <= 0:
                raise ValueError('argument `limit` must be a number greater than 0')
            self.a, self.b = -limit, limit

        # case 2: `limits` was given
        if limits:
            try:
                a, b = limits
            except (TypeError, ValueError) as e:
                raise type(e)('argument `limits` must be an iterable with exactly 2 integers')
            if a >= b:
                raise ValueError('the lower limit must be strictly less than the upper limit')
            self.a, self.b = a, b
            if b < 0 and dtype is None:
                warnings.warn(
                    'value ' + str(b) + ' as the upper limit will cause a runtime error if generation of values of unsigned type is attempted',
                    Warning,
                    stacklevel=2
                )

        self.dtype = dtype

    def __call__(self, arg=None, shape=None, dist=None, type_limits=True):
        if self.dtype is not None:
            if arg is None:
                raise TypeError('missing 1 required argument describing the limit(s)')
            return self.random(self.dtype, arg, shape, dist, type_limits)
        elif self.a is not None:
            if arg is None:
                raise TypeError('missing 1 required argument describing the dtype')
            return self.random(arg, (self.a, self.b), shape, dist, type_limits)
        raise NotImplementedError('this call can not be served')

    def random(self, dtype, params, shape=None, dist=None, type_limits=True):
        dist = dist or uniform_dist
        dtype = np.dtype(dtype)

        def gen(dtype, params, shape):
            if isstruct(dtype):
                r = np.empty(shape or 1, dtype=dtype)
                for field_name, field_dtype in fields(dtype):
                    field_params = maybe_dict_get(params, field_name)
                    r[field_name] = gen(field_dtype, field_params, shape)
                return r if shape else r[0]
            elif issubarray(dtype):
                item_dtype, sub_shape = dtype.subdtype
                field_shape = to_tuple(shape) + sub_shape
                return gen(item_dtype, params, field_shape)
            elif isscalar(dtype):
                try:
                    return dtype.type(dist(dtype, params, shape, type_limits))
                except TypeError:
                    return dtype.type(dist(dtype, params, shape))
            else:
                raise NotImplementedError(dtype)

        return gen(dtype, params, shape)
