import numpy as np
from numpy.lib.recfunctions import _get_fieldspec as fields
from utils import isscalar, issubarray, isstruct, to_tuple, maybe_dict_get, uniform_dist

def random(dtype, params, shape=None, dist=None):
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
            return dist(dtype, params, shape)
        else:
            raise NotImplementedError(dtype)

    return gen(dtype, params, shape)
