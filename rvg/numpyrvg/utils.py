import numpy as np

def uniform_dist(dtype, params, shape):

    try:
        low, high = params
    except TypeError:
        low, high = -params, params

    if np.issubdtype(dtype, np.signedinteger):
        return np.random.randint(low, high, shape, dtype.type)
    if np.issubdtype(dtype, np.unsignedinteger):
        return np.random.randint(max(low, 0), high, shape, dtype.type)
    if np.issubdtype(dtype, np.floating):
        return np.random.uniform(low, high, shape)
    raise NotImplementedError(f'no known uniform distribution for dtype {dtype}')

def isstruct(dtype):
    return hasattr(dtype, 'names') and dtype.names

def issubarray(dtype):
    return hasattr(dtype, 'subdtype') and dtype.subdtype

def isscalar(dtype):
    return hasattr(dtype, 'names') and not dtype.names

def maybe_dict_get(maybe_dict, key):
    try:
        return maybe_dict.get(key)
    except AttributeError:
        return maybe_dict

def to_tuple(shape):
    try:
        return tuple(shape)
    except TypeError:
        return (shape,) if shape is not None else ()
