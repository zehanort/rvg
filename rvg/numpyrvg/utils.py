import numpy as np

def uniform_dist(dtype, params, shape, type_limits):

    try:
        low, high = params
    except TypeError:
        low, high = -params, params

    if np.issubdtype(dtype, np.signedinteger):
        if type_limits:
            type_limits_info = np.iinfo(dtype)
            low = max(low, type_limits_info.min)
            high = min(high, type_limits_info.max)
        return np.random.randint(low, high, shape, dtype.type)
    if np.issubdtype(dtype, np.unsignedinteger):
        if type_limits:
            type_limits_info = np.iinfo(dtype)
            low = max(low, type_limits_info.min)
            high = min(high, type_limits_info.max)
        return np.random.randint(max(low, 0), high, shape, dtype.type)
    if np.issubdtype(dtype, np.floating):
        if type_limits:
            type_limits_info = np.finfo(dtype)
            low = max(low, type_limits_info.min)
            high = min(high, type_limits_info.max)
        return np.random.uniform(low, high, shape)
    raise NotImplementedError('no known uniform distribution for dtype ' + dtype)

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
