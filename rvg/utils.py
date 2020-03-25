import numpy as np

def uniform_dist(dtype, params, shape):
    base_mapping = {
        np.unsignedinteger : lambda l, h, shape: np.random.randint(max(l, 0), h, shape),
        np.signedinteger   : np.random.randint,
        np.floating        : np.random.uniform
    }

    try:
       low, high = params
    except TypeError:
        low, high = -params, params

    return base_mapping[dtype.type.__base__](low, high, shape)

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
