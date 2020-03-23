import numpy as np
from utils import isscalar, isstruct, scalar_uniform_dist

class GenericRandom:
    def __init__(self, dtype, scalar_dist=None):
        self.dtype = np.dtype(dtype)
        self.scalar_dist = scalar_dist or scalar_uniform_dist

    def __call__(self, params, shape=None):
        if isstruct(self.dtype):
            r = np.empty(shape or 1, dtype=self.dtype)
            for name, dtypename in self.dtype.descr:
                try:
                    param = params[name]
                except TypeError:
                    param = params
                g = GenericRandom(dtypename)
                r[name] = g(param, shape)
            return r if shape else r[0]
        elif isscalar(self.dtype):
            return self.scalar_dist(self.dtype, params, shape)
        else:
            raise NotImplementedError(self.dtype)

    def __repr__(self):
        return 'GenericRandom({})'.format(self.dtype)

struct = np.dtype([('f0', np.float32), ('f1', np.int64), ('f2', np.longlong)])
struct_limit = dict(f0=17, f1=128, f2=42)

struct2 = np.dtype([('f0', struct), ('f1', np.uint16), ('f2', struct)])
struct2_limit = dict(f0=struct_limit, f1=42, f2=struct_limit)

g = GenericRandom(struct)
r = g(struct_limit, (5, 2))

g2 = GenericRandom(struct2)
r2 = g2(struct2_limit, (5, 2))

