import numpy as np
def gen_scalar(dtype, limit):
    if np.issubdtype(dtype, np.floating):
        return np.random.uniform(*limit)
    elif np.issubdtype(dtype, np.unsignedinteger):
        return np.random.randint(0, limit)
    elif np.issubdtype(dtype, np.signedinteger):
        return np.random.randint(*limit)
    else:
        raise NotImplementedError('oops')

def gen_struct(dtype, limit):
    if dtype.fields is None:
        return gen_scalar(dtype, limit)
    else:
        r = tuple(gen_struct(np.dtype(dtypename), limit[name]) for name, dtypename in dtype.descr)
        return np.array([r], dtype=dtype)[0]

struct = np.dtype([('f0', np.float32), ('f1', np.int64), ('f2', np.longlong)])
struct2 = np.dtype([('f0', struct), ('f1', np.uint16), ('f2', struct)])
struct_limit = dict(f0=(-25, 25), f1=(-64, 64), f2=(-18, 17))
struct2_limit = dict(f0=struct_limit, f1=42, f2=struct_limit)

rand = gen_struct(struct, struct_limit)
rand2 = gen_struct(struct2, struct2_limit)

