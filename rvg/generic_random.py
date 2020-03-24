import numpy as np
from numpy.lib.recfunctions import _get_fieldspec as fields
from utils import isscalar, isstruct, issubarray, scalar_uniform_dist, to_tuple

class NumPyRVG:
    '''
    Examples:

    Define a struct with 3 scalar fields
    >>> simple_struct = np.dtype([
        ('f0', np.float32),
        ('f1', np.int64),
        ('f2', np.longlong)
    ])

    Define distribution parameters for each field as a dictionary tree
        that mirrors `simple_struct`'s structure
    >>> simple_struct_param = {
        'f0' : 17,
        'f1' : 128,
        'f2' : 42
    }

    Create a generator for our `simple_struct`
    >>> simple_struct_gen = NumPyRVG(simple_struct)

    Use it with parameters and `shape=(5, 2)` to generate
        an `np.ndarray` of `simple_struct`s
    >>> simple_struct_gen(simple_struct_param, (5, 2)))
    [[(-16.68794  ,  85,  34) (-11.467181 ,  46, -27)]
     [(  9.669502 , -54, -26) ( 13.273692 ,  12,  36)]
     [( -8.88329  ,  43, -16) (  5.022718 ,  44,   8)]
     [(-12.742326 , -24,   0) ( 14.89258  , -65,  37)]
     [(  5.4119368, -25,  -2) (  8.193446 , -23, -12)]]

    Or omit `shape` to get a single `simple_struct` scalar back
    >>> simple_struct_gen(simple_struct_param)
    (-3.3626056, 115, -7)

    Also works for nested structs
    >>> nested_struct = np.dtype([
    ...     ('f0', simple_struct),
    ...     ('f1', np.uint16),
    ...     ('f2', simple_struct)
    ... ])

    We can leave out the details if we want the whole subtree of a field
        to use the same parameters (note `f0`)
    >>> nested_struct_param = {
    ...     'f0' : (-17, 42),
    ...     'f1' : 42,
    ...     'f2' : simple_struct_param
    ... }
    >>> nested_struct_gen = NumPyRVG(nested_struct)
    >>> nested_struct_gen(nested_struct_param, (5, 2))
    [[((21.17895 , -12,  18), 25, ( -2.0230212, -121,  18))
      ((20.165178,  15,  36), 28, (-15.15129  ,   14,  16))]
     [((-4.782298,  32,  -5),  9, (  6.565033 , -102, -30))
      ((24.694878,  -9,  34), 14, ( 12.588115 ,    0,  34))]
     [((30.890104,  36,  39), 32, ( 14.66334  ,  -85,  20))
      ((41.150288,  30,  -6), 24, ( 16.451141 ,   20,  -1))]
     [((35.9266  , -14,  21), 37, ( 16.783556 , -111, -31))
      ((35.711254,  32, -15), 41, (-13.199838 ,   38, -37))]
     [((41.072285,  16, -10), 31, (-11.53087  ,   -1, -20))
      ((38.53658 ,  -2,   3), 36, ( -0.7612021,  102,  16))]]

    Array members are also supported!
    >>> struct_array_member = np.dtype([
    ...     ('i', np.int8),
    ...     ('a', (simple_struct, 3))
    ... ])
    >>> struct_array_member_param = {
    ...     'i' : 42,
    ...     'a' : simple_struct_param
    ... }
    >>> struct_array_member_gen = NumPyRVG(struct_array_member)
    >>> struct_array_member_gen(struct_array_member_param, (3, 2))
    [[( 1, [(-14.027751  ,    1,  11), ( -0.60646707,  -14, -37), (-16.643078  ,  103,  23)])
      (-2, [( -0.20951761,  -50,  40), ( -6.1637464 ,  -51,  28), (-14.376863  ,   -8, -17)])]
     [(25, [( -7.594184  ,   48,  32), (  1.2456906 ,   10, -32), ( 11.486396  ,  -11, -17)])
      (36, [(  4.3206005 , -108,   0), (  7.8589587 ,  112, -14), ( -5.4548306 , -112,  18)])]
     [(12, [( -2.302605  ,  -21,  27), ( -0.9145098 ,   60,  -1), (-10.5987215 ,  -17,  -5)])
      (-3, [(-10.276416  , -122, -21), ( -5.304906  ,  -89,  -6), (-11.810806  ,  106, -14)])]]
      '''

    def __init__(self, dtype, scalar_dist=None):
        self.dtype = np.dtype(dtype)
        self.scalar_dist = scalar_dist or scalar_uniform_dist

    def __call__(self, params, shape=None):
        if isstruct(self.dtype):
            r = np.empty(shape or 1, dtype=self.dtype)
            for name, dtype in fields(self.dtype):
                try:
                    param = params[name]
                except TypeError:
                    param = params
                g = NumPyRVG(dtype)
                r[name] = g(param, shape)
            return r if shape else r[0]
        elif issubarray(self.dtype):
            item_dtype, sub_shape = self.dtype.subdtype
            g = NumPyRVG(item_dtype)
            return g(params, to_tuple(shape) + sub_shape)
        elif isscalar(self.dtype):
            return self.scalar_dist(self.dtype, params, shape)
        else:
            raise NotImplementedError(self.dtype)

    def __repr__(self):
        return 'NumPyRVG({})'.format(self.dtype)

