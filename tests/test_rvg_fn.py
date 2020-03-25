import pytest
import numpy as np
import rvg

def struct(t, dtype):
    return np.array([t], dtype=dtype)[0]

def equal(a, b):
    print(a)
    print(b)
    return (a == b).all()

# Set seed for reproducibility
np.random.seed(42)

#  Define a struct with 3 scalar fields
simple_struct = np.dtype([
    ('f0', np.float32),
    ('f1', np.int64),
    ('f2', np.longlong)
])

#  Define distribution parameters for each field as a dictionary tree
#      that mirrors `simple_struct`'s structure
simple_struct_param = {
    'f0' : 17,
    'f1' : 128,
    'f2' : 42
}

#  Use it with parameters and `shape=(5, 2)` to generate
#      an `np.ndarray` of `simple_struct`s
def test_simple_struct_array():
    random = rvg.random(simple_struct, simple_struct_param, (5, 2))
    expected = np.array([
        [( -4.265636 ,   21, -22), ( 15.324286 ,  -76, -10)],
        [(  7.887794 , -127,  33), (  3.3543885,  -41,  15)],
        [(-11.695366 ,  107, -21), (-11.696186 ,   29,   6)],
        [(-15.025157 ,  -91,  16), ( 12.449989 ,    1,  -1)],
        [(  3.4379103,   63,  17), (  7.0744677,   59,  37)]
    ], dtype=np.dtype([('f0', '<f4'), ('f1', '<i8'), ('f2', '<i8')]))

    assert(equal(random, expected))

#  Or omit `shape` to get a single `simple_struct` scalar back
def test_simple_struct_single():
    random = rvg.random(simple_struct, simple_struct_param)
    expected = struct(
            (-9.085774, 61, 4),
            dtype=[('f0', '<f4'), ('f1', '<i8'), ('f2', '<i8')])

    assert(equal(random, expected))

#  Also works for nested structs
nested_struct = np.dtype([
    ('f0', simple_struct),
    ('f1', np.uint16),
    ('f2', simple_struct)
])

#  We can leave out the details if we want the whole subtree of a field
#      to use the same parameters (note `f0`)
nested_struct_param = {
    'f0' : (-17, 42),
    'f1' : 42,
    'f2' : simple_struct_param
}

def test_nestest_struct():
    random   = rvg.random(nested_struct, nested_struct_param, (5, 2))
    expected = np.array([
        [(( 19.484774 ,  40, -10),  5, ( 7.7272477 ,   59, -34)),
         ((  5.5652575,  -9,  29), 41, (-5.897614  ,   -5, -42))],
        [(( 41.010624 ,   8,  17),  3, ( 2.395095  ,  108, -35)),
         (( 10.539011 ,  35,  -4), 28, ( 0.70836484,  -88,  20))],
        [(( 33.736485 , -16,  -1), 17, (15.679849  ,   28, -32)),
         (( 23.138145 ,   2,  18), 25, (11.71415   , -114,  38))],
        [((  9.579456 ,  10,  32), 33, ( 8.408884  ,  -84, -35)),
         ((-16.217367 ,  29,  22),  9, ( 1.3495325 ,  -64,  -8))],
        [(( 38.589905 , -11, -14), 35, ( 2.9495397 ,  -40,  -8)),
         (( 16.234005 ,  26, -16), 13, (15.818681  ,  -58, -10))]
    ], dtype=[('f0', [('f0', '<f4'), ('f1', '<i8'), ('f2', '<i8')]), ('f1', '<u2'), ('f2', [('f0', '<f4'), ('f1', '<i8'), ('f2', '<i8')])])

    assert(equal(random, expected))

#  Array members are also supported!
struct_array_member = np.dtype([
    ('i', np.int8),
    ('a', (simple_struct, 3))
])

struct_array_member_param = {
    'i' : 42,
    'a' : simple_struct_param
}

def test_struct_array_member():
    random = rvg.random(struct_array_member, struct_array_member_param, (3, 2))
    expected = np.array([
        [(-38, [( -1.718677 ,   50, -30), (-13.756056 ,  -66, -11), ( -4.3921795,  -33,  28)]),
         ( -2, [(  5.7406025,  102,  16), (  5.6413603,  112, -15), (  3.1041248,  -77,  23)])],
        [(-15, [( -7.659459 ,  124,  -1), (  2.0822766,  -33,   2), ( -3.9804862,    3,  19)]),
         (-36, [( 16.038212 ,   93,  14), ( 11.8630705,  100, -37), (  7.5388036,   22, -15)])],
        [( 30, [( -8.976513 ,  102, -15), ( -8.293677 ,  108,   1), (-15.6252575,   14,  41)]),
         ( 29, [(  7.162538 ,   42, -13), (-13.2297125, -100,  19), ( -2.062559 ,  -93,  32)])]
    ], dtype=[('i', 'i1'), ('a', [('f0', '<f4'), ('f1', '<i8'), ('f2', '<i8')], (3,))])

    assert(equal(random, expected))
