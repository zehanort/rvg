import pytest
import numpy as np
import rvg

def struct(t, dtype):
    return np.array([t], dtype=dtype)[0]

def equal(a, b):
    print(a)
    print(b)
    return (a == b).all()

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
    # Set seed for reproducibility
    np.random.seed(42)

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
    # Set seed for reproducibility
    np.random.seed(42)

    random = rvg.random(simple_struct, simple_struct_param)
    expected = struct(
            (-4.265636, -36, -28),
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
    # Set seed for reproducibility
    np.random.seed(42)

    random  = rvg.random(nested_struct, nested_struct_param, (5, 2))
    expected = np.array([
        [((  5.097867 ,   4, -6), 15, ( 15.03486  ,  -48, -39)),
         (( 39.092144 ,  35, 40), 14, (  2.1517994,   35,  11))],
        [(( 26.187643 , -16,  4),  2, ( -3.895839 ,  -79,  20)),
         (( 18.32085  ,   6, 26), 36, (-16.457148 ,  -25, -25))],
        [(( -7.7949004,  26,  7),  6, ( -9.14961  ,    3,   1)),
         (( -7.7963233,  12, 31), 20, ( -8.805134 , -127,  -9))],
        [((-13.573067 ,  20,  9),  8, (  6.2309594,  125,  31)),
         (( 34.104393 , -16, 41), 38, (  3.7398863,    5,  19))],
        [(( 18.465786 ,   3, 24), 17, ( 11.328627 ,  -75, -29)),
         (( 24.776281 ,  15, 10),  3, (-11.105601 ,  -23,   5))]
    ], dtype=[('f0', [('f0', '<f4'), ('f1', '<i8'), ('f2', '<i8')]),
              ('f1', '<u2'),
              ('f2', [('f0', '<f4'), ('f1', '<i8'), ('f2', '<i8')])])

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
    # Set seed for reproducibility
    np.random.seed(42)

    random = rvg.random(struct_array_member, struct_array_member_param, (3, 2))
    expected = np.array([
        [(  9, [(-13.600853  , -114, -25), ( -1.3855376 ,   61, -39), ( -5.6539073 ,   61,  17)]),
         (-28, [(-12.142529  ,   46, -29), (  5.130208  ,   61, -34), (-15.082006  ,  -78,  10)])],
        [( 29, [(  7.5479584 ,  -21, -41), ( 14.910792  ,  -74,  41), (-16.973522  ,  115,  17)]),
         ( 18, [( 16.735193  ,  -65,  28), (  3.9943714 ,  120,   1), (  3.7962074 ,    2, -35)])],
        [(-22, [(-16.759747  ,  100,   4), (-16.215878  ,  -78,  -8), (  0.84233844,    6,  35)]),
         ( 40, [( -3.404727  , -108,  38), (-15.413367  ,  -56,  -7), ( 16.107687  ,   38,   7)])]
    ], dtype=[('i', 'i1'), ('a', [('f0', '<f4'), ('f1', '<i8'), ('f2', '<i8')], (3,))])

    assert(equal(random, expected))
